import argparse
import datetime
import logging
import os
import fsspec
from fsspec.core import split_protocol

import pickle
import socket
from typing import Union
from collections import defaultdict

import pdb
import numpy as np
import ray

import globals
from .build_def_files import *
from database import save_results_to_es
from ludwig.hyperopt.run import hyperopt
from lbt.utils.experiment_utils import *
from lbt.datasets import DATASET_REGISTRY
from ludwig.utils.fs_utils import makedirs, upload_output_file

hostname = socket.gethostbyname(socket.gethostname())


def download_data(cache_dir=None, datasets: list = None):
    """ Returns files paths for all datasets """
    data_file_paths = {}
    for dataset in datasets:
        # if dataset in dataset_metadata.keys():
        if dataset in list(DATASET_REGISTRY.keys()):
            data_class = dataset_metadata[dataset]["data_class"]
            data_path = download_dataset(data_class, cache_dir)
            process_dataset(data_path)
            data_file_paths[dataset] = data_path
        else:
            raise ValueError(
                f"{dataset} is not a valid dataset."
                "for list of valid dataets see: "
                "python experiment_driver.py -h"
            )
    return data_file_paths


def resume_training(model_config: dict, output_dir):
    results, metrics, params = collect_completed_trial_results(output_dir)
    original_num_samples = model_config["hyperopt"]["sampler"]["num_samples"]
    new_num_samples = max(original_num_samples - len(metrics), 0)
    model_config["hyperopt"]["sampler"]["search_alg"][
        "points_to_evaluate"
    ] = params
    model_config["hyperopt"]["sampler"]["search_alg"][
        "evaluated_rewards"
    ] = metrics
    model_config["hyperopt"]["sampler"]["num_samples"] = new_num_samples
    return model_config, results


#@ray.remote(num_cpus=0, resources={f"node:{hostname}": 0.001})
def run_hyperopt_exp(
    experiment_attr: dict,
    is_resume_training: bool = False,
    runtime_env: str = "local",
) -> int:

    dataset = experiment_attr["dataset"]
    encoder = experiment_attr["encoder"]
    model_config = experiment_attr["model_config"]

    # the following are temp solutions for issues in Ray
    if runtime_env == "local":
        # temp solution to ray problems
        os.environ["TUNE_PLACEMENT_GROUP_AUTO_DISABLED"] = "1"
    os.environ["TUNE_PLACEMENT_GROUP_CLEANUP_DISABLED"] = "1"

    # try:
    start = datetime.datetime.now()

    tune_executor = model_config["hyperopt"]["executor"]["type"]

    num_gpus = 0
    try:
        num_gpus = model_config["hyperopt"]["executor"][
            "gpu_resources_per_trial"
        ]
    except:
        pass

    if tune_executor == "ray" and runtime_env == "gcp":

        if (
            "kubernetes_namespace"
            not in model_config["hyperopt"]["executor"].keys()
        ):
            raise ValueError(
                "Please specify the kubernetes namespace of the Ray cluster"
            )

    if tune_executor == "ray" and runtime_env == "local":
        if (
            "kubernetes_namespace"
            in model_config["hyperopt"]["executor"].keys()
        ):
            raise ValueError(
                "You are running locally. "
                "Please remove the kubernetes_namespace param in hyperopt_config.yaml"
            )

    gpu_list = None
    if tune_executor != "ray":
        gpu_list = get_gpu_list()
        if len(gpu_list) > 0:
            num_gpus = 1

    new_model_config = copy.deepcopy(experiment_attr["model_config"])
    existing_results = None
    """if is_resume_training:
        new_model_config, existing_results = resume_training(
            new_model_config, experiment_attr["output_dir"]
        )"""

    # dataset = "s3://experiments.us-west-2.predibase.com/tabular-experiments/datasets/sarcos_1.0/processed/sarcos.csv"
    hyperopt_results = hyperopt(
        new_model_config,
        dataset=experiment_attr["dataset_path"],
        model_name=experiment_attr["model_name"],
        gpus=gpu_list,
        output_directory=experiment_attr["output_dir"],
    )
    #hyperopt_results = hyperopt_results.experiment_analysis.results_df.values.tolist()
    
    if existing_results is not None:
        hyperopt_results.extend(existing_results)
        hyperopt_results.sort(key=lambda result: result["metric_score"])

    logging.info(
        "time to complete: {}".format(datetime.datetime.now() - start)
    )

    # Save hyperopt results tor remote fs
    try:
        hyperopt_results_url = os.path.join(
                    experiment_attr["output_dir"],
                    f"{dataset}_{encoder}_hyperopt_results.pkl",
                ) 
        makedirs(hyperopt_results_url, exists_ok=True)
        with upload_output_file(hyperopt_results_url) as local_file:
            pickle.dump(
                hyperopt_results,
                open(local_file,"wb"),
            )
    except:
        pass

    # save lbt output w/additional metrics computed locall
    try:
        results_w_additional_metrics = compute_additional_metadata(
            experiment_attr, hyperopt_results, tune_executor
        )
        pickle.dump(
            results_w_additional_metrics,
            open(
                os.path.join(
                    experiment_attr["output_dir"],
                    f"{dataset}_{encoder}_hyperopt_results_w_lbt_metrics.pkl",
                ),
                "wb",
            ),
        )
    except:
        pass

    # create .completed file to indicate that experiment is completed
    try:
        _ = open(
            os.path.join(experiment_attr["output_dir"], ".completed"), "wb"
        )
    except:
        pass

    logging.info(
        "time to complete: {}".format(datetime.datetime.now() - start)
    )

    # save output to db
    if experiment_attr["elastic_config"]:
        try:
            save_results_to_es(
                experiment_attr,
                hyperopt_results,
                tune_executor=tune_executor,
                top_n_trials=experiment_attr["top_n_trials"],
                runtime_env="local",
                num_gpus=num_gpus,
            )
        except:
            logging.warning("Not all files were uploaded to elastic db!")
    return 1
    # except:
    #    logging.warning("Error running experiment...not completed")
    #    return 0


def run_experiments(
    data_file_paths: dict,
    config_files: dict,
    top_n_trials: int,
    elastic_config=None,
    run_environment: str = "local",
    resume_existing_exp: bool = False,
):
    logging.info("Running hyperopt experiments...")
    # check if overall experiment has already been run
    if os.path.exists(
        os.path.join(globals.EXPERIMENT_OUTPUT_DIR, ".completed")
    ):
        logging.info("Experiment is already completed!")
        return

    completed_runs, experiment_queue = [], []
    for dataset_name, file_path in data_file_paths.items():
        logging.info("Dataset: {}".format(dataset_name))

        for model_config_path in config_files[dataset_name]:
            config_name = model_config_path.split("/")[-1].split(".")[0]
            dataset = config_name.split("_")[1]
            encoder = "_".join(config_name.split("_")[2:])
            experiment_name = dataset + "_" + encoder

            logging.info("Experiment: {}".format(experiment_name))

            output_dir = os.path.join(
                globals.EXPERIMENT_OUTPUT_DIR, experiment_name
            )

            protocol, _ = split_protocol(output_dir)
            if protocol is not None:
                # makedirs(output_dir)
                with fsspec.open(output_dir, mode="wb") as f:
                    pass
            else:
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)

            output_dir = os.path.join(
                globals.EXPERIMENT_OUTPUT_DIR, experiment_name
            )

            if not os.path.exists(os.path.join(output_dir, ".completed")):

                model_config = load_yaml(model_config_path)
                experiment_attr = defaultdict()
                experiment_attr = {
                    "model_config": copy.deepcopy(model_config),
                    "dataset_path": file_path,
                    "top_n_trials": top_n_trials,
                    "model_name": config_name,
                    "output_dir": output_dir,
                    "encoder": encoder,
                    "dataset": dataset,
                    "elastic_config": elastic_config,
                }
                if run_environment == "local":
                    completed_runs.append(
                        run_hyperopt_exp(
                            experiment_attr,
                            resume_existing_exp,
                            run_environment,
                        )
                    )

                experiment_queue.append(experiment_attr)
            else:
                logging.info(
                    f"The {dataset} x {encoder} exp. has already completed!"
                )

    if run_environment != "local":
        completed_runs = ray.get(
            [
                # ray.remote(num_cpus=0, resources={f"node:{hostname}": 0.001})(
                run_hyperopt_exp.remote(
                    exp, resume_existing_exp, run_environment)
                for exp in experiment_queue
            ]
        )

    if len(completed_runs) == len(experiment_queue):
        # create .completed file to indicate that entire hyperopt experiment
        # is completed
        _ = open(
            os.path.join(globals.EXPERIMENT_OUTPUT_DIR, ".completed"), "wb"
        )
    else:
        logging.warning("Not all experiments completed!")


def reproduce_experiment(
    model,
    dataset,
    data_file_paths,
    elastic_config=None,
    experiment_to_replicate=None,
    run_environment: str = "local",
):
    experiment_config = load_yaml(experiment_to_replicate)
    experiment_name = dataset + "_" + model
    for dataset_name, file_path in data_file_paths.items():

        output_dir = os.path.join(
            globals.EXPERIMENT_OUTPUT_DIR, experiment_name
        )

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        output_dir = os.path.join(
            globals.EXPERIMENT_OUTPUT_DIR, experiment_name
        )

        experiment_attr = defaultdict()
        experiment_attr = {
            "model_config": experiment_config,
            "dataset_path": file_path,
            "model_name": model,
            "output_dir": output_dir,
            "encoder": model,
            "dataset": dataset,
            "elastic_config": elastic_config,
        }
        run_hyperopt_exp(
            experiment_attr,
            False,
            run_environment,
        )


def experiment(
    models: Union[str, list],
    datasets: Union[str, list],
    experiment_configs_dir: str = globals.EXPERIMENT_CONFIGS_DIR,
    experiment_output_dir: str = globals.EXPERIMENT_OUTPUT_DIR,
    datasets_cache_dir: str = globals.DATASET_CACHE_DIR,
    run_environment: str = "local",
    elastic_search_config: str = None,
    resume_existing_exp: bool = False,
):
    if isinstance(datasets, str):
        datasets = [datasets]
    data_file_paths = download_data(datasets_cache_dir, datasets)

    config_files = build_config_files()
    elastic_config = None
    if elastic_search_config is not None:
        elastic_config = load_yaml(elastic_search_config)

    if run_environment == "gcp":
        ray.init(address="auto")

    run_experiments(
        data_file_paths,
        config_files,
        top_n_trials=None,
        elastic_config=elastic_config,
        run_environment=run_environment,
        resume_existing_exp=resume_existing_exp,
    )
