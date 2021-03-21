import argparse
import datetime
import json
import logging
import os
import pickle
import socket
import sys
from collections import defaultdict
from copy import deepcopy
from multiprocessing import Pool

import GPUtil
import numpy as np
import ray
import yaml

import globals
from build_def_files import *
from database import Database, save_results_to_es
from ludwig.hyperopt.run import hyperopt
from utils.experiment_utils import *
from utils.metadata_utils import append_experiment_metadata

logging.basicConfig(
    # filename="elastic-exp.log",
    format=logging.basicConfig(
        format="[\N{books} LUDWIG-BENCH \N{books}] => %(levelname)s::%(message)s",
        level=logging.DEBUG,
    ),
    level=logging.DEBUG,
)

hostname = socket.gethostbyname(socket.gethostname())


def download_data(cache_dir=None, datasets: list = None):
    """ Returns files paths for all datasets """
    data_file_paths = {}
    for dataset in datasets:
        if dataset in dataset_metadata.keys():
            data_class = dataset_metadata[dataset]["data_class"]
            data_path = download_dataset(data_class, cache_dir)
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


@conditional_decorator(
    ray.remote(num_cpus=0, resources={f"node:{hostname}": 0.001}),
    lambda runtime_env: runtime_env != "local",
    globals.RUNTIME_ENV,
)
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

    try:
        start = datetime.datetime.now()

        combined_ds, train_set, val_set, test_set = None, None, None, None
        combined_ds, train_set, val_set, test_set = process_dataset(
            experiment_attr["dataset_path"]
        )

        tune_executor = model_config["hyperopt"]["executor"]["type"]

        if tune_executor == "ray" and runtime_env == "gcp":
            if (
                "kubernetes_namespace"
                not in model_config["hyperopt"]["executor"].keys()
            ):
                raise RuntimeError(
                    "Please specify the kubernetes namespace of the Ray cluster"
                )

        gpu_list = None
        if tune_executor != "ray":
            gpu_list = get_gpu_list()

        new_model_config = copy.deepcopy(experiment_attr["model_config"])
        existing_results = None
        if is_resume_training:
            new_model_config, existing_results = resume_training(
                new_model_config, experiment_attr["output_dir"]
            )

        hyperopt_results = hyperopt(
            new_model_config,
            dataset=combined_ds,
            training_set=train_set,
            validation_set=val_set,
            test_set=test_set,
            model_name=experiment_attr["model_name"],
            gpus=gpu_list,
            output_directory=experiment_attr["output_dir"],
        )

        if existing_results is not None:
            hyperopt_results.extend(existing_results)
            hyperopt_results.sort(key=lambda result: result["metric_score"])

        logging.info(
            "time to complete: {}".format(datetime.datetime.now() - start)
        )

        # Save output locally
        try:
            pickle.dump(
                hyperopt_results,
                open(
                    os.path.join(
                        experiment_attr["output_dir"],
                        f"{dataset}_{encoder}_hyperopt_results.pkl",
                    ),
                    "wb",
                ),
            )
            # create .completed file to indicate that experiment is completed
            _ = open(
                os.path.join(experiment_attr["output_dir"], ".completed"), "wb"
            )

        except FileNotFoundError:
            pass

        # save output to db
        if experiment_attr["elastic_config"]:
            try:
                save_results_to_es(
                    experiment_attr,
                    hyperopt_results,
                    tune_executor=tune_executor,
                    top_n_trials=experiment_attr["top_n_trials"],
                )
            except:
                logging.warning("Not all files were uploaded to elastic db!")
        return 1
    except:
        logging.warning("Error running experiment...not completed")
        return 0


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
                else:
                    experiment_queue.append(experiment_attr)
            else:
                logging.info(
                    "The {dataset} x {encoder} exp. has already completed!"
                )

    if run_environment != "local":
        completed_runs = ray.get(
            [
                run_hyperopt_exp.remote(
                    exp, resume_existing_exp, run_environment
                )
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


def main():
    parser = argparse.ArgumentParser(
        description="Ludwig-Bench experiment driver script",
    )

    parser.add_argument(
        "-hcd",
        "--hyperopt_config_dir",
        help="directory to save all model config",
        type=str,
        default=EXPERIMENT_CONFIGS_DIR,
    )

    parser.add_argument(
        "--resume_existing_exp",
        help="resume a previously stopped experiment",
        type=bool,
        default=False,
    )

    parser.add_argument(
        "-eod",
        "--experiment_output_dir",
        help="directory to save hyperopt runs",
        type=str,
        default=EXPERIMENT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--datasets",
        help="list of datasets to run experiemnts on",
        nargs="+",
        choices=[
            "all",
            "agnews",
            "amazon_reviews",
            "amazon_review_polarity",
            "dbpedia",
            "ethos_binary",
            "goemotions",
            "irony",
            "sst2",
            "sst5",
            "yahoo_answers",
            "yelp_review_polarity",
            "yelp_reviews",
            "smoke",
        ],
        default=None,
        required=True,
    )
    parser.add_argument(
        "-re",
        "--run_environment",
        help="environment in which experiment will be run",
        choices=["local", "gcp"],
        default="local",
    )
    parser.add_argument(
        "-esc",
        "--elasticsearch_config",
        help="path to elastic db config file",
        type=str,
        default=None,
    )

    parser.add_argument(
        "-dcd",
        "--dataset_cache_dir",
        help="path to cache downloaded datasets",
        type=str,
        default=None,
    )

    # list of encoders to run hyperopt search over :
    # default is 23 ludwig encoders
    parser.add_argument(
        "-cel",
        "--custom_encoders_list",
        help="list of encoders to run hyperopt experiments on. \
            The default setting is to use all 23 Ludwig encoders",
        nargs="+",
        choices=[
            "all",
            "bert",
            "rnn",
            "stacked_parallel_cnn",
            "roberta",
            "distilbert",
            "electra",
            "t5",
        ],
        default="all",
    )

    parser.add_argument(
        "-topn",
        "--top_n_trials",
        help="top n trials to save model performance for.",
        type=int,
        default=None,
    )

    parser.add_argument("-smoke", "--smoke_tests", type=bool, default=False)

    args = parser.parse_args()
    set_globals(args)

    logging.info("GPUs {}".format(os.system("nvidia-smi -L")))
    if args.smoke_tests:
        data_file_paths = SMOKE_DATASETS
    else:
        data_file_paths = download_data(args.dataset_cache_dir, args.datasets)
        logging.info("Datasets succesfully downloaded...")

    config_files = build_config_files()
    logging.info("Experiment configuration files built...")

    elastic_config = None
    if args.elasticsearch_config is not None:
        elastic_config = load_yaml(args.elasticsearch_config)

    if args.run_environment == "gcp":
        ray.init(address="auto")

    run_experiments(
        data_file_paths,
        config_files,
        top_n_trials=args.top_n_trials,
        elastic_config=elastic_config,
        run_environment=args.run_environment,
        resume_existing_exp=args.resume_existing_exp,
    )


if __name__ == "__main__":
    main()
