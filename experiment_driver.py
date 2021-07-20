import argparse
import datetime
import logging

import ray
import globals

from lbt.utils.experiment_utils import set_globals, load_yaml
from lbt.experiments import (
    run_experiments,
    reproduce_experiment,
    download_data,
)
from lbt.datasets import DATASET_REGISTRY
from lbt.experiments import (
    run_experiments,
    reproduce_experiment,
    download_data,
)
import lbt.build_def_files
from lbt.build_def_files import build_config_files

logging.basicConfig(
    format=logging.basicConfig(
        format="[\N{books} LUDWIG-BENCHMARKING-TOOLKIT \N{books}] => %(levelname)s::%(message)s",
        level=logging.DEBUG,
    ),
    level=logging.DEBUG,
)


def main():
    parser = argparse.ArgumentParser(
        description="Ludwig Benchmarking Toolkit experiment driver script",
    )

    parser.add_argument(
        "-hcd",
        "--hyperopt_config_dir",
        help="directory to save all model config",
        type=str,
        default=globals.EXPERIMENT_CONFIGS_DIR,
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
        default=globals.EXPERIMENT_OUTPUT_DIR,
    )

    parser.add_argument(
        "--datasets",
        help="list of datasets to run experiemnts on",
        nargs="+",
        choices=list(DATASET_REGISTRY.keys()),
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
        default=globals.DATASET_CACHE_DIR,
    )

    # list of encoders to run hyperopt search over :
    # default is 23 ludwig encoders
    parser.add_argument(
        "-mel",
        "--custom_model_list",
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
            "resnet",
            "stacked_cnn",
            "t5",
            "tabtrans",
            "tabconcat",
            "tabnet"
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

    parser.add_argument(
        "-reproduce",
        "--experiment_to_reproduce",
        help="path to LBT experiment config to reproduce and experiment",
        type=str,
        default=None,
    )

    args = parser.parse_args()
    set_globals(args)

    data_file_paths = download_data(args.dataset_cache_dir, args.datasets)
    logging.info("Datasets succesfully downloaded...")

    config_files = build_config_files()
    logging.info("Experiment configuration files built...")

    elastic_config = None
    if args.elasticsearch_config is not None:
        elastic_config = load_yaml(args.elasticsearch_config)

    experiment_config = None
    if args.experiment_to_reproduce is not None:
        experiment_config = load_yaml(args.experiment_to_reproduce)

    if args.run_environment == "gcp":
        ray.init(address="auto")

    if experiment_config:
        reproduce_experiment(
            model=args.custom_model_list[0],
            dataset=args.datasets[0],
            data_file_paths=data_file_paths,
            experiment_to_replicate=args.experiment_to_reproduce,
            run_environment=args.run_environment,
        )
    else:
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
