import os
from typing import List, Union

import globals
import json
import pickle
from lbt.datasets import DATASET_REGISTRY
from ludwig.visualize import (
    compare_performance,
    hyperopt_report,
    learning_curves,
)


def hyperopt_viz(
    hyperopt_stats_path: str = None,
    dataset_name: str = None,
    model_name: str = None,
    output_dir: str = None,
):
    """
    Produces a report about hyperparameter optimization.
    Creating one graph per hyperparameter to show the distribution of results
    and one additional graph of pairwise hyperparameters interactions
    """

    if hyperopt_stats_path:
        return hyperopt_report(
            hyperopt_stats_path=hyperopt_stats_path,
            output_directory=output_dir,
        )
    elif dataset_name and model_name:
        if dataset_name not in DATASET_REGISTRY.keys():
            raise ValueError("The specified dataset is not valid")
        elif model_name not in globals.ENCODER_HYPEROPT_FILENAMES.keys():
            raise ValueError("The specified model name is not valid")

        exp_name = "_".join([dataset_name, model_name])
        experiment_folder = os.path.join(
            globals.EXPERIMENT_OUTPUT_DIR, exp_name
        )

        hyperopt_stats_json = os.path.join(
            experiment_folder,
            "hyperopt_statistics.json",
        )
        json_file = json.load(open(hyperopt_stats_json, "rb"))

        # decode json
        hyperopt_results = []
        for result in json_file["hyperopt_results"]:
            for key, val in result.items():
                try:
                    val = json.loads(val)
                    result[key] = val
                except:
                    pass
            hyperopt_results.append(result)
        json_file["hyperopt_results"] = hyperopt_results

        with open(
            os.path.join(
                experiment_folder, "hyperopt_statistics_decoded.json"
            ),
            "w",
        ) as outfile:
            json.dump(json_file, outfile)

        hyperopt_stats_path = os.path.join(
            experiment_folder,
            "hyperopt_statistics_decoded.json",
        )
        return hyperopt_report(
            hyperopt_stats_path=hyperopt_stats_path,
            output_directory=output_dir,
        )
    raise ValueError(
        "Please specify either a path to the hyperopt output stats json file"
        "or the dataset and model name of the experiment"
    )


def learning_curves_viz(
    model_name: str,
    dataset_name: str,
    output_feature_name: str,
    output_directory=None,
    file_format="pdf",
):
    """
    Visualize how model metrics change over training and validation data
    epochs.
    """

    exp_name = "_".join([dataset_name, model_name])
    experiment_folder = os.path.join(globals.EXPERIMENT_OUTPUT_DIR, exp_name)

    results_file = os.path.join(
        experiment_folder, f"{exp_name}_hyperopt_results.pkl"
    )
    hyperopt_results = pickle.load(open(results_file, "rb"))

    training_stats = []
    experiment_ids = []

    for model_results in hyperopt_results:
        training_stats.append(json.loads(model_results["training_stats"]))
        experiment_ids.append(model_results["experiment_id"])

    return learning_curves(
        train_stats_per_model=training_stats,
        output_feature_name=output_feature_name,
        model_names=experiment_ids,
        output_directory=output_directory,
        file_format=file_format,
    )


def compare_performance_viz(
    model_name: str,
    dataset_name: str,
    output_feature_name: str,
    output_directory=None,
    file_format="pdf",
):
    """  Barplot visualization for each overall metric """

    exp_name = "_".join([dataset_name, model_name])
    experiment_folder = os.path.join(globals.EXPERIMENT_OUTPUT_DIR, exp_name)

    results_file = os.path.join(
        experiment_folder, f"{exp_name}_hyperopt_results.pkl"
    )
    hyperopt_results = pickle.load(open(results_file, "rb"))

    eval_stats = []
    experiment_ids = []

    for model_results in hyperopt_results:
        eval_stats.append(json.loads(model_results["eval_stats"]))
        experiment_ids.append(model_results["experiment_id"])

    return compare_performance(
        test_stats_per_model=eval_stats,
        output_feature_name=output_feature_name,
        model_names=experiment_ids,
        output_directory=output_directory,
        file_format=file_format,
    )