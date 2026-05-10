import os
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from lbt.datasets import DATASET_REGISTRY
from lbt.tools.robustnessgym import RGSUBPOPULATION_REGISTRY
from ludwig.api import LudwigModel
from lbt.tools.utils import get_dataset_features

from robustnessgym import Dataset, Identifier, Spacy
from robustnessgym.core.testbench import DevBench

from .base_subpopulation import BaseSubpopulation

OUTPUT_FEATURES = None


def get_dataset_with_predictions(
    dataset: pd.DataFrame,
    models: dict,
    output_features: list,
):
    for model_name, path_to_model in models.items():
        model = LudwigModel.load(model_dir=path_to_model)
        (predictions, output_directory) = model.predict(dataset)
        for output_feat in output_features:
            dataset[f"{model_name}_{output_feat}_pred"] = (
                predictions[f"{output_feat}_predictions"]
                .astype(float)
                .tolist()
            )
            dataset.rename(
                {output_feat: f"{output_feat}_label"}, axis=1, inplace=True
            )
    return dataset


def accuracy_eval_fn(model, dataset):
    global OUTPUT_FEATURES
    output_feat_accuracy = []
    # aggregate accuracy over all output features
    for output_feat in OUTPUT_FEATURES:
        accuracy = np.mean(
            np.array(dataset[f"{model}_{output_feat}_pred"])
            == (np.array(dataset[f"{output_feat}_label"]))
        )
        output_feat_accuracy.append(accuracy)
    return np.mean(output_feat_accuracy)


def RG(
    dataset_name: str,
    models: dict,
    path_to_dataset: str,
    subpopulations: list,
    output_directory: str,
    input_features: Union[str, list] = None,
    output_features: Union[str, list] = None,
    output_report_name: str = "rg_report.png",
):
    """
    Runs RG  evaluation on dataset across specified models

    # Inputs
    :param dataset_name: (str) name of dataset
    :param models: (dict) mapping between model name and saved model directory
    :param path_to_dataset: (str) location of dataset
    :param input_features: (list or str) names of input feature
    :param output_features: (list or str) names of output feature
    :param subpopulations: (list) subpopulations to evaluate model performance
    :param output_directory: (str) location to save all outputs of RG analysis
    :param output_report_name: (str) name of generated file


    # Return
    :return: (pd.DataFrame) performance metrics from RG analysis
    """

    # first check if slices are valid
    for subpop in subpopulations:
        if subpop not in RGSUBPOPULATION_REGISTRY.keys():
            raise ValueError(
                f"{subpop} is not in the list of supported RG Subpopulations\n"
                f"Please see lbt.tools.robustnessgym.RGSUBPOPULATION_REGISTRY for available subpopulations"
            )

    # if user has not provided input/output feature info, collect it manually
    if input_features is None or output_features is None:
        (input_features, output_features) = get_dataset_features(dataset_name)

    else:
        if isinstance(input_features, str):
            input_features = [input_features]
        if isinstance(output_features, str):
            output_features = [output_features]

    global OUTPUT_FEATURES
    OUTPUT_FEATURES = output_features

    # load data
    # TODO (ASN): fix logic for extracting eval set
    dataset = pd.read_csv(path_to_dataset)

    # get preds
    dataset = get_dataset_with_predictions(dataset, models, output_features)
    # caste as RG Dataset
    dataset = Dataset.from_pandas(dataset, Identifier(dataset_name))

    # initialize spacy
    spacy = Spacy()
    dataset = spacy(dataset, input_features)

    # for each subopulation, get subpopulation functions
    selected_subpopulations = []
    for subpop in subpopulations:
        if issubclass(RGSUBPOPULATION_REGISTRY[subpop], BaseSubpopulation):
            subpops = RGSUBPOPULATION_REGISTRY[subpop]().get_subpops(spacy)
        else:
            subpops = RGSUBPOPULATION_REGISTRY[subpop]()
        if not isinstance(subpops, list):
            subpops = [subpops]
        selected_subpopulations.extend(subpops)

    # for each subpopulation get slcies
    slices = []
    for subpop in selected_subpopulations:
        slices.extend(subpop(dataset, input_features)[0])

    # build test bench
    dataset_db = DevBench(dataset)
    # add slices to test bench
    dataset_db.add_slices(slices)

    dataset_db.add_aggregators(
        {
            model: {"accuracy": partial(accuracy_eval_fn, model)}
            for model in models.keys()
        }
    )
    # compute metrics
    metrics = dataset_db.metrics

    # save metrics dataframe
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(os.path.join(output_directory, f"{dataset_name}_rg.csv"))

    # create report
    dataset_db.create_report().figure().write_image(
        output_report_name, engine="kaleido"
    )
    return metrics
