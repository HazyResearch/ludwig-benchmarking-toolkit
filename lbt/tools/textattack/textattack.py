import inspect
import sys
import os
import pandas as pd
from pandas.core.common import SettingWithCopyWarning

import warnings

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)


from ludwig.api import LudwigModel

from textattack.attack_recipes import AttackRecipe
from textattack.attack_results import (
    MaximizedAttackResult,
    SuccessfulAttackResult,
)
from textattack.augmentation import Augmenter
from textattack.models.wrappers import ModelWrapper

from lbt.tools.utils import get_dataset_features


ATTACKRECIPE_REGISTRY = {}
AUGMENTATIONRECIPE_REGISTRY = {}

for key, obj in inspect.getmembers(sys.modules["textattack.attack_recipes"]):
    if inspect.isclass(obj):
        if issubclass(obj, AttackRecipe) and key != "AttackRecipe":
            ATTACKRECIPE_REGISTRY[key] = obj


for key, obj in inspect.getmembers(sys.modules["textattack.augmentation"]):
    if inspect.isclass(obj):
        if issubclass(obj, Augmenter) and key != "Augmenter":
            AUGMENTATIONRECIPE_REGISTRY[key] = obj


class CustomLudwigModelWrapper(ModelWrapper):
    def __init__(
        self,
        path_to_model: str,
        input_feature_name: str,
        output_feature_name: str,
    ):
        self.model = LudwigModel.load(path_to_model)
        self.input_feature_name = input_feature_name
        self.output_feature_name = output_feature_name

    def __call__(self, text_list):
        input_text_df = pd.DataFrame(
            text_list, columns=[self.input_feature_name]
        )
        model_outputs = self.model.predict(input_text_df)
        pred_outputs = model_outputs[0]
        columns = [
            col
            for col in pred_outputs.columns
            if self.output_feature_name in col
        ]
        preds = pred_outputs[columns].iloc[:, 1:-1].to_numpy()
        return preds


def load_dataset(
    path_to_dataset: str, input_feature_name: str, output_feature_name: str
):
    dataset = pd.read_csv(path_to_dataset)
    dataset = dataset[0:10]
    if "split" not in dataset.columns:
        warnings.warn(
            "Dataset doesn't contain split column. Attacking entire dataset"
        )
        test_split = dataset[[input_feature_name, output_feature_name]]
    else:
        test_split = dataset[dataset["split"] == 2][
            [input_feature_name, output_feature_name]
        ]
    return test_split


def build_custom_ta_dataset(
    path_to_dataset: str, input_feature_name: str, output_feature_name: str
):
    dataset = load_dataset(
        path_to_dataset, input_feature_name, output_feature_name
    )
    dataset[output_feature_name] = (
        dataset[output_feature_name].astype(int).tolist()
    )
    tupelize = dataset.to_records(index=False)
    return list(tupelize)


def attack(
    dataset_name: str,
    path_to_dataset: str,
    path_to_model: str,
    input_feature_name: str = None,
    output_feature_name: str = None,
    attack_recipe: str = "DeepWordBugGao2018",
    output_directory: str = "./",
):
    if input_feature_name is None or output_feature_name is None:
        (input_features, output_features) = get_dataset_features(dataset_name)
        input_feature_name = input_features[0]
        output_feature_name = output_features[0]

    custom_model = CustomLudwigModelWrapper(
        path_to_model=path_to_model,
        input_feature_name=input_feature_name,
        output_feature_name=output_feature_name,
    )

    custom_datset = build_custom_ta_dataset(
        path_to_dataset=path_to_dataset,
        input_feature_name=input_feature_name,
        output_feature_name=output_feature_name,
    )

    if attack_recipe not in ATTACKRECIPE_REGISTRY.keys():
        raise ValueError(
            f"{attack_recipe} not valid.\n"
            f"Please check ATTACKRECIPE_REGISTRY to see valid recipes"
        )
    attack = ATTACKRECIPE_REGISTRY[attack_recipe].build(custom_model)
    results_iterable = attack.attack_dataset(custom_datset)

    results = {
        "original_text": [],
        "perturbed_text": [],
        "original_result": [],
        "original_confidence_score": [],
        "perturbed_result": [],
        "perturbed_confidence_score": [],
        "success": [],
    }

    for result in results_iterable:
        results["original_text"].append(result.original_text())
        results["perturbed_text"].append(result.perturbed_text())
        results["original_result"].append(
            result.original_result.raw_output.argmax().item()
        )
        results["original_confidence_score"].append(
            result.original_result.raw_output[
                result.original_result.raw_output.argmax()
            ].item()
        )
        results["perturbed_result"].append(
            result.perturbed_result.raw_output.argmax().item()
        )
        results["perturbed_confidence_score"].append(
            result.perturbed_result.raw_output[
                result.perturbed_result.raw_output.argmax()
            ].item()
        )
        if type(result) in [SuccessfulAttackResult, MaximizedAttackResult]:
            results["success"].append(1)
        else:
            results["success"].append(0)

    results_df = pd.DataFrame.from_dict(results)
    output_path = os.path.join(
        output_directory, f"{dataset_name}_{attack_recipe}.csv"
    )
    results_df.to_csv(output_path)
    return results_df


# TODO (ASN) : fix manual input/output feature dependence
def augment(
    dataset_name: str,
    path_to_dataset: str,
    input_feature_name: str = None,
    output_feature_name: str = None,
    augmenter_name: str = "CharSwapAugmenter",
    pct_words_to_swap: float = 0.1,
    transformations_per_example: int = 1,
    save_path: str = "augmented_ds.csv",
    save=True,
):
    if input_feature_name is None or output_feature_name is None:
        (input_features, output_features) = get_dataset_features(dataset_name)
        input_feature_name = input_features[0]
        output_feature_name = output_features[0]

    dataset = load_dataset(
        path_to_dataset, input_feature_name, output_feature_name
    )

    if augmenter_name not in AUGMENTATIONRECIPE_REGISTRY.keys():
        raise ValueError(
            f"{augmenter_name} not valid.\n"
            f"Please check AUGMENTATIONRECIPE_REGISTRY to see valid recipes"
        )

    augmenter = AUGMENTATIONRECIPE_REGISTRY[augmenter_name](
        pct_words_to_swap=pct_words_to_swap,
        transformations_per_example=transformations_per_example,
    )

    text_df = (
        dataset[[input_feature_name]]
        .applymap(augmenter.augment)
        .applymap(lambda sent: sent[0])
    )

    augmented_ds = dataset
    augmented_ds.loc[:, input_feature_name] = text_df[input_feature_name]

    if save:
        augmented_ds.to_csv(save_path)

    return augmented_ds
