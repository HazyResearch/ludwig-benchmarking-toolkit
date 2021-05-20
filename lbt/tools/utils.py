from utils.experiment_utils import load_yaml
from globals import DATASET_METADATA_FILE
from lbt.datasets import DATASET_REGISTRY


def get_dataset_features(dataset_name):

    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"{dataset_name} not found in dataset registry\n"
            f"Please check that it has been properly added"
        )

    dataset_metadata = load_yaml(DATASET_METADATA_FILE)
    for ds, ds_metadata in dataset_metadata.items():
        if dataset_name == ds_metadata["data_class"]:
            input_features = [
                input_feat["name"]
                for input_feat in ds_metadata["input_features"]
            ]
            output_features = [
                output_feat["name"]
                for output_feat in ds_metadata["output_features"]
            ]
            return (input_features, output_features)

    raise ValueError(
        f"{dataset_name} not found in {DATASET_METADATA_FILE}\n"
        f"Please check that it has been properly added"
    )
