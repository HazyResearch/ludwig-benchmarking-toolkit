import logging
import os
import pdb

from copy import deepcopy

import yaml

import globals
from globals import *
from lbt.utils.experiment_utils import load_yaml

template = load_yaml(CONFIG_TEMPLATE_FILE)
dataset_metadata = load_yaml(DATASET_METADATA_FILE)
hyperopt_config = load_yaml(HYPEROPT_CONFIG_FILE)


def insert_global_vars(config):
    """ replace global variable placeholders with respective values """
    for key, value in config.items():
        if type(value) != dict and value in vars(globals):
            config[key] = getattr(globals, value)


def build_config_files():
    config_fps = {}
    config = deepcopy(template)

    encoder_hyperopt_vals = []
    # select relevant encoders
    for encoder_filename in globals.ENCODER_FILE_LIST:
        with open(os.path.join(ENCODER_CONFIG_DIR, encoder_filename)) as f:
            encoder_hyperopt_params = yaml.load(f, Loader=yaml.SafeLoader)
            encoder_hyperopt_vals.append(encoder_hyperopt_params)

    # select relevant datasets
    selected_datasets = {}
    for dataset_name in globals.DATASETS_LIST:
        if dataset_name in dataset_metadata.keys():
            selected_datasets[dataset_name] = dataset_metadata[dataset_name]
        else:
            raise ValueError(
                "The dataset you provided is not available."
                "Please see list of available datasets here: "
                "python experiment_drivery.py --h"
            )

    config["hyperopt"].update(hyperopt_config)

    for dataset, metadata in selected_datasets.items():
        # each dataset will have a model specific config file
        config_fps[dataset] = []

        for idx, input_feature_name in enumerate(metadata["input_features"]):
            ipt_feat = deepcopy(config["input_features"][0])
            ipt_feat["name"] = input_feature_name["name"]
            ipt_feat["type"] = input_feature_name["type"]
            if idx == 0:
                config["input_features"] = [ipt_feat]
            else:
                config["input_features"].append(ipt_feat)
        for idx, output_feature_info in enumerate(metadata["output_features"]):
            out_feat = deepcopy(config["output_features"][0])
            out_feat["name"] = output_feature_info["name"]
            out_feat["type"] = output_feature_info["type"]
            if idx == 0:
                config["output_features"] = [out_feat]
            else:
                config["output_features"].append(out_feat)

        if len(metadata["output_features"]) > 1:
            config["hyperopt"]["output_feature"] = "combined"
        else:
            config["hyperopt"]["output_feature"] = metadata["output_features"][
                0
            ]["name"]

        input_feature_names = metadata["input_features"]
        output_feature_names = metadata["output_features"]

        for encoder_hyperopt_params in encoder_hyperopt_vals:
            curr_config = deepcopy(config)
            encoder_name = encoder_hyperopt_params["parameters"][
                "input_features.name.encoder"
            ]

            # update input and output parameters (not preprocessing)
            for idx in range(len(curr_config["input_features"])):
                curr_config["input_features"][idx].update(
                    encoder_hyperopt_params["input_features"][idx]
                )
                insert_global_vars(curr_config["input_features"][idx])

            for idx in range(len(curr_config["output_features"])):
                if "output_features" in encoder_hyperopt_params.keys():
                    curr_config["output_features"][idx].update(
                        encoder_hyperopt_params["output_features"][idx]
                    )
                    insert_global_vars(curr_config["output_features"][idx])

            # handle encoder specific preprocessing
            for idx in range(len(curr_config["input_features"])):
                preprocessing = curr_config["input_features"][idx][
                    "preprocessing"
                ]
                for key, _ in preprocessing.items():
                    preprocessing[key] = encoder_hyperopt_params[
                        "input_features"
                    ][idx]["preprocessing"][key]

            # handle encoder specific training params
            if "training" in encoder_hyperopt_params.keys():
                curr_config["training"].update(
                    encoder_hyperopt_params["training"]
                )

            def input_or_output_feature(param_key):
                if param_key.split(".")[0] == "input_features":
                    return True
                return False

            # handle encoder specific hyperopt
            input_encoder_hyperopt_params = {
                "parameters": {
                    input_feat["name"] + "." + key.split(".")[-1]: value
                    for input_feat in input_feature_names
                    for key, value in encoder_hyperopt_params[
                        "parameters"
                    ].items()
                    if key.split(".")[-1] != "encoder"
                    and input_or_output_feature(key)
                }
            }

            # handle encoder specific hyperopt
            output_encoder_hyperopt_params = {
                "parameters": {
                    output_feat["name"] + "." + key.split(".")[-1]: value
                    for output_feat in output_feature_names
                    for key, value in encoder_hyperopt_params[
                        "parameters"
                    ].items()
                    if key.split(".")[-1] != "encoder"
                    and not input_or_output_feature(key)
                }
            }

            ds_encoder_hyperopt_params = {
                "parameters": {
                    **output_encoder_hyperopt_params["parameters"],
                    **input_encoder_hyperopt_params["parameters"],
                }
            }
            curr_config["input_features"][0]["encoder"] = encoder_name

            # populate hyperopt parameters w/encoder specific settings
            curr_config["hyperopt"].update(
                {
                    "parameters": {
                        **ds_encoder_hyperopt_params["parameters"],
                        **hyperopt_config["parameters"],
                    }
                }
            )

            config_fp = os.path.join(
                EXPERIMENT_CONFIGS_DIR, f"config_{dataset}_{encoder_name}.yaml"
            )
            with open(config_fp, "w") as f:
                yaml.dump(curr_config, f)

            config_fps[dataset].append(config_fp)

    return config_fps
