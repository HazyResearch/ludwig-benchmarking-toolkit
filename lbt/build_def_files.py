import os

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

    model_hyperopt_vals = []
    # select relevant encoders
    for model_filename in globals.MODEL_FILE_LIST:
        with open(os.path.join(MODEL_CONFIG_DIR, model_filename)) as f:
            model_hyperopt_params = yaml.load(f, Loader=yaml.SafeLoader)
            model_hyperopt_vals.append(model_hyperopt_params)

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

    # Iterate through each dataset, building a model specific config file
    # for each dataset

    for dataset, metadata in selected_datasets.items():
        # each dataset will have a model specific config file
        config_fps[dataset] = []

        # Collect all input features and output features for each dataset
        for idx, input_feature_name in enumerate(metadata["input_features"]):
            ipt_feat = {}
            ipt_feat["name"] = input_feature_name["name"]
            ipt_feat["type"] = input_feature_name["type"]
            if idx == 0:
                config["input_features"] = [ipt_feat]
            else:
                config["input_features"].append(ipt_feat)
        for idx, output_feature_info in enumerate(metadata["output_features"]):
            out_feat = {}
            out_feat["name"] = output_feature_info["name"]
            out_feat["type"] = output_feature_info["type"]
            if idx == 0:
                config["output_features"] = [out_feat]
            else:
                config["output_features"].append(out_feat)

        # set the output_feature to optimize against during hyperparam search
        if len(metadata["output_features"]) > 1:
            config["hyperopt"]["output_feature"] = "combined"
        else:
            config["hyperopt"]["output_feature"] = metadata["output_features"][
                0
            ]["name"]

        # Add model specific
        for model_hyperopt_params in model_hyperopt_vals:
            curr_config = deepcopy(config)
            model_name = model_hyperopt_params['model_name']

            # for each input type, get relevant encoder information
            type_encoder_mapping = {}
            for feature_type_params in model_hyperopt_params["input_features"]:
                type_encoder_mapping[feature_type_params["type"]
                                     ] = feature_type_params
            # for each output type, get relevant decoder information
            type_decoder_mapping = {}
            for feature_type_params in model_hyperopt_params["output_features"]:
                type_decoder_mapping[feature_type_params["type"]
                                     ] = feature_type_params

            type_input_feature_mapping = {}  # {type: list[input_features]}
            type_output_feature_mapping = {}  # {type: list[output_features]}

            # For each input feature, replace with relevant encoder information
            for idx in range(len(curr_config["input_features"])):
                input_feature = curr_config["input_features"][idx]
                feat_type = input_feature["type"]
                curr_config["input_features"][idx].update(
                    type_encoder_mapping[feat_type]
                )
                if feat_type in type_input_feature_mapping.keys():
                    type_input_feature_mapping[feat_type].append(
                        input_feature['name'])
                else:
                    type_input_feature_mapping[feat_type] = [
                        input_feature['name']]

            # For each output feature, replace with relevant decoder information
            for idx in range(len(curr_config["output_features"])):
                output_feature = curr_config["output_features"][idx]
                feat_type = output_feature["type"]
                curr_config["output_features"][idx].update(
                    type_encoder_mapping[output_feature["type"]]
                )
                if feat_type in type_output_feature_mapping.keys():
                    type_output_feature_mapping[feat_type].append(
                        output_feature['name'])
                else:
                    type_output_feature_mapping[feat_type] = [
                        output_feature['name']]

            def input_or_output_feature(param_key):
                if param_key.split(".")[0] == "input_features":
                    return True
                return False

            # handle hyperparameters
            parameters = {}
            for key, value in model_hyperopt_params["parameters"].items():
                # handle encoder / decoder param
                if "feature" in key:
                    # check input / output feature
                    type = key.split(".")[1]
                    if input_or_output_feature(key):
                        if type in type_input_feature_mapping.keys():
                            for input_feature in type_input_feature_mapping[type]:
                                parameters.update(
                                    {input_feature + "." + key.split(".")[-1]: value})
                    else:
                        for output_feature in type_output_feature_mapping[type]:
                            parameters.update(
                                {output_feature + "." + key.split(".")[-1]: value})

                else:  # handle combinar param,  training_param
                    parameters.update({key: value})

            # add `parameters` to curr_config
            if "parameters" in hyperopt_config.keys():
                curr_config["hyperopt"].update(
                    {
                        "parameters": {
                            **parameters,
                            **hyperopt_config["parameters"],
                        }
                     }
                )
            else:
                curr_config["hyperopt"].update(
                   {
                        "parameters": {
                            **parameters
                        }
                    }
                )

            # Add combiner specific training parameters
            if "combiner" in model_hyperopt_params.keys():
                curr_config["combiner"].update(
                    model_hyperopt_params["combiner"]
                )

            # Add model specific training parameters
            if "training" in model_hyperopt_params.keys():
                curr_config["training"].update(
                    model_hyperopt_params["training"]
                )

            config_fp = os.path.join(
                EXPERIMENT_CONFIGS_DIR, f"config_{dataset}_{model_name}.yaml"
            )
            with open(config_fp, "w") as f:
                yaml.dump(curr_config, f)

            config_fps[dataset].append(config_fp)

    return config_fps
