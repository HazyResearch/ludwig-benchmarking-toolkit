import logging
import os
from copy import deepcopy

import yaml

import globals
from globals import *
from utils.experiment_utils import load_yaml

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

    for encoder_filename in ENCODER_FILE_LIST:
        with open(os.path.join(ENCODER_CONFIG_DIR, encoder_filename)) as f:
            encoder_hyperopt_params = yaml.load(f, Loader=yaml.SafeLoader)
            encoder_hyperopt_vals.append(encoder_hyperopt_params)

    config['hyperopt'].update(hyperopt_config)

    for dataset, metadata in dataset_metadata.items():
        # each dataset will have a model specific config file
        config_fps[dataset] = []

        config['input_features'][0]['name'] = metadata['input_name']
        config['output_features'][0]['name'] = metadata['output_name']
        config['output_features'][0]['type'] = metadata['output_type']
        config['hyperopt']['output_feature'] = metadata['output_name']
        input_feature_name = metadata['input_name']
        output_feature_name = metadata['output_name']

        for encoder_hyperopt_params in encoder_hyperopt_vals:
            curr_config = deepcopy(config)
            encoder_name = encoder_hyperopt_params['parameters'][
                'input_features.name.encoder']

            # update input and output parameters (not preprocessing)
            curr_config['input_features'][0].update(
                encoder_hyperopt_params['input_features'][0]
            )
            insert_global_vars(curr_config['input_features'][0])

            if 'output_features' in encoder_hyperopt_params.keys():
                curr_config['output_features'][0].update(
                    encoder_hyperopt_params['output_features'][0]
                )
                insert_global_vars(curr_config['output_features'][0])

            # handle encoder specific preprocessing
            preprocessing = curr_config['input_features'][0]['preprocessing']
            for key, _ in preprocessing.items():
                preprocessing[key] = \
                encoder_hyperopt_params['input_features'][0]['preprocessing'][
                    key]

            # handle encoder specific training params
            if 'training' in encoder_hyperopt_params.keys():
                curr_config['training'].update(
                    encoder_hyperopt_params['training']
                )

            def input_or_output_feature(param_key):
                if param_key.split(".")[0] == 'input_features':
                    return input_feature_name
                return output_feature_name

            # handle encoder specific hyperopt
            ds_encoder_hyperopt_params = {
                'parameters': {
                    input_or_output_feature(key) + "." + key.split('.')[-1]: value 
                    for key, value in 
                    encoder_hyperopt_params['parameters'].items()
                    if key.split('.')[-1] != 'encoder'
                }
            }
            curr_config['input_features'][0]['encoder'] = encoder_name

            # populate hyperopt parameters w/encoder specific settings
            curr_config['hyperopt'].update(
                {
                    'parameters':
                        {**ds_encoder_hyperopt_params['parameters'],
                         **hyperopt_config['parameters']}
                }
            )

            config_fp = os.path.join(
                EXPERIMENT_CONFIGS_DIR,
                f"config_{dataset}_{encoder_name}.yaml"
            )
            with open(config_fp, "w") as f:
                yaml.dump(curr_config, f)

            config_fps[dataset].append(config_fp)

    return config_fps
