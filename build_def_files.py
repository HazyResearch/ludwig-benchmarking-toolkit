import os
import yaml
from copy import deepcopy

ENCODER_HYPEROPT_FILENAMES = ['bert_hyperopt.yaml','rnn_hyperopt.yaml']
ENCODER_CONFIG_DIR = './encoder-configs'
os.makedirs(ENCODER_CONFIG_DIR, exist_ok=True)
EXPERIMENT_CONFIGS_DIR = './experiment-configs'
os.makedirs(EXPERIMENT_CONFIGS_DIR, exist_ok=True)

if not os.path.isdir(EXPERIMENT_CONFIGS_DIR):
    os.mkdir(EXPERIMENT_CONFIGS_DIR)

with open('config_template.yaml') as f:
    template = yaml.load(f, Loader=yaml.SafeLoader)

with open('dataset_metadata.yaml') as f:
    dataset_metadata = yaml.load(f, Loader=yaml.SafeLoader)

with open('hyperopt_config.yaml') as f:
    hyperopt_config = yaml.load(f, Loader=yaml.SafeLoader)

encoder_hyperopt_vals = []

for encoder_filename in ENCODER_HYPEROPT_FILENAMES:
    with open(os.path.join(ENCODER_CONFIG_DIR, encoder_filename)) as f:
        encoder_hyperopt_params = yaml.load(f, Loader=yaml.SafeLoader)
        encoder_hyperopt_vals.append(encoder_hyperopt_params)


def build_config_files():
    config_fps = {}
    config = deepcopy(template)

    config['hyperopt'].update(hyperopt_config)

    for dataset, metadata in dataset_metadata.items():
        # each dataset will have a model specific config file
        config_fps[dataset] = []

        config['input_features'][0]['name'] = metadata['input_name']
        config['output_features'][0]['name'] = metadata['output_name']
        config['output_features'][0]['type'] = metadata['output_type']
        config['hyperopt']['output_feature'] = metadata['output_name']
        input_feature_name = metadata['input_name']

        for encoder_hyperopt_params in encoder_hyperopt_vals:

            curr_config = deepcopy(config)
            encoder_name = encoder_hyperopt_params['parameters'][
                'input_features.name.encoder']

            # deal with encoder specific preprocessing
            preprocessing = curr_config['input_features'][0]['preprocessing']
            for key, _ in preprocessing.items():
                preprocessing[key] = \
                encoder_hyperopt_params['input_features'][0]['preprocessing'][
                    key]

            # deal with encoder specific hyperopt
            ds_encoder_hyperopt_params = {
                'parameters': {
                    input_feature_name + "." + key.split('.')[-1]: value
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
