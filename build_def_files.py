
import yaml
import os
from copy import deepcopy

ENCODER_HYPEROPT_FILENAMES = ['bert_hyperopt.yaml']
ENCODER_CONFIG_DIR = './encoder-configs'
EXPERIMENT_CONFIGS_DIR = './experiment-configs'

if not os.path.isdir(EXPERIMENT_CONFIGS_DIR):
    os.mkdir(EXPERIMENT_CONFIGS_DIR)

with open('model_template.yaml') as f:
    template = yaml.load(f, Loader=yaml.SafeLoader)

with open('dataset_metadata.yaml') as f:
    dataset_metadata = yaml.load(f, Loader=yaml.SafeLoader)

with open('hyperparam_values.yaml') as f:
    hyperopt_values = yaml.load(f, Loader=yaml.SafeLoader)
    
encoder_hyperopt_vals = []

for encoder_filename in ENCODER_HYPEROPT_FILENAMES:
    with open(os.path.join(ENCODER_CONFIG_DIR,encoder_filename)) as f:
        encoder_hyperopt_params = yaml.load(f, Loader=yaml.SafeLoader)
        encoder_hyperopt_vals.append(encoder_hyperopt_params)


def build_config_files():
    config_files = {}
    template_copy = deepcopy(template)

    template_copy['hyperopt'].update(hyperopt_values)

    for dataset in dataset_metadata:
        # each dataset, will have a model specific config file
        config_files[dataset] = []

        template_copy['input_features'][0]['name'] = dataset_metadata[dataset]['input_name']
        template_copy['output_features'][0]['name'] = dataset_metadata[dataset]['output_name']
        template_copy['output_features'][0]['type'] = dataset_metadata[dataset]['output_type']
        template_copy['hyperopt']['output_feature'] = dataset_metadata[dataset]['output_name']
        input_feature_name = dataset_metadata[dataset]['input_name']

        for encoder_hyperopt_params in encoder_hyperopt_vals:

            for key, _ in template_copy['input_features'][0]['preprocessing'].items():
                template_copy['input_features'][0]['preprocessing'][key] = encoder_hyperopt_params['input_features'][0]['preprocessing'][key]
                
            encoder_name = encoder_hyperopt_params['parameters']['input_features.name.encoder']

            ds_encoder_hyperopt_params = {
                'parameters': {
                    input_feature_name + "." + key.split('.')[-1] : value
                    for key, value in encoder_hyperopt_params['parameters'].items()
                    if key.split('.')[-1] != 'encoder'
                }
            }

            template_copy['input_features'][0]['encoder'] = encoder_name

            # populate hyperopt parameters w/encoder specific settings
            template_copy['hyperopt'].update(
                {
                    'parameters' :
                        {**ds_encoder_hyperopt_params['parameters'], **hyperopt_values['parameters']}
                }
            )      
            
            config_filepath = os.path.join(EXPERIMENT_CONFIGS_DIR, f"model_definition_{dataset}_{encoder_name}.yaml")
            with open(config_filepath, "w") as f:
                yaml.dump(template_copy, f)

            config_files[dataset].append(config_filepath)
            
    return config_files





