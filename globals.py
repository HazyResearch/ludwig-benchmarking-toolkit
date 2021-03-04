ENCODER_CONFIG_DIR = './encoder-configs'
EXPERIMENT_CONFIGS_DIR = './hyperopt-experiment-configs'

ENCODER_HYPEROPT_FILENAMES = {
    'bert': 'bert_hyperopt.yaml',
    'rnn' : 'rnn_hyperopt.yaml',
    'distilbert' : 'distilbert_hyperopt.yaml',
    'electra' : 'electra_hyperopt.yaml',
    'roberta' : 'electra_hyperopt.yaml',
    'stackedparallelcnn' : 'stackedparallelcnn_hyperopt.yaml',
    #'t5' : 't5_hyperopt.yaml'
}

ENCODER_FILE_LIST = ENCODER_HYPEROPT_FILENAMES.values()

CONFIG_TEMPLATE_FILE = './experiment-templates/config_template.yaml'
DATASET_METADATA_FILE = './experiment-templates/dataset_metadata.yaml'
HYPEROPT_CONFIG_FILE = './experiment-templates/hyperopt_config.yaml'
EXPERIMENT_OUTPUT_DIR = './experiment-outputs-2'

PATH_TO_PRETRAINED_EMBEDDINGS = None

SMOKE_DATASETS = {
    'goemotions' : './smoke-datasets/goemotions.csv',
    'fever' : './smoke-datasets/fever.csv'
}
