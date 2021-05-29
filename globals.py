ENCODER_CONFIG_DIR = "./model-configs"
# EXPERIMENT_CONFIGS_DIR = '/experiments/ludwig-bench-textclassification/experiment-configs'
EXPERIMENT_CONFIGS_DIR = "./hyperopt-experiment-configs"
DATASET_CACHE_DIR = "./datasets"
ENERGY_LOGGING_DIR = "./energy_logging"

ENCODER_HYPEROPT_FILENAMES = {
    "bert": "bert_hyperopt.yaml",
    "rnn": "rnn_hyperopt.yaml",
    "distilbert": "distilbert_hyperopt.yaml",
    "electra": "electra_hyperopt.yaml",
    "roberta": "roberta_hyperopt.yaml",
    "stacked_parallel_cnn": "stackedparallelcnn_hyperopt.yaml",
    "t5": "t5_hyperopt.yaml",
}

ENCODER_FILE_LIST = ENCODER_HYPEROPT_FILENAMES.values()
DATASETS_LIST = None

CONFIG_TEMPLATE_FILE = "./experiment-templates/task_template.yaml"
DATASET_METADATA_FILE = "./experiment-templates/dataset_metadata.yaml"
HYPEROPT_CONFIG_FILE = "./experiment-templates/hyperopt_config.yaml"
EXPERIMENT_OUTPUT_DIR = "./experiment-outputs"

PATH_TO_PRETRAINED_EMBEDDINGS = None

RUNTIME_ENV = "local"
