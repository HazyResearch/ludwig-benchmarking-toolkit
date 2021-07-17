import os

PATH_HERE = os.path.abspath(os.path.dirname(__file__))
MODEL_CONFIG_DIR = os.path.join(PATH_HERE, "model-configs")
# EXPERIMENT_CONFIGS_DIR = '/experiments/ludwig-bench-textclassification/experiment-configs'
EXPERIMENT_CONFIGS_DIR = os.path.join(PATH_HERE, "hyperopt-experiment-configs")
DATASET_CACHE_DIR = os.path.join(PATH_HERE, "datasets")
ENERGY_LOGGING_DIR = os.path.join(PATH_HERE, "energy_logging")

MODEL_HYPEROPT_FILENAMES = {
    "bert": "bert_hyperopt.yaml",
    "rnn": "rnn_hyperopt.yaml",
    "distilbert": "distilbert_hyperopt.yaml",
    "electra": "electra_hyperopt.yaml",
    "roberta": "roberta_hyperopt.yaml",
    "stacked_parallel_cnn": "stackedparallelcnn_hyperopt.yaml",
    "t5": "t5_hyperopt.yaml",
    "resnet": "resnet_hyperopt.yaml",
    "stacked_cnn": "stackedcnn_hyperopt.yaml",
    "tabnet": "tabtransformer_hyperopt.yaml"
}

MODEL_FILE_LIST = MODEL_HYPEROPT_FILENAMES.values()
DATASETS_LIST = ["RossmannStoreSales"]

CONFIG_TEMPLATE_FILE = "./experiment-templates/task_template.yaml"
DATASET_METADATA_FILE = "./experiment-templates/dataset_metadata.yaml"
HYPEROPT_CONFIG_FILE = "./experiment-templates/hyperopt_config.yaml"
EXPERIMENT_OUTPUT_DIR = "./experiment-outputs"

PATH_TO_PRETRAINED_EMBEDDINGS = None

RUNTIME_ENV = "local"
