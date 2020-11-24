import yaml
from utils import *
from copy import deepcopy
from build_def_files import *
from ludwig.api import LudwigModel
from ludwig.hyperopt.run import hyperopt
import pickle

MODEL_CONFIGS_DIR = './model-configs'


def download_data():
    data_file_paths = {}
    for dataset in dataset_metadata:
        data_class = dataset_metadata[dataset]['data_class']
        data_path = download_dataset(data_class)
        data_file_paths[dataset] = data_path
    return data_file_paths


def main():
    print("Downloading datasets...")
    data_file_paths = download_data()
    print("Building config files...")
    config_files = build_config_files()

    print("Running experiments...")
    for dataset_name, file_path in data_file_paths.items():
        for model_config_path in config_files[dataset_name]:
            with open(model_config_path) as f:
                model_config = yaml.load(f, Loader=yaml.SafeLoader)
            train_stats = hyperopt(model_config, dataset=file_path, gpus=['gpu0', 'gpu1'])
            pickle.dump(train_stats, open('train_stats.pkl','wb'))
if __name__ == '__main__':
    main()
