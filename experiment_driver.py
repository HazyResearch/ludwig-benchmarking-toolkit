import yaml
from utils import *
from copy import deepcopy
from build_def_files import *
from ludwig.api import LudwigModel
from ludwig.hyperopt.run import hyperopt
import pickle
import os

MODEL_CONFIGS_DIR = './model-configs'

def download_data():
    data_file_paths = {}
    for dataset in dataset_metadata:
        data_class = dataset_metadata[dataset]['data_class']
        data_path = download_datasets(data_class)
        data_file_paths[dataset] = data_path
    return data_file_paths

def main():
    print("GPUs {}".format(os.system('nvidia-smi -L')))
    data_file_paths = download_data()
    print("Datasets downloaded...")
    config_files = build_config_files()
    print("Config files built...")

    for dataset_name, file_path in data_file_paths.items():
        print("Dataset: {}".format(dataset_name))
        for model_config_path in config_files[dataset_name]:
            print("Model config: {}".format(model_config_path))
            with open(model_config_path) as f:
                model_config = yaml.load(f, Loader=yaml.SafeLoader)
            train_stats = hyperopt(model_config, dataset=file_path, gpus="0,1")
            config_name = model_config_path.split('/')[-1].split('.')[0]
            pickle.dump(train_stats, open(f'{config_name}_train_stats.pkl','wb'))
if __name__ == '__main__':
    main()

