import yaml
from utils import *
from copy import deepcopy
from build_def_files import *
from ludwig.api import LudwigModel
from ludwig.hyperopt.run import hyperopt


def download_data():
    data_file_paths = {}
    for dataset in dataset_metadata:
        data_class = dataset_metadata[dataset]['data_class']
        data_path = download_datasets(data_class)
        data_file_paths[dataset] = data_path
    return data_file_paths

def main():
    data_file_paths = download_data()
    print("Datasets downloaded...")
    config_files = build_config_files()
    print("Config files built...")

    for dataset_name, file_path in data_file_paths.items():
        for model_config_path in config_files[dataset_name]:
            with open(model_config_path) as f:
                model_config = yaml.load(f, Loader=yaml.SafeLoader)
            train_stats = hyperopt(model_config, dataset=file_path)

if __name__ == '__main__':
    main()

