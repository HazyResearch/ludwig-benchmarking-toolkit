import yaml

from build_def_files import dataset_metadata, build_config_files
from utils import download_dataset

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
        for config_path in config_files[dataset_name]:
            with open(config_path) as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
            # train_stats = hyperopt(config, dataset=file_path)


if __name__ == '__main__':
    main()
