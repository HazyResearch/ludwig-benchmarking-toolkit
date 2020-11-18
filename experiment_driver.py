import sys
import yaml
from utils import *
from copy import deepcopy
from build_def_files import *
from ludwig.api import LudwigModel
from ludwig.hyperopt.run import hyperopt
import pickle
import os
import datetime

MODEL_CONFIGS_DIR = './model-configs'
LOG_FILE = open('./experiment-logs/mini_exp.log', 'w')
sys.stdout = LOG_FILE

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
            config_name = model_config_path.split('/')[-1].split('.')[0]
            with open(model_config_path) as f:
                model_config = yaml.load(f, Loader=yaml.SafeLoader)
            start = datetime.datetime.now()
            train_stats = hyperopt(
				model_config,
				dataset=file_path,
				model_name=config_name, 
				gpus="0",
				output_directory='/dfs/scratch1/avanika/ludwig-benchmark-results'
			)
            print("time to complete: {}".format(datetime.datetime.now() - start)) 
	
            pickle.dump(train_stats, open(f'{config_name}_train_stats.pkl','wb'))
if __name__ == '__main__':
    main()
    LOG_FILE.close()

