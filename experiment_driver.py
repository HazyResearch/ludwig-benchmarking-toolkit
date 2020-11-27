import argparse
import datetime
import os
import pickle
import sys
from copy import deepcopy

import yaml

from build_def_files import *
from database import *
from ludwig.api import LudwigModel
from ludwig.hyperopt.run import hyperopt
from utils import *

MODEL_CONFIGS_DIR = './model-configs'
LOG_FILE = open('./experiment-logs/mini_exp.log', 'w')
sys.stdout = LOG_FILE

ELASTIC_DB = {
    'host' : "localhost",
    'port' : "9200",
    'username' : "an",
    'password' : "lb"
}

RUN_LOCALLY = True

def download_data():
    data_file_paths = {}
    for dataset in dataset_metadata:
        data_class = dataset_metadata[dataset]['data_class']
        data_path = download_dataset(data_class)
        data_file_paths[dataset] = data_path
    return data_file_paths

def save_to_elastic(es_db, document):
    es_db.index(
        index='text-classification', 
        id=i, 
        body=document
    )
   

def run_local_experiments(data_file_paths, config_files, es_db=None):
    print("Running hyperopt experiments...")
    for dataset_name, file_path in data_file_paths.items():
        print("Dataset: {}".format(dataset_name))
        for model_config_path in config_files[dataset_name]:
            config_name = model_config_path.split('/')[-1].split('.')[0]
            experiment_name = config_name.split('_')[-2] + "_" + \
                config_name.split('_')[-1]
            print("Experiment: {}".format(experiment_name))
            output_dir = os.path.join(
                '/juice/scr/avanika/ludwig-benchmark-experiments', 
                experiment_name
            )
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
                with open(model_config_path) as f:
                    model_config = yaml.load(f, Loader=yaml.SafeLoader)
                start = datetime.datetime.now()
                train_stats = hyperopt(
                                    model_config,
                                    dataset=file_path,
                                    model_name=config_name, 
                                    gpus="0,1",
                                    output_directory=output_dir
                            )
                print("time to complete: {}".format(
                        datetime.datetime.now() - start)
                    ) 
           
                # Save output locally
                try:
                    pickle.dump(train_stats,
                        open(os.path.join(
                                output_dir,
                                f'{config_name}_train_stats.pkl'
                            ),
                            'wb'
                        )
                    )

                except FileNotFoundError:
                    continue

                # save output to db
                if es_db:
                    document = train_stats.update({'config': model_config})
                    save_to_elastic(es_db, document)

def main():
    # argparse
    parser = argparse.ArgumentParser(
        description='ludwig benchmark experiments driver',
    )

    parser.add_argument(
        '-hcd',
        '--hyperopt_config_dir',
        help='directory to save all model config',
        type=str,
        default='./hyperopt-config-dir'
    )
    
    parser.add_argument(
        '-eod',
        '--experiment_output_dir',
        help='directory to save hyperopt runs',
        type=str,
        default='./experiment-output-dir'
    )

    parser.add_argument(
        '-re',
        '--run_environment',
        help='environment where experiment will be run',
        choices=['local', 'gcp'],
        default='local'
    )
    parser.add_argument(
        '-esc',
        '--elasticsearch_config',
        help='path to elastic db config file',
        type=str,
    )
    
    parser.add_argument(
        '-dcd',
        '--dataset_cache_dir',
        help="path to cache downloaded datasets",
        type=str,
        default=None
    )
    
    args = parser.parse_args()    

    print("GPUs {}".format(os.system('nvidia-smi -L')))
    data_file_paths = download_data(args.dataset_cache_dir)
    print("Building config files...")
    config_files = build_config_files()
    print("Set up elastic db...")
    if ELASTIC_DB is not None:
        es_db = Database(
            ELASTIC_DB['host'],
            ELASTIC_DB['port'],
            (ELASTIC_DB['username'], ELASTIC_DB['password'])
        )

    if args.run_environment == 'local':
        run_local_experiments(data_file_paths, config_files, es_db=es_db)

if __name__ == '__main__':
    main()
    LOG_FILE.close()

