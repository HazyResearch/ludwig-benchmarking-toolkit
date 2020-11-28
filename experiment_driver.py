import argparse
import datetime
import logging
import os
import pickle
import sys
from copy import deepcopy

import yaml
from ludwig.api import LudwigModel
from ludwig.hyperopt.run import hyperopt

from build_def_files import *
from database import *
from globals import *
from utils import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def download_data(cache_dir=None):
    data_file_paths = {}
    for dataset in dataset_metadata:
        data_class = dataset_metadata[dataset]['data_class']
        data_path = download_dataset(data_class, cache_dir)
        data_file_paths[dataset] = data_path
    return data_file_paths

def run_local_experiments(data_file_paths, config_files, es_db=None):
    logging.info("Running hyperopt experiments...")
    for dataset_name, file_path in data_file_paths.items():
        logging.info("Dataset: {}".format(dataset_name))
        for model_config_path in config_files[dataset_name]:
            config_name = model_config_path.split('/')[-1].split('.')[0]
            experiment_name = config_name.split('_')[-2] + "_" + \
                config_name.split('_')[-1]
            logging.info("Experiment: {}".format(experiment_name))
            output_dir = os.path.join(EXPERIMENT_OUTPUT_DIR, experiment_name)

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
                model_config = load_yaml(model_config_path)
                start = datetime.datetime.now()
                train_stats = hyperopt(
                    model_config,
                    dataset=file_path,
                    model_name=config_name, 
                    gpus="0,1",
                    output_directory=output_dir
                )

                logging.info("time to complete: {}".format(
                    datetime.datetime.now() - start)
                ) 
           
                # Save output locally
                try:
                    pickle.dump(
                        train_stats, 
                        open(os.path.join(
                            output_dir, f"{config_name}_train_stats.pkl"),'wb'
                        )
                    )

                except FileNotFoundError:
                    continue

                # save output to db
                if es_db:
                    document = {'hyperopt_results': train_stats}
                    es_db.save_document(
                        hash_dict(model_config),
                        document
                    )

def main():
    parser = argparse.ArgumentParser(
        description='Ludwig experiments benchmarking driver script',
    )

    parser.add_argument(
        '-hcd',
        '--hyperopt_config_dir',
        help='directory to save all model config',
        type=str,
        default=EXPERIMENT_CONFIGS_DIR
    )
    
    parser.add_argument(
        '-eod',
        '--experiment_output_dir',
        help='directory to save hyperopt runs',
        type=str,
        default=EXPERIMENT_OUTPUT_DIR
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
        default=None
    )
    
    parser.add_argument(
        '-dcd',
        '--dataset_cache_dir',
        help="path to cache downloaded datasets",
        type=str,
        default=None
    )

    # list of encoders to run hyperopt search over : 
    # default is 23 ludwig encoders
    parser.add_argument(
        '-cel',
        '--custom_encoders_list',
        help="provide list of encoders to run hyperopt experiments on. \
            The default setting is to use all 23 Ludwig encoders",
        nargs='+',
        choices=['all', 'bert', 'rnn'],
        default="all"
    )
    
    args = parser.parse_args()   

    logging.info("Set global variables...")
    set_globals(args) 

    logging.info("GPUs {}".format(os.system('nvidia-smi -L')))

    data_file_paths = download_data(args.dataset_cache_dir)
    logging.info("Building hyperopt config files...")
    config_files = build_config_files()

    es_db = None
    if args.elasticsearch_config is not None:
        logging.info("Set up elastic db...")
        elastic_config = load_yaml(args.elasticsearch_config)
        es_db = Database(elastic_config['host'], elastic_config['http_auth'])
    
    if args.run_environment == 'local':
        run_local_experiments(
            data_file_paths, 
            config_files, 
            es_db=es_db
        )

if __name__ == '__main__':
    main()
