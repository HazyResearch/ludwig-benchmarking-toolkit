import argparse
import datetime
import logging
import os
import pickle
import sys
from copy import deepcopy

import yaml
from ludwig.hyperopt.run import hyperopt

import globals
from build_def_files import *
from database import *
from utils.experiment_utils import *

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

def download_data(cache_dir=None):
    data_file_paths = {}
    for dataset in dataset_metadata:
        data_class = dataset_metadata[dataset]['data_class']
        data_path = download_dataset(data_class, cache_dir)
        data_file_paths[dataset] = data_path
    return data_file_paths

def get_gpu_list():
    try:
        return os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        return None

def run_local_experiments(data_file_paths, config_files, es_db=None):
    logging.info("Running hyperopt experiments...")

    # check if overall experiment has already been run
    if os.path.exists(os.path.join(globals.EXPERIMENT_OUTPUT_DIR, \
        '.completed')):
        return 

    for dataset_name, file_path in data_file_paths.items():
        logging.info("Dataset: {}".format(dataset_name))
        for model_config_path in config_files[dataset_name]:
            config_name = model_config_path.split('/')[-1].split('.')[0]
            dataset = config_name.split('_')[-2]
            encoder = config_name.split('_')[-1]
            experiment_name = dataset + "_" + encoder
            logging.info("Experiment: {}".format(experiment_name))
            output_dir = os.path.join(globals.EXPERIMENT_OUTPUT_DIR, \
                experiment_name)

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
            
            if not os.path.exists(os.path.join(output_dir, '.completed')):
                model_config = load_yaml(model_config_path)
                start = datetime.datetime.now()
                hyperopt_results = hyperopt(
                    model_config,
                    dataset=file_path,
                    model_name=config_name, 
                    gpus=get_gpu_list(),
                    output_directory=output_dir
                )

                logging.info("time to complete: {}".format(
                    datetime.datetime.now() - start)
                ) 
           
                # Save output locally
                try:
                    pickle.dump(
                        hyperopt_results, 
                        open(os.path.join(
                            output_dir, 
                            f"{dataset}_{encoder}_hyperopt_results.pkl"
                            ),'wb'
                        )
                    )

                    # create .completed file to indicate that experiment
                    # is completed
                    _ = open(os.path.join(output_dir, '.completed'), 'wb')

                except FileNotFoundError:
                    continue

                # save output to db
                if es_db:
                    # ensures that all numerical values are of type float
                    format_fields_float(hyperopt_results)
                    for run in hyperopt_results:
                        new_config = substitute_dict_parameters(
                            copy.deepcopy(model_config),
                            parameters=run['parameters']
                        )
                        del new_config['hyperopt']

                        document = {'hyperopt_results': run}
                        formatted_document = es_db.format_document(
                            document,
                            encoder=encoder,
                            dataset=dataset,
                            config=model_config
                        )

                        formatted_document['sampled_run_config'] = new_config

                        es_db.upload_document(
                            hash_dict(new_config),
                            formatted_document
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
        help='environment in which experiment will be run',
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
        help="list of encoders to run hyperopt experiments on. \
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
        es_db = Database(
            elastic_config['host'], 
            (elastic_config['username'], elastic_config['password']),
            elastic_config['username'],
            elastic_config['index']
        )
    
    if args.run_environment == 'local':
        run_local_experiments(
            data_file_paths, 
            config_files, 
            es_db=es_db
        )

if __name__ == '__main__':
    main()
