import argparse
import datetime
import json
import logging
import os
import pickle
import sys
import numpy as np
from copy import deepcopy

import yaml
from ludwig.hyperopt.run import hyperopt

import globals
from build_def_files import *
from database import *
from utils.experiment_utils import *
from utils.metadata_utils import append_experiment_metadata

logging.basicConfig(
    filename='elastic-exp.log', 
    format='%(levelname)s:%(message)s', 
    level=logging.DEBUG
)

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


def map_runstats_to_modelpath(hyperopt_training_stats, output_dir, executor='ray'):
    """ 
    maps output of individual hyperopt() run statistics to associated 
    output directories 
    """
    # helper function for finding output folder
    model_paths = []
    def find_model_path(d):
        if "model" in os.listdir(d):
            return d
        else:
            for x in os.scandir(d):
                if os.path.isdir(x):
                    path = find_model_path(x)
                    if path is not None:
                        #model_paths.append(path)
                        return path
            return None
    
    trial_dirs = []
    def find_params_experiment_dirs(d):
        if "params.json" in os.listdir(d):
            return d
        else:
            for x in os.scandir(d):
                if os.path.isdir(x):
                    path = find_params_experiment_dirs(x)
                    if path:
                        trial_dirs.append(path)
            return None

    def compare_dict(d1, d2):
        if type(d1) == list:
            if np.allclose(d1, d2, rtol=1e-02, atol=1e-02): return True
            return False
        else:
            if type(d1) == dict:
                for key, value in d1.items():
                    new_d2 = d2[key]
                    return compare_dict(value, new_d2)

    def compare_configs(cf_non_encoded, cf_json_encoded):
        for key, value in cf_non_encoded.items():
            value_other = cf_json_encoded[key]
            if type(value) == list:
                value_other = json.loads(value_other)
            if type(value) == str:
                value_other = json.loads(value_other)
            if value_other != value:
                return False
        else:
            return True
                

    def decode_hyperopt_run(run):
        run['training_stats'] = json.loads(run['training_stats'])
        run['parameters'] = json.loads(run['parameters'])
        run['eval_stats'] = json.loads(run['eval_stats'])
        return run
    
    if executor == 'ray': # folder construction is different
        hyperopt_run_metadata = []

        # populate paths
        for x in os.scandir(output_dir):
            if os.path.isdir(x):
                find_params_experiment_dirs(x)
        
        for hyperopt_run in hyperopt_training_stats:
            hyperopt_run_metadata.append(
                        {
                            'hyperopt_results' : decode_hyperopt_run(hyperopt_run),
                            'model_path' : None
                        }
                    )

        for hyperopt_run in hyperopt_run_metadata:
            hyperopt_params = hyperopt_run['hyperopt_results']['parameters']
            for path in trial_dirs:
                config_json = json.load(open(
                                    os.path.join(path, 'params.json')
                                    )
                                )
                if compare_configs(hyperopt_params, config_json):
                    model_path = find_model_path(path)
                    hyperopt_run['model_path'] = os.path.join(model_path,
                                                    'model')
    else:
        hyperopt_run_metadata = []
        for run_dir in os.scandir(output_dir):
            if os.path.isdir(run_dir):
                sample_training_stats = json.load(
                    open(
                        os.path.join(run_dir.path, \
                            "training_statistics.json"
                            ), "rb"
                    )
                )
                for hyperopt_run in hyperopt_training_stats:
                    if hyperopt_run['training_stats'] == sample_training_stats:
                        hyperopt_run_metadata.append(
                            {
                                'hyperopt_results' : hyperopt_run,
                                'model_path' : os.path.join(run_dir.path, \
                                        'model'
                                    )
                            }
                        )
    
    return hyperopt_run_metadata

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
            dataset = config_name.split('_')[1]
            encoder = "_".join(config_name.split('_')[2:])
            experiment_name = dataset + "_" + encoder
            
            
            logging.info("Experiment: {}".format(experiment_name))
            output_dir = os.path.join(globals.EXPERIMENT_OUTPUT_DIR, \
                experiment_name)

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
           
            output_dir = os.path.join(globals.EXPERIMENT_OUTPUT_DIR, \
                experiment_name)

            if not os.path.exists(os.path.join(output_dir, '.completed')):
                model_config = load_yaml(model_config_path)
                start = datetime.datetime.now()
                hyperopt_results = hyperopt(
                    model_config,
                    dataset=file_path,
                    model_name=config_name, 
                    #gpus=get_gpu_list(),
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
                    hyperopt_run_data = map_runstats_to_modelpath(
                        hyperopt_results, output_dir)
                    # ensures that all numerical values are of type float
                    format_fields_float(hyperopt_results)
                    for run in hyperopt_run_data:
                        new_config = substitute_dict_parameters(
                            copy.deepcopy(model_config),
                            parameters=run['hyperopt_results']['parameters']
                        )
                        del new_config['hyperopt']

                        document = {
                            'hyperopt_results': run['hyperopt_results'],
                            'model_path' : run['model_path']
                        }
                        
                        append_experiment_metadata(
                            document, 
                            model_path=run['model_path'], 
                            data_path=file_path
                        )

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

        # create .completed file to indicate that entire hyperopt experiment
        # is completed
        _ = open(os.path.join(globals.EXPERIMENT_OUTPUT_DIR, '.completed'), 'wb')
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
        default='/experiments/datasets'
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
    
    if args.run_environment == 'local' or args.run_environment == 'gcp':
        run_local_experiments(
            data_file_paths, 
            config_files, 
            es_db=es_db
        )

if __name__ == '__main__':
    main()
