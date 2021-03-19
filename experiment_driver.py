import argparse
import datetime
import json
import logging
import pdb
import os
import ray
import pickle
import socket
import sys
from collections import defaultdict
from copy import deepcopy

import numpy as np
import ray
import yaml
import GPUtil
from ludwig.hyperopt.run import hyperopt
from collections import defaultdict 
from multiprocessing import Pool
import globals
from build_def_files import *
from database import Database
from utils.experiment_utils import *
from utils.metadata_utils import append_experiment_metadata

logging.basicConfig(
    filename='elastic-exp.log', 
    format='%(levelname)s:%(message)s', 
    level=logging.DEBUG
)

hostname = socket.gethostbyname(socket.gethostname())

def download_data(cache_dir=None, datasets: list=None):
    """ Returns files paths for all datasets """
    data_file_paths = {}
    for dataset in datasets:
        if dataset in dataset_metadata.keys():
            data_class = dataset_metadata[dataset]['data_class']
            data_path = download_dataset(data_class, cache_dir)
            data_file_paths[dataset] = data_path
        else:
            raise ValueError(f"{dataset} is not a valid dataset."
                              "for list of valid dataets see: \
                                python experiment_driver.py -h"
                            )
    return data_file_paths

def get_gpu_list():
    try:
        return os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        return None

def map_runstats_to_modelpath(
    hyperopt_training_stats: list, 
    output_dir: str, 
    executor: str='ray'
) -> list :
    """ 
    maps output of individual hyperopt run statistics to associated 
    output directories. Necessary for accessing model checkpoints
    """

    # helper function for finding folder which contains experiment outputs
    def find_model_path(d):
        if "model" in os.listdir(d):
            return d
        else:
            for x in os.scandir(d):
                if os.path.isdir(x):
                    path = find_model_path(x)
                    if path is not None:
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

    def get_last_checkpoint(path):
        checkpoints = [
            ckpt_dir 
            for ckpt_dir in os.scandir(path) 
            if os.path.isdir(ckpt_dir) and "checkpoint" in ckpt_dir.path
        ]
        
        sorted_cps = sorted(checkpoints, key=lambda d: d.path)
        return sorted_cps[-1]
            
    if executor == 'ray': # folder construction is different
        hyperopt_run_metadata = []

        # populate paths
        for x in os.scandir(output_dir):
            if os.path.isdir(x):
                find_params_experiment_dirs(x)
        
        for hyperopt_run in hyperopt_training_stats:
            hyperopt_run_metadata.append(
                    {
                        'hyperopt_results':decode_json_enc_dict(
                                            hyperopt_run, 
                                            ['parameters', 'training_stats', 'eval_stats']),
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
                if compare_json_enc_configs(hyperopt_params, config_json):
                    model_path = get_last_checkpoint(path)
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

def get_exp_path(exp_base_dir: str, trainable_func_prefix: str="trainable_"):
    dirs = []
    for dr in os.scandir(exp_base_dir):
        if trainable_func_prefix in dr.name:
            dirs.append(dr)
    return dirs

def json_decode(d: dict):
    for key, value in d.items():
        if key in ['parameters', 'training_stats', 'eval_stats']:
            d[key] = json.loads(d[key])
    return d

def collect_completed_trial_results(ray_exp_dirs: str):
    results, metrics, params = [], [], []
    for ray_exp_dir in ray_exp_dirs:
        for trial_dir in os.scandir(ray_exp_dir):
            if "trial_" in trial_dir.name:
                for f in os.scandir(trial_dir):
                    if "progress" in f.name:
                        progress = pd.read_csv(f)
                        last_iter = len(progress) - 1
                        last_iter_eval_stats = json.loads(progress.iloc[last_iter]['eval_stats'])
                        if 'overall_stats' in last_iter_eval_stats[list(last_iter_eval_stats.keys())[0]].keys():
                            trial_results = decode_json_enc_dict(progress.iloc[last_iter].to_dict(),
                                                                 ['parameters', 'training_stats', 'eval_stats']
                                                                )
                            trial_results['done'] = True
                            metrics.append(progress.iloc[last_iter]['metric_score'])
                            curr_path = f.path
                            params_path = curr_path.replace("progress.csv", "params.json")
                            trial_params = json.load(open(params_path, "rb"))
                            params.append(trial_params)
                            for key, value in trial_params.items():
                                config_key = "config" + "." +  key
                                trial_results[config_key] = value
                            results.append(trial_results)
    return results, metrics, params


def resume_training(model_config: dict, output_dir):
    ray_exp_dir = get_exp_path(output_dir)
    results, metrics, params = collect_completed_trial_results(ray_exp_dir)
    #params = format_params(params)
    new_num_samples = model_config['hyperopt']['sampler']['num_samples'] - len(metrics)
    model_config['hyperopt']['sampler']['search_alg']['points_to_evaluate'] = params
    model_config['hyperopt']['sampler']['search_alg']['evaluated_rewards'] = metrics
    model_config['hyperopt']['sampler']['num_samples'] = new_num_samples

    return model_config, results

def conditional_decorator(decorator, condition, *args):
    def wrapper(function):
        if condition(*args):
            return decorator(function)
        else:
            return function
    return wrapper

@conditional_decorator(
    ray.remote(num_cpus=0, resources={f"node:{hostname}": 0.001}),
    lambda runtime_env: runtime_env != 'local',
    globals.RUNTIME_ENV)
def run_hyperopt_exp(
    experiment_attr: dict,
    is_resume_training: bool=False,
    runtime_env: str="local"
) -> int:
    """if runtime_env == 'local':
        # temp solution to ray problems
        os.environ["TUNE_PLACEMENT_GROUP_AUTO_DISABLED"] = "1"
    
    os.environ["TUNE_PLACEMENT_GROUP_CLEANUP_DISABLED"] = "1"
    """

    dataset = experiment_attr['dataset']
    encoder = experiment_attr['encoder']
    output_dir = experiment_attr['output_dir']
    top_n_trials = experiment_attr['top_n_trials']
    model_config = experiment_attr['model_config']
    elastic_config = experiment_attr['elastic_config']
    
    try: 
        start = datetime.datetime.now()

        combined_ds, train_set, val_set, test_set = None, None, None, None
        combined_ds, train_set, val_set, test_set = process_dataset(
            experiment_attr['dataset_path'])

        tune_executor = model_config['hyperopt']['executor']['type']

        gpu_list = None
        if tune_executor != "ray":
            gpu_list = get_gpu_list()
     
        new_model_config = copy.deepcopy(experiment_attr['model_config'])
        existing_results = None
        if is_resume_training:
            new_model_config, existing_results = resume_training(new_model_config, output_dir)

        hyperopt_results = hyperopt(
            new_model_config, 
            dataset=combined_ds,
            training_set=train_set,
            validation_set=val_set,
            test_set=test_set,
            model_name=experiment_attr['model_name'], 
            gpus=gpu_list,
            output_directory=experiment_attr['output_dir']
        )

        hyperopt_results = [] 
        if existing_results is not None:
            hyperopt_results.extend(existing_results)
            # add logic to sort

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
            pass

        # save output to db
        if elastic_config:
            es_db = Database(
                elastic_config['host'], 
                (elastic_config['username'], elastic_config['password']),
                elastic_config['username'],
                elastic_config['index']
            )

            # save top_n model configs to elastic
            if top_n_trials is not None and len(hyperopt_results) > top_n_trials:
                hyperopt_results = hyperopt_results[0:top_n_trials]

            hyperopt_run_data = map_runstats_to_modelpath(
                hyperopt_results, output_dir, executor=tune_executor)

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
                
                try:
                    append_experiment_metadata(
                        document, 
                        model_path=run['model_path'], 
                        data_path=file_path,
                        run_stats=run
                    )
                except:
                    pass

                formatted_document = es_db.format_document(
                    document,
                    encoder=encoder,
                    dataset=dataset,
                    config=experiment_attr['model_config']
                )

                formatted_document['sampled_run_config'] = new_config
                
                try:
                    es_db.upload_document(
                        hash_dict(new_config),
                        formatted_document
                    )
                except:
                    print("error uploading to elastic...")
        
        return 1
    except:
        print("Error running experiment...not completed")
        return 0

def run_experiments(
    data_file_paths: dict, 
    config_files: dict, 
    top_n_trials: int,
    elastic_config=None,
    run_environment: str="local",
    resume_existing_exp: bool=False
):
    logging.info("Running hyperopt experiments...")
    # check if overall experiment has already been run
    if os.path.exists(os.path.join(globals.EXPERIMENT_OUTPUT_DIR, \
        '.completed')):
        print("Experiment is already completed!")
        return 

    completed_runs, experiment_queue = [], []
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
                experiment_attr = defaultdict()
                experiment_attr = {
                    'model_config' : copy.deepcopy(model_config),
                    'dataset_path' : file_path,
                    'top_n_trials' : top_n_trials,
                    'model_name' : config_name,
                    'output_dir' :  output_dir,
                    'encoder' : encoder,
                    'dataset' :  dataset,
                    'elastic_config' : elastic_config,
                }
                if run_environment == 'local':
                    completed_runs.append(run_hyperopt_exp(experiment_attr, 
                                                           resume_existing_exp,
                                                           run_environment))
                else:
                    experiment_queue.append(experiment_attr)
    
    if run_environment != 'local':
        completed_runs = ray.get([run_hyperopt_exp.remote(exp, resume_existing_exp, run_environment) 
                                  for exp in experiment_queue])

    if len(completed_runs) == len(experiment_queue):                
        # create .completed file to indicate that entire hyperopt experiment
        # is completed
        _ = open(os.path.join(
            globals.EXPERIMENT_OUTPUT_DIR, '.completed'), 'wb')
    else:
        print("Not all experiments completed!")
    

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
        '--resume_existing_exp',
        help='resume a previously stopped experiment',
        type=bool,
        default=False
    )
    
    parser.add_argument(
        '-eod',
        '--experiment_output_dir',
        help='directory to save hyperopt runs',
        type=str,
        default=EXPERIMENT_OUTPUT_DIR
    )

    parser.add_argument(
        '--datasets',
        help='list of datasets to run experiemnts on',
        nargs='+',
        choices=['all', 'agnews', 'amazon_reviews', 'amazon_review_polarity', \
                 'dbpedia', 'ethos_binary', 'goemotions', 'irony', 'sst2', 'sst5',\
                 'yahoo_answers', 'yelp_review_polarity', 'yelp_reviews', 'smoke'
                ],
        default=None,
        required=True
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
        choices=['all', 'bert', 'rnn', 'stacked_parallel_cnn', 'roberta',
                 'distilbert', 'electra', 't5'],
        default="all"
    )

    parser.add_argument(
        '-topn',
        '--top_n_trials',
        help="top n trials to save model performance for.",
        type=int,
        default=None
    )

    parser.add_argument(
        '-smoke',
        '--smoke_tests',
        type=bool,
        default=False
    )

    args = parser.parse_args()   
    logging.info("Set global variables...")
    set_globals(args) 

    logging.info("GPUs {}".format(os.system('nvidia-smi -L')))
    if args.smoke_tests:
        data_file_paths = SMOKE_DATASETS
    else:
        data_file_paths = download_data(args.dataset_cache_dir,
                                        args.datasets)
    
    logging.info("Building hyperopt config files...")
    config_files = build_config_files()

    elastic_config = None
    if args.elasticsearch_config is not None:
        logging.info("Set up elastic db...")
        elastic_config = load_yaml(args.elasticsearch_config)

    if args.run_environment == 'gcp':
        ray.init(address="auto")
    
    if args.run_environment == 'local' or args.run_environment == 'gcp':
        run_experiments(
            data_file_paths, 
            config_files, 
            top_n_trials=args.top_n_trials,
            elastic_config=elastic_config,
            run_environment=args.run_environment,
            resume_existing_exp=args.resume_existing_exp
        )

if __name__ == '__main__':
    main()
