from database import Database
from utils.experiment_utils import *
#from experiment_driver import map_runstats_to_modelpath
import pickle
import os
import json
from utils.metadata_utils import append_experiment_metadata
import ray

ray.init(address='auto')

elastic_config_file = './elasticsearch_config.yaml'
path_to_stats_file = '/experiments/ludwig-bench-textclassification/experiment-outputs/sst5_electra/sst5_electra_hyperopt_results.pkl'
path_to_output_dir = '/experiments/ludwig-bench-textclassification/experiment-outputs/sst5_electra/'
path_to_dataset = '/experiments/ludwig-bench-textclassification/data/sst5_1.0/processed/sst5.csv'
path_to_model_config = '/experiments/ludwig-bench-textclassification/experiment-configs/config_sst5_electra.yaml'

def map_runstats_to_modelpath(
    hyperopt_training_stats,
    output_dir,
    executor='ray'
):
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

    def get_last_checkpoint(path):
        checkpoints = [
            ckpt_dir
            for ckpt_dir in os.scandir(path)
            if os.path.isdir(ckpt_dir) and "checkpoint" in ckpt_dir.path
        ]

        sorted_cps = sorted(checkpoints, key=lambda d: d.path)
        return sorted_cps[-1]

    def compare_configs(cf_non_encoded, cf_json_encoded):
        for key, value in cf_non_encoded.items():
            value_other = cf_json_encoded[key]
            if type(value) == list:
                value_other = json.loads(value_other)
            if type(value) == str:
                value_other = json.loads(value_other)
            if type(value) == int:
                value_other = int(value_other)
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

@ray.remote
def push_to_es(
        elastic_config,
        hyperopt_results,
        model_config,
        path_to_output_dir=path_to_output_dir,
        file_path=path_to_dataset,
        encoder='electra',
        dataset='sst5',
):
    es_db = Database(
            elastic_config['host'],
            (elastic_config['username'], elastic_config['password']),
            elastic_config['username'],
            elastic_config['index']
        )


    hyperopt_run_data = map_runstats_to_modelpath(
        hyperopt_results, path_to_output_dir)

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
            data_path=file_path,
            run_stats=run
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
            formatted_document,
        )
        print('uploaded...')




def main():
    elastic_config = None
    elastic_config = load_yaml(elastic_config_file)
    hyperopt_results = pickle.load(open(path_to_stats_file, 'rb'))
    #hyperopt_results = hyperopt_results[-16:]
    model_config = load_yaml(path_to_model_config)
    #push_to_es(elastic_config, hyperopt_results, model_config)
    results = ray.get([push_to_es.remote(elastic_config, [hyp], model_config) for hyp in hyperopt_results])
if __name__ == '__main__':
    main()
