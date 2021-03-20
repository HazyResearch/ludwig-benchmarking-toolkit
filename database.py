import json
import os
from utils.experiment_utils import hash_dict, format_fields_float
from elasticsearch import Elasticsearch
from utils.metadata_utils import append_experiment_metadata

def save_results_to_es(
    experiment_attr: dict,
    hyperopt_results: list,
    tune_executor: str,
    top_n_trials: int=None,
):
    elastic_config = experiment_attr['elastic_config']

    es_db = Database(
                elastic_config['host'], 
                (elastic_config['username'], elastic_config['password']),
                elastic_config['username'],
                elastic_config['index']
            )
    # save top_n model configs to elastic
    if top_n_trials is not None and len(hyperopt_results) > top_n_trials:
        hyperopt_results = hyperopt_results[0:top_n_trials]

    hyperopt_run_data = get_model_ckpt_paths(hyperopt_results, 
            experiment_attr['output_dir'], executor=tune_executor)

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
                data_path=experiment_attr['dataset_path'],
                run_stats=run
            )
        except:
            pass

        formatted_document = es_db.format_document(
            document,
            encoder=experiment_attr['encoder'],
            dataset=experiment_attr['dataset'],
            config=experiment_attr['model_config']
        )

        formatted_document['sampled_run_config'] = new_config
        
        try:
            es_db.upload_document(
                hash_dict(new_config),
                formatted_document
            )
        except:
            print(f"error uploading"
                  f"{experiment_attr['dataset']} x {experiment_attr['encoder']}"
                  f"to elastic...")


class Database:
    def __init__(
        self, 
        host,
        http_auth,
        user_id,
        index
    ):
        self.host = host
        self.http_auth = http_auth
        self.user_id = user_id
        self.index=index
        self._initialize_db()
        self._create_index(self.index)

    def _initialize_db(self):
        self.es_connection = Elasticsearch(
            [self.host], 
            http_auth=self.http_auth
        )

    def _create_index(self, index_name: str):
        mapping = {
            "mappings": {
                "_doc" : {
                    "properties": {
                        "sampled_run_config" : {
                            "type" : "nested"
                            }
                        }
                    }
                }
            }
        self.es_connection.indices.create(
                index=index_name,
                body=mapping,
                include_type_name=True,
                ignore=400 
            )

    def upload_document(
        self,
        id,
        document
    ):
        self.es_connection.index(
            index=self.index, 
            id=id, 
            body=document
        )
    
    def upload_document_from_outputdir(
        self,
        dir_path,
        encoder,
        dataset,

    ):
        hyperopt_stats = json.load(
            open(os.path.join(dir_path, 'hyperopt_statistics.json'), 'rb'),
            parse_int=float
        )

        formatted_document = self.format_document(
            hyperopt_stats, encoder, dataset
        )

        self.es_connection.index(
            index=self.index, 
            id=hash_dict(hyperopt_stats['hyperopt_config']), 
            body=formatted_document
        )

    def format_document(
        self,
        document,
        encoder,
        dataset,
        config=None

    ):
        formatted_document = {
            'user_id' : self.user_id,
            'encoder' : encoder,
            'dataset' : dataset
        }
        formatted_document.update(document)
        if config is not None:
            formatted_document.update({
                'hyperopt_exp_config' : config
            })

        return formatted_document


   





    
