import json
import os
from utils import hash_dict
from elasticsearch import Elasticsearch


class Database:
    def __init__(
        self, 
        host
        http_auth
        user_id
        index
    ):
        self.host = host
        self.http_auth = http_auth
        self.user_id = user_id
        self.index=index
        initialize_db()

    def initialize_db(self):
        self.es_connection = Elasticsearch(
            [self.host], 
            http_auth=self.http_auth
        )
    
    def upload_document(
        self,
        id
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
            parse_int='float'
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
                'hyperopt_config' : config
            })


        return formatted_document


   





    
