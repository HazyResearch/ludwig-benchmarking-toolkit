from elasticsearch import Elasticsearch

class Database:
    def __init__(
        self, 
        host=None,
        user_id=None,
        http_auth=(None, None)
    ):
        self.host = host
        self.http_auth = http_auth

    def initialize_db(self):
        self.es_db = Elasticsearch(
            [self.host], 
            http_auth=self.http_auth
        )
    
    def save_document(
        self,
        index='text-classification',
        id=None,
        document={}
    ):
        self.es_db.index(
            index=index, 
            id=id, 
            body=document
        )
   





    
