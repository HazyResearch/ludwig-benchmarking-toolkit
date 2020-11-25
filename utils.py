import os

def download_dataset(dataset_class):
    if dataset_class == 'GoEmotions':
        from ludwig.datasets.goemotions import GoEmotions
        data = GoEmotions()
        data.load()
    elif dataset_class == 'Fever':
        from ludwig.datasets.fever import Fever
        data = Fever()
        data.load()
    elif dataset_class == 'SST2':
        from ludwig.datasets.sst2 import SST2
        data = SST2()
        data.load()
    else:
        return None
    return os.path.join(data.processed_dataset_path,
                        data.config['csv_filename'])

def initialize_elastic_db(host, port, username, password):
    try:
        from elasticsearch import Elasticsearch
    except ImportError:
        print ("Elastic search needs to be downloaded")
    
    es = Elasticsearch([{   
            'host': host, 
            'port': port
        }], http_auth=(username, password), timeout=99999)
    
    return es
