import os
def download_datasets(dataset_class):
    if dataset_class == 'GoEmotions':
        from ludwig.datasets.goemotions import GoEmotions
        data = GoEmotions()
    elif dataset_class == 'Fever':
        from ludwig.datasets.fever import Fever
        data = Fever()
    elif dataset_class == 'SST2':
        from ludwig.datasets.sst2 import SST2
        data = SST2()
    else:
        return None
    return os.path.join(data.processed_dataset_path,data.config['csv_filename'])


