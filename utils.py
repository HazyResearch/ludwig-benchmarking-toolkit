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
