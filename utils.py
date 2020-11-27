import os
import json

def download_dataset(dataset_class: str) -> str:
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

def hash_dict(d: dict, max_length: Union[int, None] = 6) -> bytes:
    s = json.dumps(d, sort_keys=True, ensure_ascii=True)
    h = hashlib.md5(s.encode())
    d = h.digest()
    b = base64.b64encode(d)
    return b[:max_length]