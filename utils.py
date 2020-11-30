import base64
import hashlib
import json
import os
from typing import Union

import yaml

import globals

def download_dataset(dataset_class: str, cache_dir: str=None) -> str:
    if dataset_class == 'GoEmotions':
        from ludwig.datasets.goemotions import GoEmotions
        data = GoEmotions()
        data.load(cache_dir)
    elif dataset_class == 'Fever':
        from ludwig.datasets.fever import Fever
        data = Fever()
        data.load(cache_dir)
    elif dataset_class == 'SST2':
        from ludwig.datasets.sst2 import SST2
        data = SST2()
        data.load(cache_dir)
    else:
        return None
    return os.path.join(data.processed_dataset_path,\
        data.config['csv_filename'])

def hash_dict(d: dict, max_length: Union[int, None] = 6) -> bytes:
    s = json.dumps(d, sort_keys=True, ensure_ascii=True)
    h = hashlib.md5(s.encode())
    d = h.digest()
    b = base64.b64encode(d)
    return b[:max_length]

def load_yaml(filename: str) -> dict:
    with open(filename) as f:
        file_contents = yaml.load(f, Loader=yaml.SafeLoader)
    return file_contents

def set_globals(args):
    globals.EXPERIMENT_CONFIGS_DIR = args.hyperopt_config_dir
    globals.EXPERIMENT_OUTPUT_DIR = args.experiment_output_dir
    if args.custom_encoders_list is not 'all':
        encoders_list = []
        for enc_name in args.custom_encoders_list:
            if enc_name in globals.ENCODER_HYPEROPT_FILENAMES.keys():
                encoders_list.append(globals.ENCODER_HYPEROPT_FILENAMES[
                    enc_name
                ])
        globals.ENCODER_FILE_LIST = encoders_list

    # create experiment output directories (if they don't already exist)
    for exp_dir in [globals.EXPERIMENT_CONFIGS_DIR, \
        globals.EXPERIMENT_OUTPUT_DIR]:
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)

def format_fields_float(l: list) -> list:
    def replace_ints(d):
        for k, v in d.items():
            if isinstance(v, dict):
                replace_ints(v)
            else:
                if type(v) == int:
                    v = float(v)
                d.update({k: v})
        return d
    
    formatted_out = [
                replace_ints(d)
                for d in l
            ]
    return formatted_out


