import base64
import copy
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
    elif dataset_class == 'AGNews':
        from ludwig.datasets.agnews import AGNews
        data = AGNews()
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

    print(globals.EXPERIMENT_OUTPUT_DIR)
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

def decode_str_dicts(d: str) -> dict:
    json_acceptable_string = d.replace("'", "\"")
    dct = json.loads(json_acceptable_string)
    return dct

def substitute_dict_parameters(original_dict: dict, parameters: dict) -> dict:
    """
    Fills in original ludwig config w/actual sampled hyperopt values
    """
    def subsitute_param(dct: dict, path: list, val):
        if len(path) == 1:
            dct[path[0]] = val
            return dct
        else:
            key = path.pop(0)
            subsitute_param(dct[key], path, val)
    
    # in some cases the dict is encoded as a str
    if type(parameters) == str:
        parameters = decode_str_dicts(parameters)

    for key, value in parameters.items():
        path = key.split(".")
        # Check for input/output parameter edge cases
        if path[0] not in original_dict.keys():
            # check if param is associate with output feature
            for idx, out_feature in enumerate(original_dict['output_features']):
                if out_feature['name'] == path[0]:
                    original_dict['output_features'][idx][path[1]] = value
                    break

            for idx, out_feature in enumerate(original_dict['input_features']):
                if out_feature['name'] == path[0]:
                    original_dict['input_features'][idx][path[1]] = value
                    break
        else:
            subsitute_param(original_dict, path, value)
    return original_dict



