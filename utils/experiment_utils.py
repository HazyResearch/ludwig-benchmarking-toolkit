import base64
import copy
import hashlib
import json
import os
import pdb
import logging
from typing import Union

import globals
import pandas as pd
import yaml


def get_gpu_list():
    try:
        return os.environ["CUDA_VISIBLE_DEVICES"]
    except KeyError:
        return None


def download_dataset(dataset_class: str, cache_dir: str) -> str:
    if dataset_class == "GoEmotions":
        from ludwig.datasets.goemotions import GoEmotions

        data = GoEmotions(cache_dir)
        data.load()
    elif dataset_class == "Fever":
        from ludwig.datasets.fever import Fever

        data = Fever(cache_dir)
        data.load()
    elif dataset_class == "SST2":
        from ludwig.datasets.sst2 import SST2

        data = SST2(cache_dir, include_subtrees=True, remove_duplicates=True)
        data.load()
    elif dataset_class == "AGNews":
        from ludwig.datasets.agnews import AGNews

        data = AGNews(cache_dir)
        data.load()
    elif dataset_class == "SST5":
        from ludwig.datasets.sst5 import SST5

        data = SST5(cache_dir, include_subtrees=True)
        data.load()
    elif dataset_class == "EthosBinary":
        from ludwig.datasets.ethos_binary import EthosBinary

        data = EthosBinary(cache_dir)
        data.load()
    elif dataset_class == "YelpPolarity":
        from ludwig.datasets.yelp_review_polarity import YelpPolarity

        data = YelpPolarity(cache_dir)
        data.load()
    elif dataset_class == "DBPedia":
        from ludwig.datasets.dbpedia import DBPedia

        data = DBPedia(cache_dir)
        data.load()
    elif dataset_class == "Irony":
        from ludwig.datasets.irony import Irony

        data = Irony(cache_dir)
        data.load()
    elif dataset_class == "YelpReviews":
        from ludwig.datasets.yelp_reviews import YelpReviews

        data = YelpReviews(cache_dir)
        data.load()
    elif dataset_class == "YahooAnswers":
        from ludwig.datasets.yahoo_answers import YahooAnswers

        data = YahooAnswers(cache_dir)
        data.load()
    elif dataset_class == "AmazonPolarity":
        from ludwig.datasets.amazon_review_polarity import AmazonPolarity

        data = AmazonPolarity(cache_dir)
        data.load()
    elif dataset_class == "AmazonReviews":
        from ludwig.datasets.amazon_reviews import AmazonReviews

        data = AmazonReviews(cache_dir)
        data.load()
    else:
        return None
    return os.path.join(
        data.processed_dataset_path, data.config["csv_filename"]
    )


def process_dataset(dataset_path: str):
    dataset = pd.read_csv(dataset_path)
    if "split" in dataset.columns:
        train_df = dataset[dataset["split"] == 0]
        val_df = dataset[dataset["split"] == 1]
        test_df = dataset[dataset["split"] == 2]

        # no validation set provided, sample 10% of train set
        if len(val_df) == 0:
            val_df = train_df.sample(frac=0.1, replace=False)
            train_df = train_df.drop(val_df.index)

        return None, train_df, val_df, test_df
    return dataset, None, None, None


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
    """ set global vars based on command line args """
    globals.EXPERIMENT_CONFIGS_DIR = args.hyperopt_config_dir
    logging.info(f"EXPERIMENT_CONFIG_DIR set to {args.hyperopt_config_dir}")
    globals.EXPERIMENT_OUTPUT_DIR = args.experiment_output_dir
    logging.info(f"EXPERIMENT_OUTPUT_DIR set to {args.experiment_output_dir}")
    globals.RUNTIME_ENV = args.run_environment
    logging.info(f"RUNTIME_ENV set to {args.run_environment}")
    globals.DATASET_CACHE_DIR = args.dataset_cache_dir
    logging.info(f"DATASET_CACHE_DIR set to {args.dataset_cache_dir}")

    if args.datasets is None:
        raise ValueError(
            "Please specify a dataset or list of dataset."
            "Use python experiment_driver.py --h to see: list of available datasets."
        )
    else:
        if "smoke" in args.datasets:
            globals.DATASET_LIST = list(globals.SMOKE_DATASETS.keys())
            logging.info("Setting global datasets list to smoke datasets...")
        else:
            globals.DATASET_LIST = args.datasets
            logging.info(f"Setting global datasets list to {args.datasets}")

    if "all" not in args.custom_encoders_list:
        encoders_list = []
        for enc_name in args.custom_encoders_list:
            if enc_name in globals.ENCODER_HYPEROPT_FILENAMES.keys():
                encoders_list.append(
                    globals.ENCODER_HYPEROPT_FILENAMES[enc_name]
                )
        globals.ENCODER_FILE_LIST = encoders_list

    # create experiment output directories (if they don't already exist)
    for exp_dir in [
        globals.EXPERIMENT_CONFIGS_DIR,
        globals.EXPERIMENT_OUTPUT_DIR,
        globals.DATASET_CACHE_DIR
    ]:
        if not os.path.isdir(exp_dir):
            os.mkdir(exp_dir)


def format_fields_float(field_list: list) -> list:
    """ formats fields in elastic db entries """

    def replace_ints(d):
        for k, v in d.items():
            if isinstance(v, dict):
                replace_ints(v)
            else:
                if type(v) == int:
                    v = float(v)
                d.update({k: v})
        return d

    formatted_out = [replace_ints(d) for d in field_list]
    return formatted_out


def decode_str_dicts(d: str) -> dict:
    json_acceptable_string = d.replace("'", '"')
    dct = json.loads(json_acceptable_string)
    return dct


def substitute_dict_parameters(original_dict: dict, parameters: dict) -> dict:
    """ Fills in original ludwig config w/actual sampled hyperopt values """

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
            for idx, out_feature in enumerate(
                original_dict["output_features"]
            ):
                if out_feature["name"] == path[0]:
                    original_dict["output_features"][idx][path[1]] = value
                    break

            for idx, out_feature in enumerate(original_dict["input_features"]):
                if out_feature["name"] == path[0]:
                    original_dict["input_features"][idx][path[1]] = value
                    break
        else:
            subsitute_param(original_dict, path, value)
    return original_dict


def compare_json_enc_configs(cf_non_encoded, cf_json_encoded):
    """ compars to json encoded dicts """
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


def decode_json_enc_dict(encoded_dict, json_enc_params: list):
    for key, value in encoded_dict.items():
        if key in json_enc_params and type(value) == str:
            encoded_dict[key] = json.loads(value)
    return encoded_dict


def get_ray_tune_trial_dirs(base_dir: str, trial_dirs):
    """ returns all output directories of individual ray.tune trials """
    if "params.json" in os.listdir(base_dir):
        trial_dirs.append(base_dir)
    else:
        for d in os.scandir(base_dir):
            if os.path.isdir(d):
                get_ray_tune_trial_dirs(d, trial_dirs)
        return trial_dirs


def get_lastest_checkpoint(trial_dir: str):
    checkpoints = [
        ckpt_dir
        for ckpt_dir in os.scandir(trial_dir)
        if os.path.isdir(ckpt_dir) and "checkpoint" in ckpt_dir.path
    ]

    sorted_cps = sorted(checkpoints, key=lambda d: d.path)
    return sorted_cps[-1]


def get_model_ckpt_paths(
    hyperopt_training_stats: list, output_dir: str, executor: str = "ray"
):
    """
    maps output of individual tial run statistics to associated
    output directories. Necessary for accessing model checkpoints
    """
    if executor == "ray":  # folder construction is different
        hyperopt_run_metadata = []
        # populate paths
        trial_dirs = []
        for path in os.scandir(output_dir):
            if os.path.isdir(path):
                trial_dirs.extend(get_ray_tune_trial_dirs(path, []))
        for hyperopt_run in hyperopt_training_stats:
            hyperopt_run_metadata.append(
                {
                    "hyperopt_results": decode_json_enc_dict(
                        hyperopt_run,
                        ["parameters", "training_stats", "eval_stats"],
                    ),
                    "model_path": None,
                }
            )

        for hyperopt_run in hyperopt_run_metadata:
            hyperopt_params = hyperopt_run["hyperopt_results"]["parameters"]
            for path in trial_dirs:
                config_json = json.load(
                    open(os.path.join(path, "params.json"))
                )
                if compare_json_enc_configs(hyperopt_params, config_json):
                    model_path = get_lastest_checkpoint(path)
                    hyperopt_run["model_path"] = os.path.join(
                        model_path, "model"
                    )
    else:
        hyperopt_run_metadata = []
        for run_dir in os.scandir(output_dir):
            if os.path.isdir(run_dir):
                sample_training_stats = json.load(
                    open(
                        os.path.join(run_dir.path, "training_statistics.json"),
                        "rb",
                    )
                )
                for hyperopt_run in hyperopt_training_stats:
                    if hyperopt_run["training_stats"] == sample_training_stats:
                        hyperopt_run_metadata.append(
                            {
                                "hyperopt_results": hyperopt_run,
                                "model_path": os.path.join(
                                    run_dir.path, "model"
                                ),
                            }
                        )

    return hyperopt_run_metadata


def collect_completed_trial_results(output_dir: str):
    results, metrics, params = [], [], []
    trial_dirs = get_ray_tune_trial_dirs(output_dir, [])
    for trial_dir in trial_dirs:
        for f in os.scandir(trial_dir):
            if "progress" in f.name:
                try:
                    progress = pd.read_csv(f)
                    last_iter = len(progress) - 1
                    last_iter_eval_stats = json.loads(
                        progress.iloc[last_iter]["eval_stats"]
                    )
                    if (
                        "overall_stats"
                        in last_iter_eval_stats[
                            list(last_iter_eval_stats.keys())[0]
                        ].keys()
                    ):
                        trial_results = decode_json_enc_dict(
                            progress.iloc[last_iter].to_dict(),
                            ["parameters", "training_stats", "eval_stats"],
                        )
                        trial_results["done"] = True
                        metrics.append(
                            progress.iloc[last_iter]["metric_score"]
                        )
                        curr_path = f.path
                        params_path = curr_path.replace(
                            "progress.csv", "params.json"
                        )
                        trial_params = json.load(open(params_path, "rb"))
                        params.append(trial_params)
                        for key, value in trial_params.items():
                            config_key = "config" + "." + key
                            trial_results[config_key] = value
                        results.append(trial_results)
                except:
                    pass
    return results, metrics, params


def conditional_decorator(decorator, condition, *args):
    def wrapper(function):
        if condition(*args):
            return decorator(function)
        else:
            return function

    return wrapper
