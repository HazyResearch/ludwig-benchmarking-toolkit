from database import Database, save_results_to_es
from utils.experiment_utils import *

# from experiment_driver import map_runstats_to_modelpath
import pickle
import os
import json
from utils.metadata_utils import append_experiment_metadata
import ray

ray.init(address="auto")

elastic_config_file = "./elasticsearch_config.yaml"
path_to_stats_file = "/experiments/ludwig-bench-textclassification/experiment-outputs/agnews_rnn/agnews_rnn_hyperopt_results.pkl"
path_to_output_dir = "/experiments/ludwig-bench-textclassification/experiment-outputs/agnews_rnn/"
path_to_dataset = "/experiments/ludwig-bench-textclassification/data/agnews_1.0/processed/agnews.csv"
path_to_model_config = "/experiments/ludwig-bench-textclassification/experiment-configs/config_agnews_rnn.yaml"


def main():
    elastic_config = None
    elastic_config = load_yaml(elastic_config_file)
    model_config = load_yaml(path_to_model_config)
    experiment_attr = {
        "model_config": copy.deepcopy(model_config),
        "dataset_path": path_to_dataset,
        "top_n_trials": None,
        "model_name": "config_agnews_rnn",
        "output_dir": path_to_output_dir,
        "encoder": "rnn",
        "dataset": "agnews",
        "elastic_config": elastic_config,
    }
    hyperopt_results = pickle.load(open(path_to_stats_file, "rb"))
    save_results_to_es(experiment_attr, hyperopt_results, "ray")


if __name__ == "__main__":
    main()