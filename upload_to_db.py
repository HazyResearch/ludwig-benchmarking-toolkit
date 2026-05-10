import ray
from database import Database, save_results_to_es
from utils.experiment_utils import *

# from experiment_driver import map_runstats_to_modelpath
import pickle
import os
import json
from utils.metadata_utils import append_experiment_metadata

ray.init(address="auto")

datasets = ["agnews"]
encoders = ["rnn", "distilbert", "t5", "electra"]

elastic_config_file = "./elasticsearch_config.yaml"
paths_to_dataset = {
    "agnews": "/experiments/ludwig-bench-textclassification/data/agnews_1.0/processed/agnews.csv"
}


def main():
    elastic_config = None
    elastic_config = load_yaml(elastic_config_file)

    exp_info = []
    for dataset in datasets:
        for enc in encoders:
            path_to_stats_file = f"/experiments/ludwig-bench-textclassification/experiment-outputs/{dataset}_{enc}/{dataset}_{enc}_hyperopt_results.pkl"
            path_to_output_dir = f"/experiments/ludwig-bench-textclassification/experiment-outputs/{dataset}_{enc}/"
            path_to_model_config = f"/experiments/ludwig-bench-textclassification/experiment-configs/config_{dataset}_{enc}.yaml"
            model_config = load_yaml(path_to_model_config)
            path_to_dataset = paths_to_dataset[dataset]
            experiment_attr = {
                "model_config": copy.deepcopy(model_config),
                "dataset_path": path_to_dataset,
                "top_n_trials": None,
                "model_name": f"config_{dataset}_{enc}",
                "output_dir": path_to_output_dir,
                "encoder": enc,
                "dataset": dataset,
                "elastic_config": elastic_config,
            }
            hyperopt_results = pickle.load(open(path_to_stats_file, "rb"))
            exp_info.append((experiment_attr, hyperopt_results))

    outputs = ray.get(
        [
            save_results_to_es.remote(info[0], info[1], "ray")
            for info in exp_info
        ]
    )


if __name__ == "__main__":
    main()