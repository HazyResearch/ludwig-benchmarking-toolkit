import importlib
import sys
import pdb

import ray
from lbt.metrics.base_metric import BaseMetric

METRIC_REGISTERY = {}


def register_metric(name):
    """
    New dataset types can be added to LBT with the `register_metric`
    function decorator.
    :
        @register_metric('personal_metric')
        class PersonalMetric():
            (...)
    Args:
        name (str): the name of the dataset
    """

    def register_metric_cls(cls):
        if not issubclass(cls, BaseMetric):
            raise ValueError(
                "Metric ({}: {}) must extend lbt.metrics.base_metric".format(
                    name, cls.__name__
                )
            )
        METRIC_REGISTERY[name] = cls
        return cls

    return register_metric_cls


def get_experiment_metadata(
    document: dict,
    model_path: str,
    data_path: str,
    run_stats: dict,
    train_batch_size: int = 16,
):
    for key, metrics_class in METRIC_REGISTERY.items():
        try:
            remote_class = ray.remote(metrics_class).remote()
            output = remote_class.run.remote(
                model_path=model_path,
                dataset_path=data_path,
                train_batch_size=train_batch_size,
                run_stats=run_stats,
            )
            document.update({key: ray.get(output)})

        except:
            print(f"failure processing: {key}")


PRE_BUILT_METRICS = {
    "lbt_metrics": "lbt.metrics.lbt_metrics",
}

for name, module in PRE_BUILT_METRICS.items():
    if module not in sys.modules:
        importlib.import_module("lbt.metrics.lbt_metrics")
