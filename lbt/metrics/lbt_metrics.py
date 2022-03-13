import datetime
import os
import shutil
import tempfile

import GPUtil
import ludwig
import numpy as np
import pandas as pd
import psutil
from experiment_impact_tracker.compute_tracker import ImpactTracker
from globals import ENERGY_LOGGING_DIR
from lbt.metrics import register_metric
from lbt.metrics import INSTANCE_PRICES
from lbt.metrics.base_metric import LBTMetric
from lbt.metrics.utils import scale_bytes
from ludwig.api import LudwigModel
from ludwig.collect import collect_weights


@register_metric("ludwig_version")
class LudwigVersion(LBTMetric):
    def __init__(self):
        pass

    def run(cls, **kwargs):
        return ludwig.__version__


@register_metric("hardware_metadata")
class HardwareMetadata(LBTMetric):
    num_gpus = 0

    def run(cls, **kwargs):
        machine_info = {}
        # GPU
        gpus = GPUtil.getGPUs()
        if len(gpus) != 0:
            machine_info["total_gpus"] = len(gpus)
            gpu_type = {}
            for gpu_id, gpu in enumerate(gpus):
                gpu_type[gpu_id] = gpu.name
            machine_info["gpu_info"] = gpu_type
        else:
            machine_info["total_gpus"] = 0
        # CPU
        total_cores = psutil.cpu_count(logical=True)
        machine_info["total_cores"] = total_cores
        # RAM
        svmem = psutil.virtual_memory()
        total_RAM = scale_bytes(svmem.total)
        machine_info["RAM"] = total_RAM
        return machine_info


@register_metric("inference_latency")
class InferenceLatencyMetric(LBTMetric):
    num_samples = 25
    num_gpus = 0

    def run(cls, model_path, dataset_path, **kwargs):
        """
        Returns avg. time to perform inference on 1 sample

        # Inputs
        :param model_path: (str) filepath to pre-trained model (directory that
            contains the model_hyperparameters.json).
        :param dataset_path: (str) filepath to dataset
        :param dataset_path: (int) number of dev samples to randomly sample

        # Return
        :return: (str) avg. time per inference step
        """
        # Create smaller datasets w/10 samples from original dev set
        full_dataset = pd.read_csv(dataset_path)
        # Note: split == 1 indicates the dev set
        if "split" in full_dataset:
            if len(full_dataset[full_dataset["split"] == 1]) > 0:
                sampled_dataset = full_dataset[
                    full_dataset["split"] == 1
                ].sample(n=cls.num_samples)
            elif len(full_dataset[full_dataset["split"] == 2]) > 0:
                sampled_dataset = full_dataset[
                    full_dataset["split"] == 2
                ].sample(n=cls.num_samples)
            else:
                sampled_dataset = full_dataset[
                    full_dataset["split"] == 0
                ].sample(n=cls.num_samples)
        else:
            sampled_dataset = full_dataset.sample(n=cls.num_samples)
        ludwig_model = LudwigModel.load(model_path)
        start = datetime.datetime.now()
        _, _ = ludwig_model.predict(
            dataset=sampled_dataset,
            batch_size=1,
        )
        total_time = datetime.datetime.now() - start
        avg_time_per_sample = total_time / cls.num_samples
        formatted_time = "{:0>8}".format(str(avg_time_per_sample))
        return formatted_time


@register_metric("training_cost")
class TrainingCost(LBTMetric):
    default_gpu_cost_per_hr = 0.35  # GCP cost for Tesla T4

    def run(cls, run_stats: dict, **kwargs) -> float:
        """
        Return total cost to train model using GCP compute resource
        """
        get_GPUS = GPUtil.getGPUs()
        instance_cost = None
        if len(get_GPUS) > 0:
            gpu_type = get_GPUS[0].name
            if gpu_type in INSTANCE_PRICES.keys():
                instance_cost = INSTANCE_PRICES[gpu_type]
        if instance_cost is None:
            instance_cost = cls.default_gpu_cost_per_hr

        total_time_s = int(run_stats["hyperopt_results"]["time_total_s"])
        total_time_hr = total_time_s / 3600
        return float(total_time_hr * instance_cost)


@register_metric("training_speed")
class TrainingSpeed(LBTMetric):
    num_gpus = 0

    def run(
        cls,
        dataset_path: str,
        train_batch_size: int,
        run_stats: dict,
        **kwargs,
    ) -> str:
        """
        Returns avg. time per training step

        # Inputs
        :param model_path: (str) filepath to pre-trained model (directory that
            contains the model_hyperparameters.json).
        :param dataset_path: (str) filepath to dataset

        # Return
        :return: (str) avg. time per training step
        """

        train_split_size = 0.7
        full_dataset = pd.read_csv(dataset_path)
        if "split" in full_dataset:
            total_samples = len(full_dataset[full_dataset["split"] == 0])
        else:
            total_samples = int(train_split_size * len(full_dataset))
        total_training_steps = int(total_samples / train_batch_size)
        time_per_batch = (
            int(run_stats["hyperopt_results"]["time_this_iter_s"])
            / total_training_steps
        )
        formatted_time = "{:0>8}".format(
            str(datetime.timedelta(seconds=time_per_batch))
        )
        return formatted_time


@register_metric("model_size")
class ModelSize(LBTMetric):
    num_gpus = 0

    def run(cls, model_path: str, **kwargs):
        """
        Computes minimum bytes required to store model to memory

        # Inputs
        :param model_path: (str) filepath to pre-trained model.

        # Return
        :return: (int) total bytes
        :return: (str) total bytes scaled in string format
        """
        tensor_filepaths = collect_weights(
            model_path=model_path,
            tensors=None,
            output_directory=".model_tensors",
        )
        total_size = 0
        for fp in tensor_filepaths:
            weight_tensor = np.load(fp)
            total_size += weight_tensor.size
        total_bytes = total_size * 32
        scaled_bytes = scale_bytes(total_bytes)
        model_size = {"total_bytes": total_bytes, "scaled_bytes": scaled_bytes}
        return model_size


@register_metric("carbon_footprint")
class Energy(LBTMetric):
    num_gpus = 0

    def run(cls, model_path: str, dataset_path, train_batch_size, run_stats):
        """
        Computes energy metrics for one training epoch

        # Inputs
        :param model_path: (str) filepath to pre-trained model.

        # Return
        :return: (int) total bytes
        :return: (str) total bytes scaled in string format
        """
        # First copy model_path to temp directory
        logging_path = os.path.join(
            ENERGY_LOGGING_DIR, run_stats["hyperopt_results"]["experiment_id"]
        )
        tempdir = os.path.join(logging_path, "temp_model")
        shutil.copytree(model_path, tempdir)
        model = LudwigModel.load(tempdir)

        with ImpactTracker(logging_path):
            model.train_online(dataset=dataset_path)

        data_interface = DataInterface([logging_path])
        carbon_output = {
            "kg_carbon": data_interface.kg_carbon,
            "total_power": data_interface.total_power,
            "PUE": data_interface.PUE,
            "duration_of_train_step": data_interface.exp_len_hours,
        }

        shutil.rmtree(tempdir)

        return carbon_output
