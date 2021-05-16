import datetime

import GPUtil
import ludwig
import pandas as pd
import psutil
import ray
from lbt.metrics.utils import scale_bytes
from lbt.metrics import register_metric
from lbt.metrics.base_metric import BaseMetric
from ludwig.collect import collect_weights

# TODO: ASN --> Add check to see if available GPUs before seting num_gpus=1


@register_metric("ludwig_version")
class LudwigVersion(BaseMetric):
    @ray.remote
    def run(cls, **kwargs):
        return ludwig.__version__


@register_metric("hardware_metadata")
class HardwareMetadata(BaseMetric):
    num_gpus = 1

    @ray.remote(num_gpus=num_gpus, num_returns=1)
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
class InferenceLatencyMetric(BaseMetric):
    num_samples = 20
    num_gpus = 1

    def __init__(self):
        pass

    @ray.remote(num_gpus=num_gpus, num_returns=1, max_calls=1)
    def run(cls, model_path, dataset_path, **kwargs):
        """
        Returns avg. time to perform inference on 1 sample

        # Inputs
        :param model_path: (str) filepath to pre-trained model (directory that
            contains the model_hyperparameters.json).
        :param dataset_path: (str) filepath to dataset
        :param dataset_path: (int) number of dev samples to randomly sample

        # Return
        :return: (str) avg. time per training step
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
        ludwig_model = cls.load_model(model_path)
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
class TrainingCost(BaseMetric):
    gpu_cost_per_hr = 0.35  # GCP cost for Tesla T4

    @ray.remote(num_returns=1)
    def run(cls, run_stats: dict, **kwargs) -> float:
        """
        Return total cost to train model using GCP compute resource
        """
        total_time_s = int(run_stats["hyperopt_results"]["time_total_s"])
        total_time_hr = total_time_s / 3600
        return float(total_time_hr * cls.gpu_cost_per_hr)


@register_metric("training_speed")
class TrainingSpeed(BaseMetric):
    num_gpus = 1

    @ray.remote(num_gpus=1, num_returns=1, max_calls=1)
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
class ModelSize(BaseMetric):
    num_gpus = 1

    @ray.remote(num_gpus=num_gpus, num_returns=1, max_calls=1)
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
