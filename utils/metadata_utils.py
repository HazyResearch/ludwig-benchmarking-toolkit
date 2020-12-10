import os
import platform
import datetime

import GPUtil
import ludwig
import numpy as np
import pandas as pd
import psutil
from ludwig.api import LudwigModel
from ludwig.collect import collect_weights
import tensorflow as tf


def get_ludwig_version():
    return ludwig.__version__

def scale_bytes(bytes: int, suffix: str = 'B') -> str:
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]: 
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_hardware_metadata() -> dict:
    """Returns GPU, CPU and RAM information"""

    machine_info = {} 
    # GPU 
    gpus = GPUtil.getGPUs()
    if len(gpus) != 0:
        machine_info['total_gpus'] = len(gpus)
        gpu_type = {}
        for gpu_id, gpu in enumerate(gpus):
            gpu_type[gpu_id] = gpu.name
        machine_info['gpu_info'] = gpu_type
    else: 
        machine_info['total_gpus'] = 0
    # CPU 
    total_cores = psutil.cpu_count(logical=True)
    machine_info['total_cores'] = total_cores
    # RAM
    svmem = psutil.virtual_memory()
    total_RAM = scale_bytes(svmem.total)
    machine_info['RAM'] = total_RAM
    return machine_info

def get_inference_latency(
	model_path: str, 
	dataset_path: str, 
	num_samples: int = 10
) -> str:
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
    inference_dataset = full_dataset[full_dataset['split'] == 1].sample(
                                                                n=num_samples)
    ludwig_model = LudwigModel.load(model_path)
    start = datetime.datetime.now()
    _, _ = ludwig_model.predict(
        dataset=inference_dataset,
        batch_size=1,
    )
    total_time = datetime.datetime.now() - start
    avg_time_per_sample = total_time/num_samples
    formatted_time = "{:0>8}".format(str(avg_time_per_sample))
    return formatted_time

def get_train_speed(
    model_path: str, 
    dataset_path: str, 
    train_batch_size: int
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
    ludwig_model = LudwigModel.load(model_path)
    start = datetime.datetime.now()
    ludwig_model.train_online(
        dataset=dataset_path,
    )
    total_time = datetime.datetime.now() - start
    avg_time_per_minibatch = total_time/train_batch_size
    formatted_time = "{:0>8}".format(str(avg_time_per_minibatch))
    return formatted_time

def model_flops(model_path: str) -> int:
    """
    Computes total model flops

    # Inputs
    :param model_path: (str) filepath to pre-trained model.

    # Return
    :return: (int) total number of flops.
    """
    tf.compat.v1.reset_default_graph()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = LudwigModel.load(model_path)
            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, 
                                                  cmd='op',
                                                  options=opts)
        
            return flops.total_float_ops

def get_model_size(model_path: str) -> (int, str):
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
        output_directory='.model_tensors'
    )
    total_size = 0
    for fp in tensor_filepaths:
        weight_tensor = np.load(fp)
        total_size += weight_tensor.size
    total_bytes = total_size * 32
    scaled_bytes = scale_bytes(total_bytes)
    return total_bytes, scaled_bytes

