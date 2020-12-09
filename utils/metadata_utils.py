import os
import platform
from datetime import datetime

import GPUtil
import ludwig
import pandas as pd
import psutil
from ludwig.api import LudwigModel
from tensorflow import tf


def get_ludwig_version():
    return ludwig.__version__

def scale_bytes(bytes, suffix='B'):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]: 
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_hardware_metadata():
    """Returns GPU, CPU and RAM information"""
    # NOTE: NOT TESTED 

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

def get_inference_latency(model_dir, dataset_dir, num_samples=10):
    """Returns avg. time to perform inference on 1 sample"""
    # NOTE: NOT TESTED 

    # Create smaller datasets w/10 samples
    full_dataset = pd.read_csv(dataset_dir)
    # split == 1 indicates the dev set
    inference_dataset = full_dataset[full_dataset['split'] == 1].sample(
                                                                n=num_samples)
    ludwig_model = LudwigModel.load(model_dir)
    start = datetime.datetime.now()
    _, _ = ludwig_model.predict(
        dataset=inference_dataset,
        batch_size=1,
    )
    total_time = datetime.datetime.now() - start
    avg_time_per_sample = total_time/num_samples
    formatted_time = "{:0>8}".format(str(avg_time_per_sample))
    return formatted_time

def get_train_speed(model_dir, dataset_dir, train_batch_size):
    """
    NOTE: NOT TESTED
    Returns avg. time to train model on a given mini-batch size
    model_dir: path to directory which contains model_hyperparameters.json
    dataset_dir: path to directory which contains dataset file
    """
    ludwig_model = LudwigModel.load(model_dir)
    start = datetime.datetime.now()
    ludwig_model.train_online(
        dataset=dataset_dir,
    )
    total_time = datetime.datetime.now() - start
    avg_time_per_minibatch = total_time/train_batch_size
    formatted_time = "{:0>8}".format(str(avg_time_per_minibatch))
    return formatted_time

def model_flops(model_dir):
    """
    NOTE: NOT TESTED 
    """
    tf.compat.v1.reset_default_graph()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = LudwigModel.load(model_dir)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, 
                                                  cmd='op',
                                                  options=opts)
        
            return flops.total_float_ops


