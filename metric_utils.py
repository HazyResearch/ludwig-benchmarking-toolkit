import os
import platform
from datetime import datetime

import GPUtil
import pandas as pd
import psutil
from ludwig.api import LudwigModel
from tensorflow import tf


def scale_bytes(bytes, suffix='B'):
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]: 
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor

def get_hardware_metadata():
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
    "Returns avg. time to perform inference on 1 sample "
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
    "Returns avg. time to train model on a given mini-batch size"
    ludwig_model = LudwigModel.load(model_dir)
    start = datetime.datetime.now()
    ludwig_model.train_online(
        dataset=dataset_dir,
    )
    total_time = datetime.datetime.now() - start
    avg_time_per_minibatch = total_time/train_batch_size
    formatted_time = "{:0>8}".format(str(avg_time_per_minibatch))
    return formatted_time

def model_flops(checkpoint_dir):
    pass

    """latest = tf.train.latest_checkpoint(checkpoint_dir)

    #load model
    tf.compat.v1.reset_default_graph()
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_weights(latest)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, 
                                                  cmd='op',
                                                  options=opts)
        
            return flops.total_float_ops"""


