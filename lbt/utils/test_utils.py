from metadata_utils import *

DATAPATH = "/sailhome/avanika/.ludwig_cache/sst2_1.0/processed/sst2.csv"
MODEL_PATH = "/juice/scr/avanika/ludwig-benchmark-dev/ludwig-benchmark/experiment-outputs/sst2_bert/hyperopt_0_config_sst2_bert/model"

machine_info = get_hardware_metadata()
print(machine_info)

#model_flops = model_flops(MODEL_PATH)
#print(model_flops)

#model_size = get_model_size(MODEL_PATH)
#print(model_size)

#latency = get_inference_latency(MODEL_PATH, DATAPATH)
#print(latency)

print(DATAPATH)
train_speed = get_train_speed(MODEL_PATH, DATAPATH, train_batch_size=16)
print(train_speed)
