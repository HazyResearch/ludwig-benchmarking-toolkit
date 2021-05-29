# Ludwig Benchmarking Toolkit
The Ludwig Benchmarking Toolkit is a personalized benchmarking toolkit for running end-to-end benchmark studies across an extensible set of tasks, deep learning models, standard datasets and evaluation metrics.

# Getting set-up
To get started, use the following commands to set-up your conda environment. 
```
git clone https://github.com/HazyResearch/ludwig-benchmark.git
cd ludwig-benchmark
conda env create -f environments/{environment-osx.yaml, environment-linux.yaml}
conda activate ludwig-bench
```

# Relevant files and directories
`experiment-templates/task_template.yaml`: Every task (i.e. text classification) will have its owns task template. The template specifies the model architecture (encoder and decoder structure), training parameters, and a hyperopt configuration for the task at hand. A large majority of the values of the template will be populated by the values in the hyperopt_config.yaml file and dataset_metadata.yaml at training time. The sample task template located in `experiment-templates/task_template.yaml` is for text classification. See `sample-task-templates/` for other examples.

`experiment-templates/hyperopt_config.yaml`: provides a range of values for training parameters and hyperopt params that will populate the hyperopt configuration in the model template

`experiment-templates/dataset_metadata.yaml`: contains list of all available datasets (and associated metadata) that the hyperparameter optimization can be performed over.

`model-configs/`: contains all encoder specific yaml files. Each files specifies possible values for relevant encoder parameters that will be optimized over. Each file in this directory adheres to the naming convention {encoder_name}_hyperopt.yaml

`hyperopt-experiment-configs/`: houses all experiment configs built from the templates specified above (note: this folder will be populated at run-time) and will be used when the hyperopt experiment is called. At a high level, each config file specifies the training and hyperopt information for a (task, dataset, architecture) combination. An example might be (text classification, SST2, BERT)

`elasticsearch_config.yaml `: this is an optional file that is to be defined if an experiment data will be saved to an elastic database.


# USAGE
### **Command-Line Usage**

### *Running your first TOY experiment*:

For testing/setup purposes we have included a toy dataset called toy_agnews. This dataset contains a small set of training, test and validation samples from the original agnews dataset. 

Before running a full-scale experiment, we recommend running an experiment locally on the toy dataset:
```
python experiment_driver.py --run_environment local --datasets toy_agnews --custom_models_list rnn
```

### *Running your first REAL experiment*:

Steps for configuring + running an experiment:
1. Declare and configure the search space of all non-model specific training and preprocessing hyperparameters in the `experiment-templates/hyperopt_config.yaml` file. The parameters specified in this file will be used across all model experiments.
2. Declare and configure the search space of model specific hyperparameters in the `{encoder}_hyperopt.yaml` files in `./model_configs` 

    **NOTE**: 
    * for both (1) and (2) see the [Ludwig Hyperparamter Optimization guide](https://ludwig-ai.github.io/ludwig-docs/user_guide/#hyper-parameter-optimization) to see what parameters for training, preprocessing, and input/ouput features
    can be used in the hyperopt search
    * if the exectuor type is `Ray` the list of available search spaces and input format differs slightly than the built-in ludwig types. Please see the [Ray Tune search space docs](https://docs.ray.io/en/master/tune/api_docs/search_space.html) for more information.

3. Run the following command specifying the datasets, encoders, path to elastic DB index config file, run environment and more:

    ```
        python experiment_driver.py \
            --experiment_output_dir  <path to dir to save experiment outputs>
            --run_environment {local, gcp}
            --elasticsearch_config <path to config file>
            --dataset_cache_dir <path to dir to save downloaded datasets>
            --custom_encoders_list <list of encoders>
            --datasets <list of datasets>
            --resume_existing_exp bool

    ``` 

**NOTE:** Please use `python experiment_driver.py -h` to see list of available datasets, encoders and args

### **API Usage**
It is also possible to run, customize and experiments using LBTs APIs. In the following section,
we describe the three flavors of APIs included in LBT.

### `experiment` API
This API provides an alternative method for running experiments. Note that runnin experiments via the API still requires populating the aforemented configuration files

```python
from lbt.experiments import experiment

experiment(
    models = ['rnn', 'bert'],
    datasets = ['agnews'],
    run_environment = "local",
    elastic_search_config = None,
    resume_existing_exp = False,
)
```

### `tools` API
This API provides access to two tooling integrations (TextAttack and Robustness Gym (RG)). The TextAttack API can be used to generate adversarial attacks. Moreover, users can use the TextAttack interface to augment data files. The RG API which empowers users to inspect model performance on a set of generic, pre-built slices and to add more slices for their specific datasets and use cases. 

```python
from lbt.tools.robustnessgym import RG 
from lbt.tools.textattack import attack, augment

# Robustness Gym API Usage
RG( 
    dataset_name="AGNews",
    models=["bert", "rnn"],
    path_to_dataset="agnews.csv", 
    subpopulations=[ "entities", "positive_words", "negative_words"])
)

# TextAttack API Usage
attack(dataset_name="AGNews", path_to_model="agnews/model/rnn_model",
    path_to_dataset="agnews.csv", attack_recipe=["CharSwapAugmenter"])

augment(dataset_name="AGNews", transformations_per_example=1
   path_to_dataset="agnews.csv", augmenter=["WordNetAugmenter"])
```

### `visualizations` API
This API provides out-of-the-box support for visualizations for learning behavior, model performance, and hyperparameter optimization using the training and evaluation statistics generated during model training

```python
import lbt.visualizations

# compare model performance
compare_performance_viz(
    dataset_name="toy_agnews",
    model_name="rnn",
    output_feature_name="class_index",
)

# compare training and validation trajectory
learning_curves_viz(
    dataset_name="toy_agnews",
    model_name="rnn",
    output_feature_name="class_index",
)

# visualize hyperoptimzation search
hyperopt_viz(
    dataset_name="toy_agnews",
    model_name="rnn",
    output_dir="."
)
```

# EXPERIMENT EXTENSIBILITY
### **Adding new custom datasets**

Adding custom dataset requires creating a new `LBTDataset` class and adding it
to the dataset registry. Creating an `LBTDataset` object requires implementing
three class methods: download, process and load. Please see the the [`ToyAGNews`](lbt/datasets/toy_datasets.py) dataset as an example.

### **Adding new metrics**

Adding custom evaluation metrics requires creating a new `LBTMetric` class and adding it
to the metrics registry. Creating an `LBTMetric` object requires implementing
the run class method which takes as potential inputs a path to a model directory, path to a dataset, training batch size, and training statistics. Please see the [`pre-built LBT metrics`](lbt/metrics/lbt_metrics.py) for examples.




