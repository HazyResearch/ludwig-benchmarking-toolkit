# Ludwig Benchmark
A framework for running large-scale comparative analysis across common deep learning NLP architectures using Ludwig.

## **Getting set-up**
To get started, use the following commands to set-up your conda environment. 
```
git clone https://github.com/HazyResearch/ludwig-benchmark.git
cd ludwig-benchmark
conda env create -f environments/{environment-osx.yaml, environment-linux.yaml}
conda activate ludwig-bench
```

## **Relevant files and directories:**
`experiment-templates/model_template.yaml`: Every task (i.e. text classification) will have its owns model template. The template specifies the model architecture (encoder and decoder structure), training parameters, and a hyperopt configuration. A large majority of the values of the template will be populated by the values in the hyperopt_config.yaml file and dataset_metadata.yaml at training time.

`experiment-templates/hyperopt_config.yaml`: provides a range of values for training parameters and hyperopt params that will populate the hyperopt configuration in the model template

`experiment-templates/dataset_metadata.yaml`: contains list of all available datasets (and associated metadata) that the hyperparameter optimization can be performed over.

`encoder-configs`: contains all encoder specific yaml files. Each files specifies possible values for relevant encoder parameters that will be optimized over. Each file in this directory adheres to the naming convention {encoder_name}_hyperopt.yaml

`hyperopt-experiment-configs`: houses all experiment configs built from the templates specified above (note: this folder will be populated at runtime) and will be used when the hyperopt experiment is called. At a high level, each config file specifies the training and hyperopt information for a (task, dataset, architecture) combination. An example might be (text classification, SST2, BERT)

`elasticsearch_config.yaml `: this is an optional file that is to be defined if an experiment data will be saved to an elastic database.


## **Running an experiment:**

### *Running your first DUMMY experiment*:

For testing/setup purposes we have included a set of datasets which we refer to as the "smoke" datasets. Smoke datasets are comprised of samples from an original datasets in the datasets list. For example the smoke sst5 datasets contains a small set of training, test and val samples from the original SST5 datasets. If you would like to use one of the smoke datasets, simply set `--datasets` param to "smoke".

Before running a full-scale experiment, we recommend running an experiment locally on one of the smoke datasets:
```
python experiment_driver.py --run_environment local --datasets smoke --custom_encoders_list rnn
```

### *Running your first REAL experiment*:

To run experiment an experiment do the following:
1. Declare and configure the search space of your training and preprocessing hyperparameters in the `experiment-templates/hyperopt_config.yaml` file
2. Declare and configure the search space of your  encoder specific hyperparams in the `{encoder}_hyperopt.yaml` files in `./encoder_configs` 

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




