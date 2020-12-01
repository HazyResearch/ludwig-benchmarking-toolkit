# Ludwig Benchmark
A framework for running large-scale comparative analysis across common deep learning NLP architectures using Ludwig.

## **Relevant files and directories:**
`experiment-templates/model_template.yaml`: Every task (i.e. text classification) will have its owns model template. The template specifies the model architecture (encoder and decoder structure), training parameters, and a hyperopt configuration. A large majority of the values of the template will be populated at training time.

`experiment-templates/hyperopt_config.yaml`: provides a range of values for training parameters that will populate the hyperopt configuration in the model template

`experiment-templates/dataset_metadata.yaml`: lists the datasets (and associated metadata) that the hyperparameter optimization will be performed over.

`encoder-configs`: contains all encoder specific yaml files. Each files specifies possible values for relevant encoder parameters that will be optimized over. Each file in this directory adheres to the naming convention {encoder_name}_hyperopt.yaml

`hyperopt-experiment-configs`: houses all experiment configs built from the templates specified above (note: this folder will be populated at runtime). At a high level, each config file specifies the training and hyperopt information for a (task, dataset, architecture) combination. An example might be (text classification, SST2, BERT)

`elasticsearch_config.yaml `: this is an optional file that is to be defined if an experiment data will be saved to an elastic database.


## **Running an experiment:**


To run experiment an experiment, simply modify the aforementioned yaml files and run the following command:
 ```
    python experiment_driver.py \
        --hyperopt_config_dir <path to dir>
        --experiment_output_dir  <path to dir>
        --run_environment {local, gcp}
        --elasticsearch_config <path to config file>
        --dataset_cache_dir <path to dir>
        --custom_encoders_list <list>

 ``` 


