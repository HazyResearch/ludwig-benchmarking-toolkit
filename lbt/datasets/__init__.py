import importlib
import inspect

from lbt.datasets.base_dataset import LBTDataset
from ludwig.datasets.base_dataset import BaseDataset

DATASET_REGISTRY = {}


def register_dataset(name):
    """
    New dataset types can be added to LBT with the `register_dataset`
    function decorator.
    :
        @register_dataset('personal_dataset')
        class PersonalDataset():
            (...)
    Args:
        name (str): the name of the dataset
    """

    def register_dataset_cls(cls):
        if not issubclass(cls, LBTDataset):
            raise ValueError(
                "Dataset ({}: {}) must extend lbt.base_datast.LBTDataset".format(
                    name, cls.__name__
                )
            )
        DATASET_REGISTRY[name] = cls
        return cls

    return register_dataset_cls


def build_dataset(dataset_name: str, cache_dir: str, **kwargs):
    if dataset_name not in DATASET_REGISTRY:
        if dataset_name in PRE_BUILT_DATASETS:
            importlib.import_module(PRE_BUILT_DATASETS[dataset_name])
        else:
            raise ValueError(
                "Dataset ({}) is not supported by LBT".format(dataset_name)
            )
            exit(1)

    dataset = DATASET_REGISTRY[dataset_name](cache_dir=cache_dir, **kwargs)
    dataset.load()
    return dataset


PRE_BUILT_DATASETS = {
    "AGNews": "ludwig.datasets.agnews",
    "SST5": "ludwig.datasets.sst5",
    "GoEmotions": "ludwig.datasets.goemotions",
    "Fever": "ludwig.datasets.fever",
    "SST2": "ludwig.datasets.sst2",
    "EthosBinary": "ludwig.datasets.ethos_binary",
    "YelpPolarity": "ludwig.datasets.yelp_review_polarity",
    "DBPedia": "ludwig.datasets.dbpedia",
    "Irony": "ludwig.datasets.irony",
    "YelpReviews": "ludwig.datasets.yelp_reviews",
    "YahooAnswers": "ludwig.datasets.yahoo_answers",
    "AmazonPolarity": "ludwig.datasets.amazon_review_polarity",
    "AmazonReviews": "ludwig.datasets.amazon_reviews",
    "HateSpeech": "ludwig.datasets.hate_speech",
    "MDGenderBias": "ludwig.datasets.md_gender_bias",
    "toyAGNews": "lbt.datasets.toy_datasets",
    "Mnist" : "ludwig.datasets.mnist",
    "CIFAR10" : "ludwig.datasets.cifar10",
}

# TODO: ASN -> CHECK PLACEMENT
for dataset_name, module_path in PRE_BUILT_DATASETS.items():
    module = importlib.import_module(module_path)
    for obj in dir(module):
        if obj != "BaseDataset" and inspect.isclass(getattr(module, obj)):
            if issubclass(getattr(module, obj), BaseDataset):
                DATASET_REGISTRY[dataset_name] = getattr(module, obj)
