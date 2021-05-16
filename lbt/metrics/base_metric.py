import abc
from abc import ABC, ABCMeta, abstractmethod
from typing import Tuple, Union

import pandas as pd
from ludwig.api import LudwigModel


class BaseMetric(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def run(self, model_path, dataset_path, train_batch_size, run_stats):
        pass

    def load_model(self, model_path: str) -> LudwigModel:
        return LudwigModel.load(model_path)

    def evaluate(
        self,
        model: LudwigModel,
        dataset: Union[str, dict, pd.DataFrame] = None,
        **kwargs
    ) -> Tuple[dict, Union[dict, pd.DataFrame], str]:
        return model.evaluate(dataset, **kwargs)

    def predict(
        self,
        model: LudwigModel,
        dataset: Union[str, dict, pd.DataFrame] = None,
        **kwargs
    ) -> Tuple[Union[dict, pd.DataFrame], str]:
        return model.predict(dataset, **kwargs)
