import abc
from abc import ABC
import pandas as pd


class BaseSubpopulation(ABC):
    def __init__(self, name):
        self.name = name

    @abc.abstractmethod
    def score_fn(self):
        """ scores a sample based on subpopulation the sample is a part of """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_subpops(self):
        raise NotImplementedError()

    @property
    def slice_name(self):
        return self.name