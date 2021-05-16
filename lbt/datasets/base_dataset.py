from ludwig.datasets.base_dataset import BaseDataset, DEFAULT_CACHE_LOCATION
import abc
import pandas as pd


class LBTDataset(BaseDataset):
    """Base LBT Dataset -- subclass wrapper around Ludwig data class"""

    def __init__(self, dataset_name, processed_file_name, cache_dir):
        self.name = dataset_name
        self.config = {"csv_filename": processed_file_name}
        self.cache_dir = cache_dir

    @abc.abstractmethod
    def download(self) -> None:
        """ Download the file from config url that represents the raw unprocessed training data."""
        raise NotImplementedError()

    @abc.abstractmethod
    def process(self) -> None:
        """Process the dataset to get it ready to be plugged into a dataframe.
        Converts into a format to be used by the ludwig training API. To do this we create
        a new dictionary that contains the KV pairs in the format that we need.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self) -> pd.DataFrame:
        """ Load the processed data into a Pandas DataFrame """
        raise NotImplementedError()

    @property
    def processed_dataset_path(self) -> str:
        """ Return path of the processed dataset """
        raise NotImplementedError()

    def __repr__(self):
        return "{}()".format(self.name)
