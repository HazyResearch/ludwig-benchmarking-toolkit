import os
import pdb
import pandas as pd
from lbt.datasets import register_dataset
from lbt.datasets.base_dataset import LBTDataset


@register_dataset("toy_agnews")
class ToyAGNews(LBTDataset):
    def __init__(
        self,
        dataset_name="toy_agnews",
        processed_file_name="toy_agnews.csv",
        cache_dir=os.path.join(os.getcwd(), "lbt/datasets/toy-datasets"),
    ):
        super().__init__(
            dataset_name=dataset_name,
            processed_file_name=processed_file_name,
            cache_dir=os.path.join(os.getcwd(), "lbt/datasets/toy-datasets"),
        )

    def download(self) -> None:
        pass

    def process(self) -> None:
        pass

    def load(self) -> pd.DataFrame:
        toy_agnews_ds = pd.read_csv(
            os.path.join(self.cache_dir, self.config["csv_filename"])
        )
        return toy_agnews_ds

    @property
    def processed_dataset_path(self):
        return self.cache_dir
