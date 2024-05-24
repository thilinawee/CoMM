from typing import List

import torch
import torchvision
from torch.utils.data import Subset


class LabelDistributer:
    def __init__(
        self,
        dataset: torchvision.datasets,
    ) -> None:
        """
        Base class for creating label distributions
        """
        self._dataset = dataset

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value


class ClassDropDistributer(LabelDistributer):
    """
    Drops selected classes from the given dataset
    """

    def __init__(self, dataset: torchvision.datasets, drop_list: List) -> None:

        super().__init__(dataset)

        self.drop_list = drop_list

    def generate_indices(self) -> List[int]:

        include_indices = []

        for ii, (_, label) in enumerate(self.dataset):
            if label not in self.drop_list:
                include_indices.append(ii)

        return include_indices

    def generate_dataset(self) -> torchvision.datasets:

        indices = self.generate_indices()
        dataset = Subset(self.dataset, indices)

        return dataset
