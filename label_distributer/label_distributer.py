from typing import List
from random import sample
import math

import torch
import torchvision
from torch.utils.data import Subset, Dataset

from logger.logger import TTALogger

logger = TTALogger(__file__)

class LabelDistributer:
    def __init__(
        self
    ) -> None:
        """
        Base class for creating label distributions
        """
        self._dataset = None

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def dataset(self, value):
        self._dataset = value


class ClassFilter(LabelDistributer):
    """
    Drop or include  selected classes from the given dataset
    """

    def __init__(self, class_list: List, drop: bool = True) -> None:

        super().__init__()
        self.class_list = class_list
        self._drop = drop

    def generate_indices(self) -> List[int]:

        include_indices = []

        if self._drop:
            # remove the classes in the class_list from the dataset
            for ii, (_, label) in enumerate(self.dataset):
                if label not in self.class_list:
                    include_indices.append(ii)
        else:
            for ii, (_, label) in enumerate(self.dataset):
                if label in self.class_list:
                    include_indices.append(ii)

        return include_indices

    def generate_dataset(self) -> Dataset:

        indices = self.generate_indices()
        dataset = Subset(self.dataset, indices)

        return dataset

    def __call__(self, dataset: Dataset) -> Dataset:
        self.dataset = dataset
        return self.generate_dataset()


class DownSamplingDistributer(LabelDistributer):
    """
    Reduces the number of images to a certain ratio while keeping the original ratio
    among classes
    """

    def __init__(self, ratio) -> None:

        super().__init__()
        self._ratio = ratio
        self._n_class_samples = {}

    @property
    def ratio(self):
        return self._ratio

    @ratio.setter
    def ratio(self, value):
        if value > 1:
            raise Exception(f"Downsampling ratio {value} should be < 1")

        self._ratio = value

    def generate_indices(self):

        class_indices = {}

        for ii, (_, label) in enumerate(self.dataset):
            if label in class_indices:
                class_indices[label].append(ii)
            else:
                class_indices[label] = [ii]

        sampled_indices = []

        for cls, idx_list in class_indices.items():
            n_new_samples = math.floor(len(idx_list) * self.ratio)
            new_sample_indices = sample(idx_list, n_new_samples)
            sampled_indices.extend(new_sample_indices)

            self._n_class_samples[cls] = n_new_samples

        return sampled_indices

    def generate_dataset(self) -> Dataset:

        indices = self.generate_indices()
        dataset = Subset(self.dataset, indices)

        return dataset

    def __call__(self, dataset: Dataset) -> Dataset:
        self.dataset = dataset
        return self.generate_dataset()

    def __repr__(self):
        string = ""
        string += f"Downsampling ratio: {self.ratio}\t"
        string += f"Original Total Samples: {len(self.dataset)}\t"
        string += f"New Samples per class: {list(self._n_class_samples.values())[0]}\n"
        return string
       