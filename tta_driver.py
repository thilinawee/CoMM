import os
from typing import Tuple 

import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import numpy as np
import torch
from torch.utils import model_zoo

from network.wide_resnet import WideResNet
from network.resnet import resnet18, resnet50, model_urls
from utils import prepare_cifar_loader, prepare_imagenet_loader, CORRUPTIONS
from logger.logger import TTALogger


logger = TTALogger(__file__)


class TTADriver:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    @staticmethod
    def init_random_seeds(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

    def read_params(self):
        """
        Read parameters from the user
        """
        ...

    def prepare_dataset(self, name: str, method: str):
        """
        Prepare the dataset according to the given distribution.
        """
        ...

    def prepare_data_loader(self, dataset):
        """
        Creates a data loader from a given dataset
        """

    def get_model(self, args):
        """
        Get the pretrained model
        """
        device = self.device
        # Load model
        if args.source_dataset.upper() == "CIFAR-10":
            num_classes = 10
            model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, bias_last=True).to(device)
            state_dict = torch.load(args.model_path, map_location = device)
            _ = model.load_state_dict(state_dict, strict=True)
            logger.info(f"Model loaded from {args.model_path}, {_}")

        elif args.source_dataset.upper() == "CIFAR-100":
            num_classes = 100
            model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, bias_last=True).to(device)
            state_dict = torch.load(args.model_path)
            _ = model.load_state_dict(state_dict, strict=True)
            print(f"[INFO] Model loaded from {args.model_path}, {_}")

        elif args.source_dataset.upper() == "IMAGENET":

            assert args.network.find("resnet") != -1, f"[INFO] Model must be ResNet for ImageNet"

            num_classes = 1000

            # ResNet-18
            model = resnet18(pretrained=False, classes=num_classes).to(device)
            # args.lr = 0.005 * (args.tta_batchsize / 128)  # RN18
            if args.model_path is not None:
                state_dict = torch.load(args.model_path)
                _ = model.load_state_dict(state_dict, strict=True)
                print(f"[INFO] Model loaded from {args.model_path}, {_}")
            elif args.model_path is None:
                state_dict = model_zoo.load_url(model_urls['resnet18'])
                _ = model.load_state_dict(state_dict, strict=True)
                print(f"[INFO] Model loaded from Torchvision URL for ResNet18, {_}")

            else:
                raise Exception(f"[INFO] Architecture {args.network} not supported.")

        else:
            raise Exception(f"[INFO] Invalid dataset: {args.source_dataset}")
        
        self.model = model

    def prepare_data_loaders_for_sota_env(self, args) -> Tuple[torch.utils.data.DataLoader]: 
        # Get dataloaders (OOD)
        data_path = os.path.join(args.data_path, args.target_dataset.upper())
        if not os.path.exists(data_path):
            raise Exception(f"[INFO] Dataset not found at {data_path}")
        batch_size = args.tta_batchsize

        if args.source_dataset.upper().find("CIFAR") != -1:
            tta_train_loaders = prepare_cifar_loader(data_path=data_path, train=True, batch_size=batch_size,
                                                    severity=args.severity, corruptions=CORRUPTIONS)
            tta_test_loaders = prepare_cifar_loader(data_path=data_path, train=False, batch_size=1024,
                                                    severity=args.severity, corruptions=CORRUPTIONS)

        elif args.source_dataset.upper().find("IMAGENET") != -1:
            tta_train_loaders = prepare_imagenet_loader(data_path=data_path, train=True, batch_size=batch_size,
                                                        severity=args.severity, corruptions=CORRUPTIONS)
            tta_test_loaders = prepare_imagenet_loader(data_path=data_path, train=False, batch_size=512,
                                                    severity=args.severity, corruptions=CORRUPTIONS)
            
        return tta_train_loaders, tta_test_loaders
    
    def apply_tta_for_sota_env(self, args):
        """
        Applies the test time adaptation algorithm to the original dataset with distribution shifts.
        """
        ...

    def test_for_sota_env(self):
        """
        Evaluate the adapted sota.
        """
        ...

    def prepare_data_loaders_for_novel_env(self): ...
    def apply_tta_for_novel_env(self):
        """
        Applies the test time adaptation algorithm to the label imbalanced dataset with distribution shifts.
        """
        ...

    def test_for_novel_env(self): ...

    def display_params(self, args):
        """
        Display the important parameters of the input script.
        """
        # print args
        for arg, value in vars(args).items():
            logger.info(f"{arg:<30}:  {str(value)}")
        print("--------------------  Initializing TTA  --------------------")

    def collect_results_for_sota_env(self):
        """
        Collect the outcomes of the experiment.
        """
        ...

    def collect_results_for_novel_env(self): ...
