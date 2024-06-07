import os
from typing import Tuple
import random 

import torch.utils
import torch.utils.data
import torch.utils.data.dataloader
import torchvision
import numpy as np
import torch
from torch.utils import model_zoo
import torch.nn as nn

from network.wide_resnet import WideResNet
from network.resnet import resnet18, resnet50, model_urls
from utils import prepare_cifar_loader, prepare_imagenet_loader, CORRUPTIONS, test, \
                prepare_modified_cifar_loader
from methods import com
from label_distributer import ClassDropDistributer, DownSamplingDistributer
from report_gen import JsonDump, DirGen
from logger.logger import TTALogger
from tta_config import TTAConfig
from dataloader.cifar import CIFAR10Config, CIFAR100Config

logger = TTALogger(__file__)


class TTADriver:
    def __init__(self):
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._report_path = None

        self._args = TTAConfig.get_args()
        self._dataset_metadata = self._get_dataset_metadata()

    def set_gpu_id(self, gpu_id):
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    @staticmethod
    def init_random_seeds(seed):
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def _create_report_dirs(self, args):
        """
        Create directories to store the reports.
        """
        dir_gen = DirGen(args)
        dir_path = dir_gen.create_dir()
        
        self._report_path = dir_path

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

    def _get_dataset_metadata(self):
        dataset = self._args.source_dataset.upper()

        if dataset == "CIFAR-10":
            return CIFAR10Config()
        elif dataset == "CIFAR-100":
            return CIFAR100Config()
        else:
            raise Exception(f"[INFO] Invalid dataset: {dataset}")

    def _reset_model(self, args, model):
        state_dict = torch.load(args.model_path)
        _ = model.load_state_dict(state_dict, strict=True)
        logger.info(f"Resetting model to original state. {_}")

    def get_model(self, args) -> nn.Module:
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
            down_sample_ratio = (self._dataset_metadata.num_classes-len(args.drop_classes))/self._dataset_metadata.num_classes


            tta_train_loaders = prepare_modified_cifar_loader(
                data_path=data_path,
                label_distributer=DownSamplingDistributer(down_sample_ratio),
                train=True,
                batch_size=batch_size,
                severity=args.severity,
                corruptions=CORRUPTIONS,
            )

            tta_test_loaders = prepare_modified_cifar_loader(
                data_path=data_path,
                label_distributer=DownSamplingDistributer(1.0),
                train=False,
                batch_size=1024,
                severity=args.severity,
                corruptions=CORRUPTIONS,
            )
 
        elif args.source_dataset.upper().find("IMAGENET") != -1:
            tta_train_loaders = prepare_imagenet_loader(data_path=data_path, train=True, batch_size=batch_size,
                                                        severity=args.severity, corruptions=CORRUPTIONS)
            tta_test_loaders = prepare_imagenet_loader(data_path=data_path, train=False, batch_size=512,
                                                    severity=args.severity, corruptions=CORRUPTIONS)

        return tta_train_loaders, tta_test_loaders

    def apply_tta(self, args, ctx, tta_train_loaders, tta_test_loaders):
        """
        Applies the test time adaptation algorithm to the original dataset with distribution shifts.
        """
        device = self.device
        json_gen = JsonDump(self._report_path, ctx)
        tta_error = dict()
        for domain in tta_train_loaders.keys():

            self._reset_model(args, self.model)
            logger.info(f'domain - {domain}')

            tr_loader = tta_train_loaders[domain][str(args.severity)]
            te_loader = tta_test_loaders[domain][str(args.severity)]

            logger.info(f"----------  Corruption: {domain}, Severity: {args.severity}  ------------")

            if args.eval_before:
                # Compute before adaptation performance
                self.model.eval()
                before_loss, before_acc, before_cos_acc = test(self.model, te_loader, device)
                logger.info(f"Accuracy before adaptation - {before_acc}")
                json_gen.collect_corruption_data(domain, "original_acc", before_acc)

            # Perform test-time adaptation
            best_error = 0
            self.model.train()
            _, self.model, _, _ = com(self.model,
                        tr_loader,
                        te_loader,
                        args.criterion,
                        device,
                        lr=args.lr)

            # Compute after adaptation performance
            self.model.eval()
            after_loss, after_acc, _ = test(self.model, te_loader, device)
            tta_error[domain] = (1 - after_acc) * 100
            json_gen.collect_corruption_data(domain, "adapted_acc", after_acc)
            logger.info(f"Accuracy after adaptation - {after_acc}")

        json_gen.dump_json()

    def test_for_sota_env(self):
        """
        Evaluate the adapted sota.
        """
        ...

    def prepare_data_loaders_for_novel_env(self, args):
        # Get dataloaders (OOD)
        data_path = os.path.join(args.data_path, args.target_dataset.upper())
        if not os.path.exists(data_path):
            raise Exception(f"[INFO] Dataset not found at {data_path}")
        batch_size = args.tta_batchsize

        if args.source_dataset.upper().find("CIFAR") != -1:
            tta_train_loaders = prepare_modified_cifar_loader(
                data_path=data_path,
                label_distributer=ClassFilter(class_list=args.drop_classes),
                train=True,
                batch_size=batch_size,
                severity=args.severity,
                corruptions=CORRUPTIONS,
            )

            tta_test_loaders = prepare_modified_cifar_loader(
                data_path=data_path,
                label_distributer=DownSamplingDistributer(1.0),
                train=False,
                batch_size=1024,
                severity=args.severity,
                corruptions=CORRUPTIONS,
            )

        # @TODO write test case for IMAGENET dataset as well
        return tta_train_loaders, tta_test_loaders

    def test_for_novel_env(self): ...

    def collect_results_for_sota_env(self):
        """
        Collect the outcomes of the experiment.
        """
        ...

    def collect_results_for_novel_env(self): ...
