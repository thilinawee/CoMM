import argparse
import pandas as pd
from utils import *
from methods import com

import torch

from logger.logger import TTALogger
from tta_driver import TTADriver
from tta_config import TTAConfig

logger = TTALogger(__file__)

print(
    f"[INFO] Is CUDA available: {torch.cuda.is_available()} \n[INFO] Number of GPU detected: {torch.cuda.device_count()}")


CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
    'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
    'brightness', 'contrast', 'elastic_transform', 'pixelate',
    'jpeg_compression'
]


def main(args):
    # Seet seed for reproducability
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load model
    if args.source_dataset.upper() == "CIFAR-10":
        num_classes = 10
        model = WideResNet(depth=40, num_classes=num_classes, widen_factor=2, bias_last=True).to(device)
        state_dict = torch.load(args.model_path, map_location = device)
        _ = model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Model loaded from {args.model_path}, {_}")

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

    print("[INFO] Dataloaders ready")

    # Test-time training
    print(f"[INFO] Starting TTA")
    # TTA: train
    tta_error = dict()
    for domain in tta_train_loaders.keys():
        _ = model.load_state_dict(state_dict, strict=True)
        print(f"[INFO] Resetting model to original state. {_}")

        tr_loader = tta_train_loaders[domain][str(args.severity)]
        te_loader = tta_test_loaders[domain][str(args.severity)]

        print(f"----------  Corruption: {domain}, Severity: {args.severity}  ------------")

        if args.eval_before:
            # Compute before adaptation performance
            model.eval()
            before_loss, before_acc, before_cos_acc = test(model, te_loader, device)

        # Perform test-time adaptation
        best_error = 0
        model.train()
        _, model, _, _ = com(model,
                    tr_loader,
                    te_loader,
                    args.criterion,
                    device,
                    lr=args.lr)

        # Compute after adaptation performance
        model.eval()
        after_loss, after_acc, _ = test(model, te_loader, device)
        tta_error[domain] = (1 - after_acc) * 100

        if args.eval_before:
            # Print results
            print(f"Before Adaptation: Error: {1 - before_acc:.3%},  Cosine Error: {1 - before_cos_acc:.3%} \n"
                  f"After Adaptation: Error: {1 - after_acc:.3%}, Best Error: {best_error:.3%}")
        else:
            print(
                f"After Adaptation: Error: {1 - after_acc:.3%}, Best Error: {best_error:.3%}")
        print(f"------------------------------------------------------------------")
        print(" ")

    final_error = 0
    for domain in tta_error.keys():
        final_error += tta_error[domain]
    final_error /= len(tta_error.keys())
    print(f"[INFO] Final Accuracy: {final_error:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CoMM Test-time Adaptation")
    parser.add_argument('--model_path', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, default="/home/wei/data2/Dataset/cifar/", help='Path to data')
    parser.add_argument('--eval_before', type=bool, default=False, help='Evaluate before adaptation')
    parser.add_argument('--source_dataset', type=str, default="cifar-10", help='Source dataset')
    parser.add_argument('--target_dataset', type=str, default="cifar-10-c", help='Target dataset')
    parser.add_argument('--criterion', type=str, default="cosine", help='Loss function to use')
    parser.add_argument('--network', default="wrn-40x2", type=str, help='Network architecture')
    parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')
    parser.add_argument('--tta_batchsize', default=128, type=int, help='Batch size for test-time training')
    parser.add_argument('--severity', default=5, type=int, help='Severity of corruption')
    parser.add_argument('--verbose', action='store_true', default=False, help='Verbose')
    parser.add_argument('--seed', default=123, type=int, help='Random seed')
    parser.add_argument('--drop_classes', nargs="+", type=int, help='list of classes going to drop')
    parser.add_argument('--drop_ratio', default=0.1, type=float, help='Ratio of classes to drop')
    parser.add_argument('--report_dir', type=str, default="reports", help='Directory to save reports')
    parser.add_argument('--gpu_id', type=str, default="0", help='GPU ID')
    arguements = parser.parse_args()

    if arguements.model_path == "None":
        arguements.model_path = None

    config = TTAConfig()
    config.set_args(arguements)
    args = config.get_args()
    config.print_args()

    
    driver = TTADriver()
    driver.set_gpu_id(args.gpu_id)

    driver.init_random_seeds(args.seed)
    driver._create_report_dirs(args)
    driver.get_model(args)

    # SOTA experiment
    sota_train_loaders, sota_test_loaders = driver.prepare_data_loaders_for_sota_env(args)
    driver.apply_tta(args, 'sota', sota_train_loaders, sota_test_loaders)

    # Novel Experiment
    novel_train_loaders, novel_test_loaders = driver.prepare_data_loaders_for_novel_env(args)
    driver.apply_tta(args, 'novel', novel_train_loaders, sota_test_loaders)
    # main(args)
