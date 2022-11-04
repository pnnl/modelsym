#!/usr/bin/env python
import pathlib as pa
from typing import List
import torch
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np
import torchvision
from torchvision import datasets, transforms
from ffcv.fields import IntField, RGBImageField
from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
from ffcv.loader import Loader, OrderOption
from ffcv.pipeline.operation import Operation
from ffcv.transforms import (Convert, Cutout, RandomHorizontalFlip,
    RandomTranslate, ToDevice, ToTensor, ToTorchImage, RandomResizedCrop, NormalizeImage)
from ffcv.transforms.common import Squeeze
from ffcv.writer import DatasetWriter
from .constants import cifar_dir

def cifar_loaders(rewrite:bool = False, cutout:bool=False, half:bool = True):
    aug_suffix = ("cutout" if cutout else "stock")
    betons = {
        k: "cifar_" + k + "_" + aug_suffix +".beton" for k in ['train', 'test']
    }
    if rewrite:
        datasets = {
            "train": torchvision.datasets.CIFAR10(cifar_dir, train=True, download=True),
            "test": torchvision.datasets.CIFAR10(cifar_dir, train=False, download=True),
        }
        for (name, ds) in datasets.items():
            print(f'generating beton ', betons[name])
            writer = DatasetWriter(
                cifar_dir / betons[name], 
                {"image": RGBImageField(), "label": IntField()}
            )
            writer.from_indexed_dataset(ds)

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    BATCH_SIZE = 32

    loaders = {}
    for name in ["train", "test"]:
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice("cuda:0"),
            Squeeze(),
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name == "train":
            if cutout:
                image_pipeline.extend(
                    [
                        RandomHorizontalFlip(),
                        RandomTranslate(padding=2, fill=CIFAR_MEAN),
                        Cutout(
                            8, tuple(map(int, CIFAR_MEAN))
                        ),  # Note Cutout is done before normalization.
                    ]
                )
            else: 
                image_pipeline.extend(
                    [ 
                        RandomHorizontalFlip(),
                        RandomTranslate(padding=2, fill=CIFAR_MEAN)
                    ]
                )
        image_pipeline.extend(
            [
                ToTensor(),
                ToDevice("cuda:0", non_blocking=True),
                ToTorchImage(),
            ]
        )
        if half:
            image_pipeline.extend(
                [NormalizeImage(np.array(CIFAR_MEAN), np.array(CIFAR_STD), type=np.float16)]
            )
        else:
            image_pipeline.extend(
                [NormalizeImage(np.array(CIFAR_MEAN), np.array(CIFAR_STD), type=np.float32)]
            )

        # Create loaders
        loaders[name] = Loader(
            cifar_dir / betons[name],
            batch_size=BATCH_SIZE,
            num_workers=8,
            order=OrderOption.RANDOM,
            drop_last=(name == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

    train_loader, test_loader = loaders["train"], loaders["test"]
    return train_loader, test_loader

def cifar_loaders_crossval(rewrite:bool = False, cutout:bool=False, half:bool = True):
    aug_suffix = ("cutout" if cutout else "stock")
    betons = {
        k: "cifar_crossval" + k + "_" + aug_suffix +".beton" for k in ['train', 'val', 'test']
    }
    if rewrite:
        tds = torchvision.datasets.CIFAR10(cifar_dir, train=True, download=True)
        splitidxs = np.random.permutation(len(tds))[:int(0.8*len(tds))], np.random.permutation(len(tds))[int(0.8*len(tds)):]
        datasets = {
            "train": Subset(tds, splitidxs[0]),
            "val": Subset(tds, splitidxs[-1]),
            "test": torchvision.datasets.CIFAR10(cifar_dir, train=False, download=True),
        }
        for (name, ds) in datasets.items():
            print(f'generating beton ', betons[name])
            writer = DatasetWriter(
                cifar_dir / betons[name], 
                {"image": RGBImageField(), "label": IntField()}
            )
            writer.from_indexed_dataset(ds)

    CIFAR_MEAN = [125.307, 122.961, 113.8575]
    CIFAR_STD = [51.5865, 50.847, 51.255]
    BATCH_SIZE = 32

    loaders = {}
    for name in ["train", 'val',  "test"]:
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            ToDevice("cuda:0"),
            Squeeze(),
        ]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        # Add image transforms and normalization
        if name != "test":
            if cutout:
                image_pipeline.extend(
                    [
                        RandomHorizontalFlip(),
                        RandomTranslate(padding=2, fill=CIFAR_MEAN),
                        Cutout(
                            8, tuple(map(int, CIFAR_MEAN))
                        ),  # Note Cutout is done before normalization.
                    ]
                )
            else: 
                image_pipeline.extend(
                    [ 
                        RandomHorizontalFlip(),
                        RandomTranslate(padding=2, fill=CIFAR_MEAN)
                    ]
                )
        image_pipeline.extend(
            [
                ToTensor(),
                ToDevice("cuda:0", non_blocking=True),
                ToTorchImage(),
            ]
        )
        if half:
            image_pipeline.extend(
                [NormalizeImage(np.array(CIFAR_MEAN), np.array(CIFAR_STD), type=np.float16)]
            )
        else:
            image_pipeline.extend(
                [NormalizeImage(np.array(CIFAR_MEAN), np.array(CIFAR_STD), type=np.float32)]
            )

        # Create loaders
        loaders[name] = Loader(
            cifar_dir / betons[name],
            batch_size=BATCH_SIZE,
            num_workers=8,
            order=OrderOption.RANDOM,
            drop_last=(name == "train"),
            pipelines={"image": image_pipeline, "label": label_pipeline},
        )

    train_loader, val_loader, test_loader = (loaders[k] for k in ['train', 'val', 'test'])
    return train_loader, val_loader, test_loader

def cifar_loaders_old_school():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.8, 1), ratio=(9/10, 10/9)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    test_transform = transforms.Compose([
        # transforms.RandomCrop(32, 4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset1 = datasets.CIFAR10(cifar_dir, train=True, download=True,
                    transform=train_transform)
    dataset2 = datasets.CIFAR10(cifar_dir, train=False,
                    transform=test_transform)

    train_loader = DataLoader(dataset1, num_workers = 4, pin_memory = True, batch_size=32, shuffle = True)
    test_loader = DataLoader(dataset2,  num_workers = 4, pin_memory = True, batch_size=32)
    return train_loader, test_loader
