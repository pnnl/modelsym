#!/usr/bin/env python
import torch
from torch import nn
from torch.utils.data import DataLoader
from .alignment import *
from ..models import *
from ..datasets import cifar_loaders, cifar_loaders_old_school
import pathlib as pa
from typing import Callable
import argparse as ap
from ..constants import model_symmetries_dir
import numpy as np
from ffcv import Loader
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
experiment_dir = model_symmetries_dir / 'cifar10-alignment'
model_dir = experiment_dir / 'models'
model_library = model_symmetries_dir / 'model_library'



def alignment_experiment(arch, 
    feature_loader: DataLoader or Loader,
    name: str ='wreath-procrustes', 
    repeats=1,  similarity: Callable = wreath_procrustes, 
    sim_device=torch.cuda.current_device(),
    batched=False, progress:bool=False
    ):
    if arch == resnet20:
        model_stash = model_library / 'resnet'
    elif arch == resnet18:
        model_stash = model_library / 'resnet18'
    elif arch == mCNN:
        model_stash = model_library / 'mCNN'
    elif arch == cwCNN_bn_k:
        model_stash = model_library / 'cwCNN'
    model_files = list(model_stash.iterdir())
    for r in range(repeats):
        np.random.shuffle(model_files)
        m1file, m2file = model_files[:2]
        m1 = arch()
        m1.load_state_dict(torch.load(m1file))
        m2 = arch()
        m2.load_state_dict(torch.load(m2file))
        m1.to(memory_format=torch.channels_last)
        m2.to(memory_format=torch.channels_last)
        features1, features2 = feature_collector(feature_loader, m1, m2, layer_type=nn.ReLU, progress=progress, batched=batched)
        s = similarity(features1, features2, device=sim_device, progress=progress, batched=batched)
        if not (experiment_dir / name ).exists():
            (experiment_dir / name ).mkdir(parents=True)
        torch.save(s, experiment_dir / name / f'run{r}.pt')
        print(f'saved similarity matrix to {experiment_dir / name}. Full matrix\n{s}')

if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--repeats', type=int, help='repeats of experiment')
    parser.add_argument('--progress', action='store_true', help='status bars?')
    parser.add_argument('--batched', action='store_true', help='use batches?')
    parser.add_argument('--arch', type=str, help='architecture')
    parser.add_argument('--similarity', type=str, help='similarity metric')
    args = parser.parse_args()
    archs = {
        'mCNN': mCNN,
        'mCNN_k': mCNN_k,
        'cwCNN': cwCNN_bn_k,
        'cwCNN_k': cwCNN_k,
        'resnet20': resnet20,
        'resnet18': resnet18
    }
    sims = {
        'procrustes': wreath_procrustes,
        'cka': wreath_cka,
        'ortho_procrustes': ortho_procrustes,
        'ortho_cka': ortho_cka
    }
    arch = archs[args.arch]
    sim = sims[args.similarity]
    if (arch not in [cwCNN_bn_k, cwCNN_k]):
        sim_device = 'cuda' if (sim in [wreath_cka, ortho_cka]) else 'cpu'
    else:
        sim_device = 'cpu'
    _, test_loader = cifar_loaders(half=False)
    expname = f'{args.arch}-{args.similarity}'
    alignment_experiment(arch=arch, feature_loader=test_loader, 
        name=expname, repeats=args.repeats, 
        progress=args.progress, similarity=sim,
        sim_device=torch.device(sim_device), batched=args.batched)