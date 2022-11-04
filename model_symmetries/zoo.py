#!/usr/bin/env python
import torch
from .stitching import *
from .models import *
from .datasets import cifar_loaders
from .train import run_train_test_loop, test, train1model
import pathlib as pa
import argparse as ap
from tqdm.auto import tqdm
from .constants import model_symmetries_dir 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_library = model_symmetries_dir / 'model_library'
train_up_dir = model_symmetries_dir / 'train_up_models'

train_loader, test_loader = cifar_loaders()

def train_model_zoo(archs = [mCNN, resnet20, resnet18], 
    num_models=100, num_chunks:int=None, chunk:int=None,
    base_epochs=50, lr=1e-3, 
    library=model_library, progress:bool=False):
    device = torch.device('cuda:0')
    for a in archs:
        if a == mCNN:
            name = 'mCNN'
        elif a == resnet20:
            name = 'resnet'
        elif a == resnet18:
            name = 'resnet18'
        elif a == cwCNN_bn_k:
            name = 'cwCNN'
        print(f'training {num_models} {name} model pairs')
        loop = range(num_models)
        if num_chunks != None:
            print(f'working on chunk {chunk+1} of {num_chunks}')
            start = chunk*(num_models//num_chunks)
            stop = ((chunk+1)*(num_models//num_chunks) if chunk < num_chunks-1 else None)
            loop = loop[start:stop]
        if progress: 
            loop = tqdm(loop)
        for i in loop:
            n, m = train1model(train_loader=train_loader, test_loader=test_loader,
                arch=a, name=f'{name}base', 
                experiment_dir=train_up_dir / f'{name}_rr_{i}', 
                model_dir=train_up_dir /f'{name}_rr_{i}'/'models',
                epochs=base_epochs, lr=lr, progress=progress, device=device,
            )
            if (library / name ).exists() == False:
                (library / name ).mkdir(parents=True)
            torch.save(m.state_dict(), library / name /f'{i}.pt')

if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--num_models', type=int, default=1, help='models per arch')
    parser.add_argument('--arch', type=str, default=None, help='which arch to train')
    parser.add_argument('--progress', action='store_true', help='bars?')
    parser.add_argument('--num_chunks', type=int, default=None, help='number of zoo chunks')
    parser.add_argument('--chunk', type=int, default=None, help='working chunk')
    args = parser.parse_args()
    arch_dict = {
        'mCNN': mCNN,
        'resnet20': resnet20,
        'resnet18': resnet18,
        'cwCNN': cwCNN_bn_k,
    }
    if args.arch != None:
        archs=[arch_dict[args.arch]]
    else:
        archs = [v for k, v in arch_dict.items()]
    if args.num_chunks != None:
        train_model_zoo(archs=archs, num_models=args.num_models, progress=args.progress,
            num_chunks=args.num_chunks, chunk=args.chunk)
    else:
        train_model_zoo(archs=archs, num_models=args.num_models, progress=args.progress)