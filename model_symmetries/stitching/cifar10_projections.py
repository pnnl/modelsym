#!/usr/bin/env python
import pathlib as pa
import argparse as ap
import torch
from ..datasets import cifar_loaders
from torchinfo import summary
from .projections import *
from ..train import run_train_test_loop, test
from ..models import mCNN_k, mCNN_k_sigmoid
from ..constants import model_symmetries_dir

experiment_dir = model_symmetries_dir /  'cifar10-projections'
model_dir = experiment_dir / 'models'
train_loader, test_loader = cifar_loaders()
device = torch.device('cuda:0')

def train_baseline(arch = mCNN_k, retrain=False, name="cifar10baseline", progress=True):
    model = arch().to(device)
    summary(model, input_size=(1, 3, 32, 32))
    if retrain:
        run_train_test_loop(model, train_loader, test_loader, name, 
        epochs=50, lr=1e-3, device=device,  progress=progress,
        output_dir=experiment_dir, model_dir=model_dir)    
    else:
        model = arch().to(device).eval()
        state_dict = torch.load(model_dir / f"{name}.pt",) # map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    return model, name 

def rotation_experiment(model, name="cifar10baseline", matrix_group='orthogonal', repeats=1, progress=True):
    convs = [l for l, m in enumerate(model) if isinstance(m, torch.nn.Conv2d)]
    for l in convs[:-1]:
        for r in range(repeats):
            net = rotate_layer(model=model, layer=l, matrix_group=matrix_group)
            summary(net, input_size=(1, 3, 32, 32))
            test(net, device, test_loader, torch.nn.CrossEntropyLoss())
            run_train_test_loop(
                net, 
                train_loader, 
                test_loader, 
                f"{name}_{matrix_group}_layer_{l}_run_{r}", 
                epochs=50, 
                lr=1e-3,
                device=device,
                output_dir=experiment_dir,
                model_dir= model_dir,
                progress=progress
            )    

if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--repeats', type=int, help='repeats per transformation, layer pair')
    parser.add_argument('--progress', action='store_true', help='status bars?')
    args = parser.parse_args()
    archs = {'cifar10baseline': mCNN_k, "cifar10baseline_sigmoid": mCNN_k_sigmoid}
    matrix_groups = ['orthogonal', 'Grelu']
    for k, a in archs.items():
        model, name = train_baseline(arch=a, name=k, retrain=True, progress=args.progress)
        for mg in matrix_groups:
            rotation_experiment(model, name=name, matrix_group=mg, 
            repeats=args.repeats, progress=args.progress)

