#!/usr/bin/env python
from typing import List
import torch
from torch import nn
from .stitching import *
from ..models import *
from ..datasets import cifar_loaders, cifar_loaders_old_school
from ..alignment.alignment import feature_collector
from ..train import run_train_test_loop, test
import pathlib as pa
import re
from torchinfo import summary
from torch.utils.data import Subset, DataLoader
import argparse as ap
import numpy as np
from tqdm.auto import tqdm
from ..constants import model_symmetries_dir 

experiment_dir = model_symmetries_dir /  'cifar10-stitching'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = experiment_dir / "models"
model_library = model_symmetries_dir / 'model_library'
train_up_dir = model_symmetries_dir / 'train_up_models'

train_loader, test_loader = cifar_loaders()
train_loader_oldschool, _ = cifar_loaders_old_school()

def stitching_experiment(
    arch=resnet20,
    layers: List[int] = None,
    name="cifar10stitchingResnet",
    stitch_repeats=1,
    retrain_repeats=1,
    progress=True,
    stitch_epochs=50,
    wreath=False,
    birkhoff=False,
    lowrank=False,
    lasso=False,
    pgd_hook=False,
    alpha:float = 1e-3,
    multiplier: float = 0.1,
    rank:int = 1,
    permwarmup=True,
    alternating=False,
    stitch_lr=None,
    schedule=True, 
    bn_epoch = False,
    sr_offset:int = 0,
):
    feature_ds = Subset(train_loader_oldschool.dataset, [0])
    feature_dl = DataLoader(feature_ds, batch_size=1)
    if layers == None:
        m = arch()
        nm = list(enumerate(m.named_modules()))
        layers = [i for i, (n, m) in nm if isinstance(m, nn.ReLU)]
        layers.sort(reverse=True)
        print(
            "stitching layers:",
            [(n, type(m)) for i, (n, m) in nm if isinstance(m, nn.ReLU)],
        )
        del m

    if stitch_lr == None:
        if wreath:
            stitch_lr = 0.01
        elif birkhoff:
            stitch_lr = 0.01
        else:
            stitch_lr = 1e-3
    if arch == resnet20:
        model_stash = model_library / 'resnet'
    elif arch == resnet18:
        model_stash = model_library / 'resnet18'
    elif arch == mCNN:
        model_stash = model_library / 'mCNN'
    model_files = list(model_stash.iterdir())
    for rr in range(retrain_repeats):
        np.random.shuffle(model_files)
        m1file, m2file = model_files[:2]
        m1 = arch()
        m1.load_state_dict(torch.load(m1file))
        m2 = arch()
        m2.load_state_dict(torch.load(m2file))
        sd1, sd2 = m1.state_dict(), m2.state_dict()
        _, m1_clean_acc = test(m1, device=device, test_loader=test_loader, progress=progress)
        _, m2_clean_acc = test(m2, device=device, test_loader=test_loader, progress=progress)
        clean_acc = (m1_clean_acc + m2_clean_acc)/2
        output_dir=experiment_dir/f'{name}_rr_{rr}'
        if not output_dir.exists():
            output_dir.mkdir(parents=True)
        np.save(output_dir / "clean_acc", np.array(clean_acc))
        for sr in range(stitch_repeats):
            for i, l in enumerate(layers):
                sm1, sm2 = arch(), arch()
                sm1.load_state_dict(sd1)
                sm2.load_state_dict(sd2)
                f1, f2 = feature_collector(
                    feature_dl, sm1, sm2, layer_type=nn.ReLU, progress=True
                )
                del f1, f2
                StitchClass = StitchedModelR20 if (arch in [resnet20, resnet18]) else StitchedModel
                sm = StitchClass(sm1, sm2, stitch_idx=l, wreath=wreath,   
                    birkhoff=birkhoff, pgd_hook=pgd_hook, alpha=alpha, scalars=False, 
                    lasso=lasso, multiplier=multiplier, lowrank=lowrank, rank=rank)
                print(f"stitched arch: \n{summary(sm, input_size=(1, 3, 32, 32))}")
                test(sm, device, test_loader, nn.CrossEntropyLoss())
                if wreath:
                    suffix = 'wreath'
                elif birkhoff:
                    suffix = 'birkhoff'
                elif lowrank:
                    suffix = 'lowrank'
                elif lasso:
                    suffix = 'lasso'
                else:
                    suffix = 'standard'
                sgd_anyway = (not (wreath or birkhoff))
                if permwarmup:
                    tnm = [
                        x
                        for x in sm.stitches.named_parameters()
                        if re.search("perm", x[0]) != None
                    ]
                    print(f"stitching parameters:\n{[x for x , y in tnm]}")
                    run_train_test_loop(
                        sm, 
                        train_loader, 
                        test_loader, 
                        f"{name}_{suffix}_warmup_layer_{l}_run_{sr_offset+sr}", 
                        epochs=int(stitch_epochs/2), 
                        device=device,
                        output_dir=output_dir,
                        model_dir= output_dir/'models',
                        progress=progress, 
                        trainable_module=sm.stitches,
                        lr=stitch_lr,
                        decay=(1e-4 if (wreath or birkhoff) else 1e-4),
                        regularize=(birkhoff or lasso),
                        project=birkhoff,
                        pgd_hook=pgd_hook,
                        strict_state=False,
                        full_model_train_mode=True,
                        sgd_anyway = sgd_anyway,
                        schedule=schedule,
                        bn_epoch=bn_epoch,
                        save_weights=False,
                    ) 
                    if birkhoff or wreath:
                        print('perms weight',
                            sm.stitches.A.weight_perms, 
                            "perm matrix",
                            sm.stitches.A.P,
                        )
                        sm.scalars()
                    tnm = sm.stitches.named_parameters()
                    print(f"stitching parameters:\n{[x for x , y in tnm]}")
                    test(sm, device, test_loader, nn.CrossEntropyLoss())
                    run_train_test_loop(
                            sm, 
                            train_loader, 
                            test_loader, 
                            f"{name}_{suffix}_layer_{l}_run_{sr_offset+sr}", 
                            epochs=int(stitch_epochs/2), 
                            device=device,
                            output_dir=output_dir,
                            model_dir=output_dir/'models',
                            progress=progress, 
                            trainable_module=sm.stitches,
                            lr=(stitch_lr/(2**4) if schedule else stitch_lr),
                            decay=(1e-4 if (wreath or birkhoff) else 1e-4),
                            regularize=(birkhoff or lasso),
                            project=birkhoff,
                            pgd_hook=pgd_hook,
                            strict_state=False,
                            alternating=alternating,
                            full_model_train_mode=True,
                            sgd_anyway = sgd_anyway,
                            schedule=schedule,
                            bn_epoch=bn_epoch,
                            save_weights=False,
                        )
                    if birkhoff or wreath:
                        print('perms weight',
                            sm.stitches.A.weight_perms, 
                            'scalars_weight',
                            sm.stitches.A.weight_scalars,
                            "perm matrix",
                            sm.stitches.A.P,
                        )
                else:
                    if birkhoff or wreath:
                        sm.scalars()
                    tnm = sm.stitches.named_parameters()
                    print(f"stitching parameters:\n{[x for x , y in tnm]}")
                    test(sm, device, test_loader, nn.CrossEntropyLoss())
                    run_train_test_loop(
                            sm, 
                            train_loader, 
                            test_loader, 
                            f"{name}_{suffix}_layer_{l}_run_{sr_offset+sr}", 
                            epochs=stitch_epochs, 
                            device=device,
                            output_dir=output_dir,
                            model_dir= output_dir/'models',
                            progress=progress, 
                            trainable_module=sm.stitches,
                            lr=stitch_lr,
                            decay=(1e-4 if (wreath or birkhoff) else 1e-4),
                            regularize=(birkhoff or lasso),
                            project=birkhoff,
                            pgd_hook=pgd_hook,
                            strict_state=False,
                            alternating=alternating,
                            full_model_train_mode=True,
                            sgd_anyway = sgd_anyway,
                            schedule=schedule,
                            bn_epoch=bn_epoch,
                            save_weights=False,
                        )
                    if birkhoff or wreath:
                        print('perms weight',
                            sm.stitches.A.weight_perms, 
                            'scalars_weight',
                            sm.stitches.A.weight_scalars,
                            "perm matrix",
                            sm.stitches.A.P,
                        )
                    if lasso:
                        # hack to save sparsity values
                        sparsity = (sm.stitches.A.P.abs()  <= 1e-3).sum().item()/sm.stitches.A.P.numel()
                        sparsity = np.array([sparsity])
                        np.save(output_dir / f"{name}_{suffix}_layer_{l}_run_{sr_offset+sr}" / 'sparsity', sparsity)
                        print('\nlasso sparsity:\n', f'{sparsity.sum():.4g}')
                    if lowrank:
                        print('\nranks\n', [(n, p.shape) for (n, p) in sm.stitches.named_parameters()])
                del sm, sm1, sm2

if __name__=='__main__':
    parser = ap.ArgumentParser()
    parser.add_argument('--retrain_repeats', type=int, help='repeats of experiment')
    parser.add_argument('--stitch_repeats', type=int, help='repeats of stitching run')
    parser.add_argument('--stitch_epochs', type=int, help='stitch epochs')
    parser.add_argument('--progress', action='store_true', help='status bars?')
    parser.add_argument('--wreath', action='store_true', help='wreath layer?')
    parser.add_argument('--birkhoff', action='store_true', help='birkhoff layer?')
    parser.add_argument('--lowrank', action='store_true', help='lowrank layer?')
    parser.add_argument('--rank', type=int, help='rank of lowrank')
    parser.add_argument('--lasso', action='store_true', help='lasso layer?')
    parser.add_argument('--multiplier', type=float, default=0.1, help='lasso reg multiplier')
    parser.add_argument('--name', type=str, help='experiment name')
    parser.add_argument('--arch', type=str, help='architecture')
    parser.add_argument('--alpha', type=float, default=1e-3, help='L2 reg multiplier')
    parser.add_argument('--nopermwarmup', action='store_false', help='skip perm warmup?')
    parser.add_argument('--noschedule', action='store_false', help='skip lr schedule?')
    parser.add_argument('--alternating', action='store_true', help='alternating updates?')
    parser.add_argument('--stitch_lr', type=float, default=None, help='stitch learning rate')
    parser.add_argument('--pgd_hook', action='store_true', help='pgd_hook?')
    parser.add_argument('--bn_epoch', action='store_true', help='bn_epoch?')
    parser.add_argument('--sr_offset', type=int, default=0, help='stitch run offset')

    args = parser.parse_args()
    archs = {
        'mCNN': mCNN,
        'mCNN_k': mCNN_k,
        'cwCNN': cwCNN_bn_k,
        'cwCNN_k': cwCNN_k,
        'resnet20': resnet20,
        'resnet18': resnet18
    }
    arch = archs[args.arch]
    stitching_experiment(arch=arch, name=args.name, 
        retrain_repeats=args.retrain_repeats, stitch_repeats=args.stitch_repeats,
        stitch_epochs=args.stitch_epochs, 
        wreath=args.wreath, birkhoff=args.birkhoff, 
        pgd_hook=args.pgd_hook, 
        alpha=args.alpha, permwarmup=args.nopermwarmup,
        progress=args.progress, alternating=args.alternating, 
        schedule=args.noschedule, bn_epoch=args.bn_epoch, sr_offset=args.sr_offset,
        lowrank=args.lowrank, rank=args.rank, lasso=args.lasso, multiplier=args.multiplier)
    