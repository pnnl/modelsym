import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
import seaborn as sns
import pathlib as pa
import re
from .models import *  
from typing import Callable
sns.set_theme(context='paper')
from .constants import model_symmetries_dir

align_dir = model_symmetries_dir / 'cifar10-alignment'
stitch_dir = model_symmetries_dir / 'cifar10-stitching'
crossval_dir = model_symmetries_dir / 'cifar10-stitching-crossval'
rotation_dir = model_symmetries_dir / 'cifar10-projections'
model_library = model_symmetries_dir / 'model_library'
tum_dir = model_symmetries_dir / 'train_up_models'


def alignment_map(folder: pa.Path or list[pa.Path], arch: Callable =mCNN, 
    title: str = None, labels: list[str] = None, reverse=False, diagonly=False, asterisks:list[int]=None):
    if not isinstance(folder, list):
        folder = [folder]
    ss = []
    s_stds = []
    if labels == None:
        labels = []
    for f in folder:
        sims = [torch.load(g) for g in f.iterdir() 
            if re.search('pt', g.suffix) != None]
        sims = torch.stack(sims, dim=0)
        s = sims.mean(dim=0,keepdim=False)
        s_std = sims.var(dim=0, keepdim=False).sqrt()
        if reverse:
            s = 1-s
        ss += [s]
        s_stds += [s_std]
        if labels == None:
            labels.append(f.parts[-1])
    m = arch()
    if arch == mCNN:
        wts = torch.load(model_library/'mCNN'/f'0.pt')
    elif arch ==cwCNN_bn_k:
        wts = torch.load(model_library/'cwCNN'/f'0.pt')
    elif arch == resnet20:
        wts = torch.load(model_library/'resnet'/f'0.pt')
    elif arch == resnet18:
        wts = torch.load(model_library/'resnet18'/f'0.pt')
    m.load_state_dict(wts)
    layernames = [i for (i, (n, m)) in enumerate(m.named_modules()) if isinstance(m, nn.ReLU)]
    if asterisks:
        layernames = [str(i) + ('*' if i in asterisks else '') for i in layernames]
    if not diagonly:
        scale_figsize = 2 if arch in [resnet20, resnet18] else 1
        fig, ax = plt.subplots(1,1, figsize=(scale_figsize*4,scale_figsize*2))
        sns.heatmap(s.numpy(), ax=ax, xticklabels=layernames, 
            yticklabels=layernames, annot=(arch==mCNN or arch==cwCNN_bn_k))
        ax.set_title(title)
        ax.set_xlabel('model 2 layer')
        ax.set_ylabel('model 1 layer')
        plt.yticks(rotation=0)
    else:
        fig, ax = plt.subplots(1,1, figsize=(8,3))
        for i in range(len(ss)):
            ax.bar(x = np.arange(ss[i].shape[0]) -0.25 + i/(2*len(ss)),  height = np.diag(ss[i]), yerr=np.diag(s_stds[i]), width=1/(2*len(ss)), label=labels[i])
        ax.set_xlabel('layer')
        ax.set_ylabel('similarity')
        ax.set_xticks(np.arange(s.shape[0]), layernames)
        ax.legend(bbox_to_anchor=(1,1))
        ax.set_title(title)

    if diagonly:
        predfs = [{f'layer {l}': [f'{s[i, i].numpy():.4f} +- {s_std[i, i].numpy():.3f}'] for i, l in enumerate(layernames)} for (s, s_std) in zip(ss, s_stds)]
        dfs = [pd.DataFrame(p, index = [labels[i]]) for (i,p) in enumerate(predfs)]
        df = pd.concat(dfs)
    else:
        predf = {f'layer {l}': [ f'{x:.4f} +- {y:.3f}' for x, y in zip(s[:, i].numpy(), s_std[:,i].numpy()) ] for i, l in enumerate(layernames)}
        df = pd.DataFrame(predf, index=[f'layer {l}' for l in layernames])
    return df, fig, ax

def compute_stitch_penalties(basename:str, stitch_dir: pa.Path, 
    run_limit=None, layerlist: list[int] = None, cross_val:bool=False, return_alphas:bool=False):
    retrain_dirs = [d for d in stitch_dir.iterdir() if re.search(basename, d.parts[-1])!= None]
    clean_accs = dict()
    stitch_dict = dict()
    for d in retrain_dirs:
        # we must keep track of the corresponding clean accs. 
        # below, s is the pieces of the retrain dir name, and int(s[-1]) is
        # the retrain index. So clean_accs has keys the retrain indices and
        # vals the clean accs.
        clean_acc = np.load(d / 'clean_acc.npy')
        s = d.parts[-1].split('_')
        rr = int(s[-1])
        clean_accs[rr] = clean_acc
        if cross_val:
            # in this case, stitch_dict has keys (l, rr) and vals (test_acc, best_alpha)
            test_acc_files = [e for e in d.iterdir() if (e.is_file() and re.search('test_acc', e.parts[-1]) != None)]
            best_alpha_files = [e for e in d.iterdir() if (e.is_file() and re.search('best_alpha', e.parts[-1]) != None)]
            for e in test_acc_files:
                t = e.parts[-1].split('_')
                l = int(t[-3])
                stitch_dict[(l, rr)] = [np.load(e).sum()]
            for e in best_alpha_files:
                t = e.parts[-1].split('_')
                l = int(t[-3])
                stitch_dict[(l, rr)].append(np.load(e).sum())
        else:
            stitch_dirs = [e for e in d.iterdir() if re.search('layer', e.parts[-1]) != None]
            for e in stitch_dirs:
                # so  t is the "stitch run dir"
                t = e.parts[-1].split('_')
                l = int(t[-3])
                sr = int(t[-1])
                warm = 1 if (re.search('warmup', e.parts[-1])!= None) else 0
                # so the dict key is of form (warmup: bool, stitch layer,
                # retrain run, stitching run) and its value is the corresponding
                # directory of training output
                stitch_dict[(warm, l, rr, sr)] = e
    if cross_val:
        stitch_accs = {k: v[0] for k, v in stitch_dict.items()}
        layers = sorted(list({l for (l, rr) in stitch_dict.keys()}))
        retrain_runs = sorted(list({rr for (l, rr) in stitch_dict.keys()}))
    else:
        stitch_accs = {k: np.load(v/'val_acc.npy').max() for k, v in stitch_dict.items()}
        layers = sorted(list({l for (w, l, rr, sr) in stitch_dict.keys()}))
        retrain_runs = sorted(list({rr for (w, l, rr, sr) in stitch_dict.keys()}))
        stitching_runs = sorted(list({sr for (w, l, rr, sr) in stitch_dict.keys()}))
    if (run_limit != None):
        if  (run_limit[i] != None):
            retrain_runs = retrain_runs[:run_limit[i]+1]
    clean_acc_arr = np.array([clean_accs[rr] for rr in retrain_runs])
    print('mean clean accuracy: ', clean_acc_arr.mean(), ' +- ', np.sqrt(clean_acc_arr.var()))
    if cross_val:
        stitch_mean_stats = {l: [stitch_accs[(l, rr)] for rr in retrain_runs] for l in layers}
        stitch_mean_stats = {l: np.array(v) for l, v in stitch_mean_stats.items()}
        # subtract from the clean acc.
        stitch_mean_stats = {l: clean_acc_arr - v for l, v in stitch_mean_stats.items()}
        stitch_mean_stats = {l: (v.mean(), np.sqrt(v.var())) for l, v in stitch_mean_stats.items()}
    else:
        # now build a dict with layers as keys and value a (retrain runs,
        # stitching runs) array of corresponding max val accs..
        if any([w == 1 for (w, l, rr, sr) in stitch_dict.keys()]):
            stitch_mean_stats = {l: [[max(stitch_accs[(0, l, rr, sr)], stitch_accs[(1, l, rr, sr)]) for sr in stitching_runs] for rr in retrain_runs] for l in layers}
            stitch_mean_stats = {l: np.array(v) for l, v in stitch_mean_stats.items()}
        else:
            stitch_mean_stats = {l: np.array([[stitch_accs[(0, l, rr, sr)] for sr in stitching_runs] for rr in retrain_runs]) for l in layers}
        # take max over stitching runs. 
        stitch_mean_stats = {l: np.max(v, axis=-1, keepdims=False) for l, v in stitch_mean_stats.items()}
        # subtract from the clean acc.
        stitch_mean_stats = {l: clean_acc_arr - v for l, v in stitch_mean_stats.items()}
        stitch_mean_stats = {l: (v.mean(), np.sqrt(v.var())) for l, v in stitch_mean_stats.items()}
    if layerlist != None:
        stitch_mean_stats = {l: v for l, v in stitch_mean_stats.items() if l in layerlist}
    if cross_val and return_alphas:
        alphas = {l: [stitch_dict[(l, rr)][-1] for rr in retrain_runs] for l in layers}
        return stitch_mean_stats, alphas
    else:
        return stitch_mean_stats


def stitch_penalties(basename: str or list[str], arch: Callable = mCNN, title: str = None, 
    label: str or list[str] = None, run_limit=None, layerlist: list[int] = None, asterisks:list[int] = None,
    cross_val:list[bool]=None, debug_mode:bool=False, redirect_anyway:bool=False):
    if not isinstance(basename, list):
        basename = [basename]
    if cross_val == None:
        cross_val = [False for b in basename]
    penalties = []
    penalty_stds = []
    if debug_mode:
        alphas = None
    for i, b in enumerate(basename):
        if cross_val[i] and debug_mode:
            stitch_mean_stats, alphas = compute_stitch_penalties(b, 
                stitch_dir=(crossval_dir if cross_val[i] else stitch_dir),
                run_limit=run_limit, layerlist=layerlist, 
                cross_val=(False if (cross_val[i] and redirect_anyway) else cross_val[i]),return_alphas=debug_mode)

        else:
            stitch_mean_stats = compute_stitch_penalties(b, 
                stitch_dir=(crossval_dir if cross_val[i] else stitch_dir),
                run_limit=run_limit, layerlist=layerlist, 
                cross_val=cross_val[i],)
        penalties.append({k: v[0] for k, v in stitch_mean_stats.items()})
        penalty_stds.append({k: v[-1] for k, v in stitch_mean_stats.items()})
    size_multiplier = 2 if arch in [resnet20, resnet18] else 1
    fig, ax = plt.subplots(1,1, figsize=(size_multiplier*8,size_multiplier*3))
    for i, (p, ps) in enumerate(zip(penalties, penalty_stds)):
        ax.bar(x=np.arange(len(p.keys())) -0.4 + 0.8*i/(len(penalties)), height=[v for k, v in p.items()],
            yerr=[v for k, v in ps.items()], capsize=2, width=(0.8/len(penalties)), label = label[i],
            )
    if asterisks != None:
        ax.set_xticks(ticks=np.arange(len(penalties[0].keys())),labels=[str(k) + ('*' if k in asterisks else '') for k, v in penalties[0].items()])
    else:
        ax.set_xticks(ticks=np.arange(len(penalties[0].keys())),labels=[str(k) for k, v in penalties[0].items()])
    ax.set_xlabel('stitching layer')
    ax.set_ylabel('stitching penalty (% acc)')
    ax.legend(bbox_to_anchor=(1,1))
    if title == None:
        title = "stitching penalties"
    ax.set_title(title)
    if label != None:
        dfs = [pd.DataFrame({f'layer {k}':[f'{p[k]:.3g} +- {ps[k]:.2g}'] for k in p.keys()}, index = [label[i]]) for (i, (p, ps)) in enumerate(zip(penalties, penalty_stds))]
        df = pd.concat(dfs)
    else:
        df = [pd.DataFrame({f'layer {k}':[f'{p[k]:.3g} +- {ps[k]:.2g}'] for k in p.keys()}) for (i, (p, ps)) in enumerate(zip(penalties, penalty_stds))]
    if debug_mode:
        return df, fig, ax, alphas
    else:
        return df, fig, ax

def rotation_penalties(basename: str or list[str], arch: Callable = mCNN_k, title: str = None, 
    label: str or list[str] = None,):
    if not isinstance(basename, list):
        basename = [basename]
    penalties = []
    penalty_stds = []
    for i, b in enumerate(basename):
        bn_dirs = [d for d in rotation_dir.iterdir() if re.search(b, d.parts[-1])!= None]
        baseline_dir = rotation_dir/'cifar10baseline'

        base_vc = np.load(baseline_dir/'val_acc.npy').max()
        if i == 0:
            print('base acc', base_vc)
        bn_dir_dict = dict()
        for d in bn_dirs:
            t = d.parts[-1].split('_')
            bn_dir_dict[(int(t[-3]), int(t[-1]))] = d

        bn_accs = {k: np.load(v/'val_acc.npy').max() for k, v in bn_dir_dict.items()}
        layers = sorted(list({l for (l, r) in bn_dir_dict.keys()}))
        re_runs = sorted(list({r for (l, r) in bn_dir_dict.keys()}))
        bn_stats = {l: np.array([bn_accs[(l,r)] for r in re_runs] ) for l in layers}
        bn_stats = {l: (v.mean(), np.sqrt(v.var())) for l, v in bn_stats.items()}
        penalties.append({k: base_vc -v[0] for k, v in bn_stats.items()})
        penalty_stds.append({k: np.sqrt(v[1]**2) for k, v in bn_stats.items()})


    fig, ax = plt.subplots(1,1, figsize=(8,3))
    for i, (p, ps) in enumerate(zip(penalties, penalty_stds)):
        ax.bar(x=np.arange(len(p.keys())) -0.25 + i/(2*len(penalties)), height=[v for k, v in p.items()],
            yerr=[v for k, v in ps.items()], width=(1/(2*len(penalties))), capsize=2, label = label[i],
            )
    ax.set_xticks(ticks=np.arange(len(penalties[0].keys())),labels=[str(k) for k, v in penalties[0].items()])
    ax.set_xlabel('rotation layer')
    ax.set_ylabel('rotation penalty (% acc)')
    ax.legend(bbox_to_anchor=(1,1))
    if title != None:
        ax.set_title(title)
    if label != None:
        dfs = [pd.DataFrame({f'layer {k}':[f'{p[k]:.3g} +- {ps[k]:.2g}'] for k in p.keys()}, index = [label[i]]) for (i, (p, ps)) in enumerate(zip(penalties, penalty_stds))]
        df = pd.concat(dfs)
    else:
        df = [pd.DataFrame({f'layer {k}':[f'{p[k]:.3g} +- {ps[k]:.2g}'] for k in p.keys()}) for (i, (p, ps)) in enumerate(zip(penalties, penalty_stds))]
    return df, fig, ax

def lasso_penalties(basename: str or list[str], arch: Callable = mCNN, title: str = None, 
    label: str or list[str] = None, run_limit=None, layerlist: list[int] = None, 
    asterisks:list[int] = None, lambdas:list[float]=None):
    mCNN_widths = [64, 128, 256, 512]
    resnet20_widths = [16] + [16]*6 +[32]*6 +[64]*6
    if not isinstance(basename, list):
        basename = [basename]
    if arch == mCNN:
        basename.append('mCNNbirkhoff')
    elif arch == resnet20:
        basename.append('resnet20birkhoff')
    penalties = []
    penalty_stds = []
    sparsity_list, sparsity_std_list = [], []
    layers = None
    for i, b in enumerate(basename):
        retrain_dirs = [d for d in stitch_dir.iterdir() if re.search(b, d.parts[-1])!= None]
        if (run_limit != None):
            if  (run_limit[i] != None):
                retrain_dirs = [d for d in retrain_dirs if int(d.parts[-1].split('_')[-1]) < run_limit[i]]
        clean_accs = dict()
        stitch_dir_dict = dict()
        for d in retrain_dirs:
            stitch_dirs = [e for e in d.iterdir() if re.search('layer', e.parts[-1]) != None]
            # we must keep track of the corresponding clean accs. 
            # below, s is the pieces of the retrain dir name, and int(s[-1]) is
            # the retrain index. So clean_accs has keys the retrain indices and
            # vals the clean accs.
            clean_acc = np.load(d / 'clean_acc.npy')
            s = d.parts[-1].split('_')
            clean_accs[int(s[-1])] = clean_acc
            for e in stitch_dirs:
                # so s is the "retrain run dir" and t is the "stitch run dir"
                # i.e. s is same as above
                s, t = e.parts[-2].split('_'), e.parts[-1].split('_')
                # so the dict key is of form (warmup: bool, stitch layer,
                # retrain run, stitching run) and its value is the corresponding
                # directory of training output
                stitch_dir_dict[(int(t[-3]), int(s[-1]), int(t[-1]))] = e
        stitch_accs = {k: np.load(v/'val_acc.npy').max() for k, v in stitch_dir_dict.items()}
        if i < len(basename) -1:
            sparsities = {k: np.load(v/'sparsity.npy').sum() for k, v in stitch_dir_dict.items()}
        layers = sorted(list({l for (l, rr, sr) in stitch_dir_dict.keys()}))
        retrain_runs = sorted(list({rr for (l, rr, sr) in stitch_dir_dict.keys()}))
        stitching_runs = sorted(list({sr for (l, rr, sr) in stitch_dir_dict.keys()}))
        # now build a dict with layers as keys and value a (retrain runs,
        # stitching runs) array of corresponding max val accs..
        stitch_mean_stats = {l: np.array([[stitch_accs[(l, rr, sr)] for sr in stitching_runs] for rr in retrain_runs]) for l in layers}
        if i < len(basename) -1:
            sparsity_stats = {l: np.array([[sparsities[(l, rr, sr)] for sr in stitching_runs] for rr in retrain_runs]) for l in layers}
        # take max over stitching runs. 
        sparsity_indices = {l: np.argmax(v, axis=-1) for l, v in stitch_mean_stats.items()}
        stitch_mean_stats = {l: np.max(v, axis=-1, keepdims=False) for l, v in stitch_mean_stats.items()}
        if i < len(basename) -1:
            sparsity_stats = {l: v.sum(axis=-1, keepdims=False) for l, v in sparsity_stats.items()}
        # subtract from the clean acc.
        clean_acc_arr = np.array([clean_accs[rr] for rr in retrain_runs])
        stitch_mean_stats = {l: clean_acc_arr - v for l, v in stitch_mean_stats.items()}
        stitch_mean_stats = {l: (v.mean(), np.sqrt(v.var())) for l, v in stitch_mean_stats.items()}
        if i < len(basename) -1:
            sparsity_stats = {l: (v.mean(), np.sqrt(v.var())) for l, v in sparsity_stats.items()}
        if layerlist != None:
            stitch_mean_stats = {l: v for l, v in stitch_mean_stats.items() if l in layerlist}
            if i < len(basename) -1:
                sparsity_stats = {l: v for l, v in sparsity_stats.items() if l in layerlist}
        penalties.append({k: v[0] for k, v in stitch_mean_stats.items()})
        penalty_stds.append({k: v[-1] for k, v in stitch_mean_stats.items()})
        if i < len(basename) -1:
            sparsity_list.append({k: v[0] for k, v in sparsity_stats.items()})
            sparsity_std_list.append({k: v[-1] for k, v in sparsity_stats.items()})
        else:
            if arch == mCNN:
                sparsity_list.append({k: 1- 1/j for k, j in zip(sparsity_stats.keys(), mCNN_widths)})
            elif arch == resnet20:
                sparsity_list.append({k: 1- 1/j for k, j in zip(sparsity_stats.keys(), resnet20_widths)})
    if arch in [resnet20, resnet18]:
        fig, ax = plt.subplots(4, 5, figsize=(16,12))
    else:
        fig, ax = plt.subplots(1,len(layerlist) if layerlist != None else len(layers), figsize=(16,3))
    loop = (layerlist if layerlist != None else layers)
    return sparsity_list, penalties
    for i, l in enumerate(loop):
        x = [s[l] for s in sparsity_list]
        y = [p[l] for p in penalties]
        c = lambdas 

        a = ax[i//5, i%5] if arch in [resnet20, resnet18] else ax[i]
        cax = a.scatter(x=x, y=y, 
            # c=c, cmap='viridis', norm=colors.LogNorm(),
            )
        for j, m in enumerate(lambdas):
            a.annotate(r'$\lambda$ = '+f'{m}', (x[j]+0.01, y[j]+0.01))
        a.annotate(r'$G_{\mathrm{ReLU}}$', (x[-1]+0.01, y[-1]+0.01))
        # a.plot(x, y,)
        a.set_xlabel('sparsity')
        a.set_ylabel('stitching penalty')
        a.set_title(f'layer {l}')
        # fig.colorbar(cax, ax=a)
    if title != None:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, ax
    
    