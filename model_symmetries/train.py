import pathlib as pa
from copy import deepcopy
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import StepLR
from torchinfo import summary
import numpy as np
from tqdm.auto import tqdm

# train loop
def train(model, device, train_loader, 
    optimizer: optim.Optimizer or list[optim.Optimizer] = None, 
    scaler: GradScaler = None, 
    epoch:int = 1, criterion:nn.Module = nn.CrossEntropyLoss(), 
    progress=True,  project=False, regularize=False,
    alternating = False, trainable_module: nn.Module = None, bn_epoch=False):
    if alternating and (isinstance(optimizer, list) == False):
        raise RuntimeError(f'need multiple optimizers to alternate')
    model = model.to(memory_format=torch.channels_last).cuda()
    model.train()
    losses = []
    epoch_fracs = []
    train_loop = enumerate(train_loader)
    if progress:
        train_loop = tqdm(train_loop, total=len(train_loader), desc='loss:')
    if bn_epoch:
        trainable_module.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in train_loop:
                data, target = data.to(device), target.to(device)
                with autocast():
                    output = model(data)
                    loss = criterion(output, target)
                    loss_log = f'criter: {loss.item():.4g} '
                if progress:
                    train_loop.set_description(loss_log)
                if batch_idx % 10 == 0:
                    losses.append(loss.item())
                    epoch_frac = (epoch - 1) + batch_idx / len(train_loader)
                    epoch_fracs.append(epoch_frac)
    else:
        for batch_idx, (data, target) in train_loop:
            data, target = data.to(device), target.to(device)
            if alternating:
                for o in optimizer:
                    o.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
            with autocast():
                output = model(data)
                loss = criterion(output, target)
                loss_log = f'criter: {loss.item():.4g} '
                if regularize:
                    reg = model.regularization()
                    loss += reg
                    reg_item = reg.item() if isinstance(reg, torch.Tensor) else reg 
                    loss_log += f'l2 saturation: {-reg_item/model.alpha:.4g} '
                    loss_log += f'tot: {loss.item():.4g}'
            scaler.scale(loss).backward()
            if alternating:
                oidx = np.random.randint(0, len(optimizer))
                scaler.step(optimizer[oidx])
            else:
                scaler.step(optimizer)
            scaler.update()
            if project:
                model.project(verbose=(batch_idx==0))
            if progress:
                train_loop.set_description(loss_log)
            if batch_idx % 10 == 0:
                losses.append(loss.item())
                epoch_frac = (epoch - 1) + batch_idx / len(train_loader)
                epoch_fracs.append(epoch_frac)
    return epoch_fracs, losses


# test loop
def test(model, device, test_loader, criterion:nn.Module = nn.CrossEntropyLoss(), progress=True):
    model = model.to(memory_format=torch.channels_last).cuda()
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            with autocast():
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += torch.sum(criterion(output, target)).item()  # sum up batch loss
                _, pred = output.max(dim=-1)  # get the index of the max log-probability
                test_acc += (1.0*(pred == target)).mean().item()
    test_loss /= len(test_loader)
    test_acc /= len(test_loader)
    if progress:
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}\n')
    return  test_loss, 100. * test_acc


def run_train_test_loop(model: nn.Module, train_loader, test_loader, model_name: str, 
    epochs: int = 20, device: torch.device = torch.device("cpu"), progress=True,
    model_dir = './data/experiment/models', output_dir='./data/experiment', 
    save_every_epoch=True, trainable_module: nn.Module = None,
    project=False, regularize=False, pgd_hook=False,
    decay=1e-4, lr:float=1.0, save_weights=False, strict_state=True, 
    alternating=False, full_model_train_mode=False, sgd_anyway=False, 
    schedule=True, bn_epoch=False, return_val_best:bool = False):
    if trainable_module == None:
        trainable_module = model 
    if alternating:
        perms_opt =  optim.SGD([
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight_perm', x[0])!= None], 'weight_decay': 0.0}
        ], lr=lr, 
        )
        scalars_opt = optim.SGD([
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight_scalars', x[0])!= None], 'weight_decay': decay}
        ], lr=lr, 
        )
        optimizer = [perms_opt, scalars_opt]
        scheduler = [StepLR(o, step_size=max(1, int(epochs/4)), gamma=0.5) for o in optimizer]
    elif pgd_hook:
        optimizer = optim.SGD([
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight_scalars', x[0])!= None], 'weight_decay': decay},
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight_scalars', x[0]) == None], 'weight_decay': 0.0}
        ], lr=lr, 
        )
        scheduler = StepLR(optimizer, step_size=max(1, int(epochs/4)), gamma=0.5)
    elif project:
        optimizer = optim.SGD([
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight_scalars', x[0])!= None], 'weight_decay': decay},
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight_scalars', x[0]) == None], 'weight_decay': 0.0}
        ], lr=lr, 
        )
        scheduler = StepLR(optimizer, step_size=max(1, int(epochs/4)), gamma=0.5)
    elif sgd_anyway:
        print(f'using SGD anyway!')
        optimizer = optim.SGD([
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight', x[0])!= None], 'weight_decay': decay},
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight', x[0]) == None], 'weight_decay': 0.0}
        ], lr=lr, 
        )
        scheduler = StepLR(optimizer, step_size=max(1, int(epochs/4)), gamma=0.5)
    else:
        optimizer = optim.Adam([
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight', x[0])!= None], 'weight_decay': decay},
            {'params': [x[1] for x in trainable_module.named_parameters() if re.search('weight', x[0]) == None], 'weight_decay': 0.0}
        ], lr=lr)
        scheduler = StepLR(optimizer, step_size=max(1, int(epochs/4)), gamma=0.5)   
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()
    output_dir = pa.Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
    if not (output_dir/model_name).exists():
        (output_dir/model_name).mkdir(parents=True)
    model_dir = pa.Path(model_dir)
    if not model_dir.exists():
        model_dir.mkdir(parents=True)
    with open(output_dir / model_name / 'log.txt', 'w') as f:
            # Low-tech way of printing a summary of training parameters for
            # reproducibility purposes
            message = f'run of run_train_test_loop with arguments:\n{locals()}\n'
            f.write(message)
    val_best = 0.0
    best_weights = None
    val_accs = []
    val_losses = []
    train_losses = []
    train_iters = []
    val_epochs = []
    def save_stuff():
        metrics = {
            'val_acc': val_accs,
            'val_loss': val_losses,
            'train_loss': train_losses,
            'train_iter': train_iters,
            'val_epoch': val_epochs,
            'val_best': val_best
        }
        # could also pickle the above, or convert to a pandas df and save as csv
        # ...
        for k, m in metrics.items():
            m = np.array(m)
            np.save(output_dir / model_name /  k, m)
    epoch_loop = range(1, epochs + 1)
    if progress:
        epoch_loop = tqdm(epoch_loop)
    if bn_epoch:
        sd = deepcopy(model.state_dict())
    for epoch in epoch_loop:
        epoch_fracs, train_loss = train(model=model, device=device, train_loader=train_loader, 
            optimizer=optimizer, scaler=scaler, 
            criterion=criterion, epoch=epoch,  progress=progress, project=project, 
            regularize=regularize, alternating=alternating, trainable_module=trainable_module)
        if bn_epoch:
            print(f'updating batch norm running stats\n')
            sd = deepcopy(model.state_dict())
            epoch_fracs, train_loss = train(model=model, device=device, train_loader=train_loader, 
                optimizer=optimizer, scaler=scaler, 
                criterion=criterion, epoch=epoch,  progress=progress, project=project, 
                regularize=regularize, alternating=alternating, 
                bn_epoch = bn_epoch, trainable_module=trainable_module)
            val_loss, val_acc = test(model, device, test_loader, criterion, progress=progress)
            model.load_state_dict(sd)
        else:
            val_loss, val_acc = test(model, device, test_loader, criterion, progress=progress)
        if schedule:
            if alternating:
                for s in scheduler:
                    s.step()
            else:
                scheduler.step()
        train_iters.extend(epoch_fracs)
        train_losses.extend(train_loss)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        val_epochs.append(epoch)
        if val_acc > val_best:
            val_best = val_acc
            best_weights = deepcopy(model.state_dict())
            if save_weights:
                torch.save(best_weights, model_dir / f"{model_name}.pt")
            if progress:
                print(f'weights saved to {model_dir}')
        if save_every_epoch:
            save_stuff()
            if progress:
                print(f'saved metrics to {output_dir / model_name}') 
    # ensure we end up w/ model w/ best (not final) weights
    model.load_state_dict(best_weights, strict=strict_state)
    save_stuff()
    print(f"{model_name} best acc: {val_best}")
    if return_val_best:
        return val_best
        
def train1model(train_loader, test_loader,
    arch, name, experiment_dir, 
    model_dir, progress=True, epochs=50, lr=0.1,
    device=torch.cuda.current_device()
):
    m = arch().to(device)
    run_train_test_loop(
        m,
        train_loader,
        test_loader,
        f"{name}",
        epochs=epochs,
        device=device,
        progress=progress,
        output_dir=experiment_dir,
        model_dir=model_dir,
        lr=lr,
    )
    return (f"{name}", m)

def train2models(train_loader, test_loader,
    arch, name, experiment_dir, 
    model_dir, retrain=True, progress=True, epochs=50, lr=0.1,
    device=torch.cuda.current_device()
):
    m1 = arch().to(device)
    m2 = arch().to(device)
    print(f"unstitched arch:\n{summary(m1, input_size=(1, 3, 32, 32))}")
    for i, m in enumerate((m1, m2)):
        if retrain:
            run_train_test_loop(
                m,
                train_loader,
                test_loader,
                f"{name}_{i}",
                epochs=epochs,
                device=device,
                progress=progress,
                output_dir=experiment_dir,
                model_dir=model_dir,
                lr=lr,
            )
        else:
            state_dict = torch.load(
                model_dir / f"{name}_{i}.pt",
            )  # map_location=torch.device('cpu'))
            m.load_state_dict(state_dict)
    return ((f"{name}_{i}", m) for i, m in enumerate((m1, m2)))