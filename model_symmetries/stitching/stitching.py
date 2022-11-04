import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from scipy import optimize as spo
from copy import deepcopy

def softsort(s: torch.Tensor, tau: float = 1.0, o: torch.Tensor = None, 
    normalized: bool = True, experimental=False):
    sorts, _ = torch.sort(s, dim=0, descending=True)
    if normalized: 
        tau /= s.numel()
    if o != None:
        o = torch.ones_like(s, device=s.device)
    ans = torch.abs(sorts @ o.t() - o @ s.t())/tau
    if experimental:
        ans = 1/(1+ans)
        rowsums = ans.sum(dim=-1, keepdim=True)
        return ans/rowsums
    else:
        return torch.softmax(-ans, dim=-1)

def sinkhorn(A:torch.Tensor, iters=16, verbose=False):
    zr, zc = None, None
    for i in range(iters):
        sum = A.sum(dim=-1, keepdim=True)
        if i == iters-1:
            zr = (sum == 0.0).sum()
        denom = (sum == 0.0)*torch.ones_like(sum) + (sum != 0.0)*sum
        A /= denom
        sum = A.sum(dim=0, keepdim=True)
        if i == iters-1:
            zc = (sum == 0.0).sum()
        denom = (sum == 0.0)*torch.ones_like(sum) + (sum != 0.0)*sum
        A /= denom
    if verbose:
        print(f'zero rows: {zr}, zero cols: {zc}')
    return A

class Wreath(nn.Module):
    def __init__(self, features:int, tau: float = 1.0, style='conv',
         bias=False, scalars=True):
        super().__init__()
        self.features = features
        self.tau = tau
        self.style = style
        self.scalars = scalars
        pre_weight_perms = 2*torch.rand((features,1)) - 1
        self.weight_perms = nn.Parameter(pre_weight_perms, 
            requires_grad=True)
        self.o = torch.ones_like(self.weight_perms,
            requires_grad=True)
        self.weight_scalars = nn.Parameter(torch.zeros((features,)), 
            requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros((features,)), 
                requires_grad=True)
        self.P = None

    def forward(self, x):
        if self.training:
            if self.P != None:
                self.P = None
            w = softsort(self.weight_perms, self.tau, self.o) 
        else:
            if self.P == None:
                softP = softsort(self.weight_perms, self.tau, self.o) 
                ridx = torch.argmax(softP, dim=-1)
                P = torch.zeros_like(softP)
                cidx = torch.argmax(softP, dim=-1)
                ridx = torch.arange(ridx.numel())
                P[ridx, cidx] = 1.0
                self.P = P.to(self.weight_perms.device)
            w = self.P
        if self.scalars:
            w = w @ torch.diag(torch.exp(self.weight_scalars))
        b = self.bias if hasattr(self, 'bias') else torch.zeros((self.features,)).to(w.device)
        if self.style == 'linear':
            return x @ w.t() + b
        elif self.style == 'conv':
            w = w.reshape(w.shape+(1,1))
            return F.conv2d(x, w, b)

class Birkhoff(nn.Module):
    def __init__(self, features:int, style='conv',
        bias=False, scalars=True, rounding='hungarian',
        sink=True, alpha:float = 1e-3, beta:float = 1e-3, 
        eps:float=1e-3, pgd_hook=False, rms=False, experimental=False):
        super().__init__()
        self.features=features
        self.style=style
        self.scalars=scalars
        self.rounding=rounding
        self.sink=sink
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.rms, self.experimental = rms, experimental
        pre_weight_perms = torch.rand((features,features))
        pre_weight_perms = sinkhorn(pre_weight_perms, iters=100)
        self.weight_perms = nn.Parameter(pre_weight_perms, requires_grad=True)
        self.weight_scalars = nn.Parameter(torch.ones((features,)), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros((features,)), 
                requires_grad=True)
        self.P = None
        self.r = nn.ReLU()
        self.o = torch.ones((features,1))
        if pgd_hook:
            self.apply_pgd_hooks()

    def project(self, verbose=False):
        self.weight_perms.data = self.r(self.weight_perms.data)
        self.weight_scalars.data = self.r(self.weight_scalars.data)
        if self.sink:
            if verbose:
                print(f'sinking weight_perms')
            self.weight_perms.data = sinkhorn(self.weight_perms.data, verbose=verbose)

    def regularization(self):
        reg = - self.alpha*torch.linalg.norm(self.weight_perms)
        if self.rms:
            reg /= torch.sqrt(1.0*torch.tensor(self.features))
        elif self.experimental:
            reg *= torch.tensor(self.features)
        if not self.sink:
            self.o = self.o.to(self.weight_perms.device)
            reg += self.beta*(
                torch.linalg.norm(self.o.t() @ self.weight_perms - self.o.t()) \
                + torch.linalg.norm(self.weight_perms @ self.o - self.o)
            )/torch.sqrt(1.0*torch.tensor(self.features))
        return reg

    def birkhoff_pgd(self, g: torch.Tensor):
        w = self.weight_perms.data
        e = self.eps
        v = sinkhorn(self.r(w - e*g))
        pgdg = (w - v)/e
        return (g.norm()/(e +pgdg.norm()))*pgdg
        
    def relu_pgd(self, g: torch.Tensor):
        w = self.weight_scalars.data
        e = self.eps
        v = self.r(w - e*g)
        pgdg =  (w - v)/e
        return (g.norm()/(e +pgdg.norm()))*pgdg

    def apply_pgd_hooks(self):
        self.weight_perms.register_hook(self.birkhoff_pgd)
        self.weight_scalars.register_hook(self.relu_pgd)
    
    def forward(self, x):
        if self.training:
            if self.P != None:
                self.P = None
            w = self.weight_perms 
        else:
            if self.P == None:
                softP = self.weight_perms
                P = torch.zeros_like(softP).to(self.weight_perms.device)
                if self.rounding == 'monotone_random':
                    mrv, _ = torch.sort(
                        torch.rand((self.features,))
                        )
                    mrv = self.features*mrv.to(self.weight_perms.device)
                    _, perm = torch.sort(softP @ mrv)
                    P[perm, torch.arange(self.features)] = 1.0
                elif self.rounding == 'hungarian':
                    print(f'running hungarian alg. on weight_perms\n')
                    I, J = spo.linear_sum_assignment(softP.cpu().numpy(),
                        maximize=True)
                    P[I, J] = 1.0
                self.P = P.to(self.weight_perms.device)
            w = self.P
        if self.scalars:
            w = w @ torch.diag(self.weight_scalars)
        b = self.bias if hasattr(self, 'bias') else torch.zeros((self.features,)).to(w.device)
        if self.style == 'linear':
            return x @ w.t() + b
        elif self.style == 'conv':
            w = w.reshape(w.shape+(1,1))
            return F.conv2d(input=x, weight=w, bias=b)

class Lasso(nn.Module):
    def __init__(self, features:int, style='conv',
        bias=False, scalars=True, multiplier:float = 0.1, 
        pgd_hook=False, threshold = 1e-4):
        super().__init__()
        self.features=features
        self.style=style
        self.scalars=scalars
        self.multiplier = multiplier
        self.threshold = threshold
        self.l = nn.Linear(features, features, bias=bias)
        self.P = None
        if pgd_hook:
            self.apply_pgd_hooks()

    def project(self, verbose=False):
        pass

    def regularization(self):
        reg = self.multiplier*(torch.abs(self.l.weight).sum())
        return reg

    def apply_pgd_hooks(self):
        pass
    
    def forward(self, x):
        if self.training:
            if self.P != None:
                self.P = None
            w = self.l.weight 
        else:
            if self.P == None:
                P = ((self.l.weight.abs() < self.threshold)*torch.zeros_like(self.l.weight) 
                + (self.l.weight.abs() >= self.threshold)*self.l.weight)
                self.P = P.to(self.l.weight.device)
            w = self.P
        b = self.l.bias if hasattr(self.l, 'bias') else torch.zeros((self.features,)).to(w.device)
        if self.style == 'linear':
            return x @ w.t() + b
        elif self.style == 'conv':
            w = w.reshape(w.shape+(1,1))
            return F.conv2d(input=x, weight=w, bias=b)

class LowRank(nn.Module):
    def __init__(self, features:int, style='conv',
        bias=False, scalars=True, rank=1,
        pgd_hook=False, ):
        super().__init__()
        self.features=features
        self.style=style
        self.scalars=scalars
        self.rank=rank
        if style == 'linear':
            self.l1 = nn.Linear(features, rank, bias=False)
            self.l2 = nn.Linear(rank, features, bias=bias)
        elif style == 'conv':
            self.l1 = nn.Conv2d(features, rank, kernel_size=1, bias=False)
            self.l2 = nn.Conv2d(rank, features, kernel_size=1, bias=bias)
        if pgd_hook:
            self.apply_pgd_hooks()

    def project(self, verbose=False):
        pass

    def regularization(self):
        pass

    def apply_pgd_hooks(self):
        pass
    
    def forward(self, x):
        return self.l2(self.l1(x))


class Stitcher(nn.Module):
    def __init__(self, style: str, previous_module: nn.Module, bias=False, 
        batch_norm=False, wreath=False, tau: float = 1.0, alpha:float=1e-3,
        scalars=True, birkhoff=False, lasso=False, multiplier = 0.1, lowrank=False, 
        rank=1):
        super().__init__()
        if wreath and birkhoff:
            raise NotImplementedError('wreath, birkhoff are mut ex.')
        self.bn1 = nn.Identity()
        self.bn2 = nn.Identity()
        pmf = previous_module.features
        if style == 'linear':
            if wreath:
                self.A = Wreath(pmf.shape[-1], 
                    style=style, bias=(bias and not batch_norm), 
                    tau=tau, scalars=scalars)
            elif birkhoff:
                self.A = Birkhoff(pmf.shape[-1], 
                    style=style, bias=(bias and not batch_norm), 
                    alpha=alpha, scalars=scalars)
            elif lasso:
                self.A = Lasso(pmf.shape[-1], 
                    style=style, bias=(bias and not batch_norm),
                    multiplier=multiplier)
            elif lowrank:
                self.A = LowRank(pmf.shape[-1], 
                    style=style, bias=(bias and not batch_norm),
                    rank=rank)
            else: 
                self.A = nn.Linear(pmf.shape[-1], 
                    pmf.shape[-1], bias=(bias and not batch_norm))
            if batch_norm:
                self.bn1 = nn.BatchNorm1d(pmf.shape[-1], affine=bias)
                self.bn2 = nn.BatchNorm1d(pmf.shape[-1], affine=bias)
        elif style == 'conv':
            if wreath:
                self.A = Wreath(pmf.shape[-3], 
                    style=style, bias=(bias and not batch_norm), 
                    tau=tau, scalars=scalars)
            elif birkhoff:
                self.A = Birkhoff(pmf.shape[-3], 
                    style=style, bias=(bias and not batch_norm), 
                    alpha=alpha, scalars=scalars)
            elif lasso:
                self.A = Lasso(pmf.shape[-3], 
                    style=style, bias=(bias and not batch_norm),
                    multiplier=multiplier)
            elif lowrank:
                self.A = LowRank(pmf.shape[-3], 
                    style=style, bias=(bias and not batch_norm),
                    rank=rank)
            else:
                self.A = nn.Conv2d(pmf.shape[-3], 
                    pmf.shape[-3], kernel_size=1, bias=(bias and not batch_norm))
            if batch_norm:
                self.bn1 = nn.BatchNorm2d(pmf.shape[-3], affine=bias)
                self.bn2 = nn.BatchNorm2d(pmf.shape[-3], affine=bias)
        else:
            raise NotImplementedError()
        self.f = nn.Sequential(self.bn1, self.A, self.bn2)

    def forward(self, x):
        return self.f(x)

class StitchedModel(nn.Module):
    def __init__(self, model1: nn.Module, model2: nn.Module, 
        stitch_idx: int, bias=False, batch_norm=False, 
        wreath=False, birkhoff=False, pgd_hook=False,
        tau: float = 1.0, alpha:float = 1e-3, scalars=True, 
        lasso=False, multiplier = 0.1, lowrank=False, 
        rank=1):
        super().__init__()
        if wreath and birkhoff:
            raise NotImplementedError('wreath, birkhoff are mut ex.')
        self.birkhoff = birkhoff
        self.alpha = alpha
        self.lasso, self.lowrank = lasso, lowrank
        self.multiplier, self.rank = multiplier, rank
        self.f1 = deepcopy(model1)
        self.f2 = deepcopy(model2)
        self.f1.eval()
        self.f2.eval()
        for i, (m, n) in enumerate(zip(self.f1.named_modules(), self.f2.named_modules())):
            if i < stitch_idx:
                continue
            elif i == stitch_idx:
                print(type(m[1]))
                self.stitches = Stitcher('conv', m[1], bias=bias, 
                    batch_norm=batch_norm, 
                    wreath=wreath, birkhoff = birkhoff,
                    tau=tau, alpha=alpha, scalars=scalars, lasso=lasso, 
                    lowrank=lowrank, multiplier=multiplier, rank=rank)
                self.stitches.train()
                setattr(self.f1, m[0], nn.Sequential(m[1], self.stitches))
            else:
                setattr(self.f1, m[0], n[1])
        if pgd_hook:
            self.apply_pgd_hooks()
        
    def scalars(self):
        if self.stitches.A.scalars == False:
            self.stitches.A.scalars = True

    def regularization(self):
        if not (self.birkhoff or self.lasso):
            return 0.0
        else: 
            return self.stitches.A.regularization()

    def project(self, verbose=False):
        if not self.birkhoff:
            pass
        else:
            self.stitches.A.project(verbose=verbose)

    def apply_pgd_hooks(self):
        self.stitches.A.apply_pgd_hooks()

    def forward(self, x):
        return self.f1(x)


    
class StitchedModelR20(nn.Module):
    def __init__(
        self,
        model1: nn.Module,
        model2: nn.Module,
        stitch_idx: int,
        stitch_off: bool = False,
        bias=False,
        batch_norm=False,
        wreath=False,
        birkhoff=False,
        pgd_hook=False,
        tau: float = 1.0,
        alpha: float =1e-6,
        scalars=True,
        lasso=None,
        multiplier=None,
        lowrank=None,
        rank=None
    ):
        super().__init__()
        self.birkhoff=birkhoff
        self.alpha=alpha
        self.lasso = lasso
        self.f1, self.f2 = deepcopy(model1), deepcopy(model2)
        self.f1.eval()
        self.f2.eval()
        for i, (m, n) in enumerate(zip(self.f1.named_modules(), self.f2.named_modules())):
            if i <= stitch_idx:
                for p in m[1].parameters():
                    p.requires_grad = False
            if i > stitch_idx:
                for p, q in zip(m[1].parameters(), n[1].parameters()):
                    p.data = q.data
                    p.requires_grad = False

        self.stitches = Stitcher('conv', 
            list(self.f1.named_modules())[stitch_idx][1], bias=bias, 
            batch_norm=batch_norm, 
            wreath=wreath, birkhoff = birkhoff,
            tau=tau, alpha=alpha, scalars=scalars, lasso=lasso, 
            lowrank=lowrank, multiplier=multiplier, rank=rank)
        self.stitches.train()
        named_mod = list(self.f1.named_modules())[stitch_idx][0].split('.')
        if named_mod[0] == 'block_seq':
            setattr(self.f1.block_seq[int(named_mod[1])], named_mod[2], nn.Sequential(nn.ReLU(), 
                self.stitches))
        else:
            setattr(self.f1, named_mod[-1], nn.Sequential(nn.ReLU(), 
                self.stitches)) 

        self.name = f"""
        Stitching model for {self.f1} and {self.f2}, {stitch_idx} layer
        """
        if pgd_hook:
            self.apply_pgd_hooks()

    def __repr__(self):
        return self.name
    
    def scalars(self):
        if self.stitches.A.scalars == False:
            self.stitches.A.scalars = True

    def regularization(self):
        if not (self.birkhoff or self.lasso):
            return 0.0
        else: 
            return self.stitches.A.regularization()

    def project(self, verbose=False):
        if not self.birkhoff:
            pass
        else:
            self.stitches.A.project(verbose=verbose)
        
    def apply_pgd_hooks(self):
        self.stitches.A.apply_pgd_hooks()

    def forward(self, x):
        return self.f1(x)


def get_stitch_model(layer):
    # vanilla cifar10 R20 model
    model_vanilla = resnet20()
    model_vanilla.load_state_dict(torch.load("data/cifar10-stitching/models/cifar10baseR20_1.pt"))
    model = StitchedModelR20(model_vanilla, model_vanilla, layer, stitch_off=True)
    return model, model_vanilla