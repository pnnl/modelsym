from typing import Optional, Tuple
import torch.nn as nn
import torch
from copy import deepcopy


class LayerModifier(nn.Module):
    """Modifies an nn.Module from W to MW,
    where W is the original op and M is an orthonormal matrix.
    Projection implemented via a 1x1 convolution."""

    def __init__(
        self,
        original_op: nn.Module,
        dims: Optional[Tuple] = None,
        matrix_group: str = 'orthogonal',
        rand_weight: Optional[torch.tensor] = None,
        device: Optional[torch.device] = None
    ):
        super(LayerModifier, self).__init__()
        self.original_op = original_op
        if dims == None:
            dims = self.find_dims()
        self.dims = dims
        if not device:
            device = original_op.weight.device
        self.device = device
        if not rand_weight:
            if matrix_group == 'orthogonal':
                rand_weight = self.orthogonal_sample(dims)
            elif matrix_group == 'Grelu':
                rand_weight = self.Grelu_sample(dims)
        if isinstance(original_op, torch.nn.Conv2d):
            self.transform = self.conv_1x1(rand_weight)
        elif isinstance(original_op, torch.nn.Linear):
            self.transform = self.lin(rand_weight)
        else:
            raise NotImplementedError(
                f"Only for linear and conv2d, not {type(original_op)}"
                )      
    def orthogonal_sample(self, d):
        return torch.nn.init.orthogonal_(torch.empty((d, d))).to(
                    self.device
                )
    def Grelu_sample(self, d, sigma=1.0):
        sigma = torch.tensor(sigma)
        l = sigma*torch.randn((d, ))
        l = torch.exp(l)/torch.exp(0.5*sigma**2)
        p = torch.randperm(d)
        weight = torch.zeros((d,d))
        weight[torch.arange(d), p] = l
        return weight.to(self.device)
    def conv_1x1(self, rand_weight):
        rand_weight = torch.nn.Parameter(rand_weight[..., None, None])
        rand_weight.requires_grad = False
        conv_1x1 = nn.Conv2d(
            in_channels=self.dims, out_channels=self.dims, kernel_size=1, bias=False
        ).to(self.device)
        conv_1x1.weight = rand_weight  
        return conv_1x1
    def lin(self, rand_weight):
        rand_weight = torch.nn.Parameter(rand_weight)
        rand_weight.requires_grad = False
        l = nn.Linear(in_features=self.dims, 
            out_features=self.dims, bias=False).to(self.device)
        l.weight = rand_weight
        return l
    def find_dims(self):
        if isinstance(self.original_op, torch.nn.Conv2d):
            d = self.original_op.out_channels
        elif isinstance(self.original_op, torch.nn.Linear):
            d = self.original_op.out_features
        else:
            raise NotImplementedError(
                f"Only for linear and conv2d, not {type(self.original_op)}"
            )
        return d

    def forward(self, x):
        x = self.original_op(x)
        x = self.transform(x)
        return x

def rotate_layer(model, layer, matrix_group = 'orthogonal', freeze_features=True):
    m = deepcopy(model)
    m[layer] = LayerModifier(m[layer], matrix_group=matrix_group)
    if freeze_features:
        for l in range(layer+1):
            for p in m[l].parameters():
                p.requires_grad = False
    return m