import torch
import torch.nn as nn
import math

"""
Reimplements kaiming_uniform directly from pytorch, making it usuable for initializing new neurons within an existing layer.
"""
def kaiming_uniform(tensor: torch.Tensor, a: float = 5**(1/2), mode: str = 'fan_in', nonlinearity: str = 'relu', fan: int = -1):
    if fan == -1:
        fan = nn.init._calculate_correct_fan(tensor, mode)
    bound = nn.init.calculate_gain(nonlinearity, a) * math.sqrt(3.0/fan)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

"""
Implementation of iterative orthogonal initialization from Daneshmand et al. (2021)
"""
def iterative_orthogonalization(weights: torch.Tensor, input: torch.Tensor, stride: int = 1, output_normalize: bool = False):
    if len(weights.shape) == 4:
        input = torch.nn.functional.unfold(input,kernel_size=weights.size(2),stride=stride)
        input = input.transpose(1,2)
        input = input.reshape(input.size(0)*input.size(1),input.size(2))
    if len(input.shape) == 4 and len(weights.shape) == 2:
        input = input.flatten(start_dim=1)
    numneurons = weights.size(0)
    u, s, v = torch.svd(input)
    weights = (u[:numneurons,:numneurons].mm(torch.diag(1/torch.sqrt(s[:numneurons]))).mm(v[:,:numneurons].t())).reshape(numneurons, *weights.shape[1:])
    if output_normalize:
        if len(weights.shape) == 4:
            output = torch.nn.functional.conv2d(input, weights, stride=stride)
        else:
            output = input.mm(weights.t())
        weights = weights / torch.norm(output)
    return weights

