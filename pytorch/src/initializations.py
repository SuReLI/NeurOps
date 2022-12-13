import torch
import torch.nn as nn

def kaiming_uniform(tensor: torch.Tensor, a: float = 5**(1/2), mode: str = 'fan_in', nonlinearity: str = 'leaky_relu', fan: int = -1):
    if fan == -1:
        fan = nn.init._calculate_correct_fan(tensor, mode)
    bound = torch.sqrt(3.0) *nn.init.calculate_gain(nonlinearity, a) / torch.sqrt(fan)
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

def iterative_orthogonalization(weights: torch.Tensor, input: torch.Tensor, stride: int = 1, fill_kaiming: bool = True):
    if len(weights.shape) == 4:
        input = torch.nn.functional.unfold(input,kernel_size=weights.size(2),stride=stride)
        input = input.transpose(1,2)
        input = input.reshape(input.size(0)*input.size(1),input.size(2))
    numneurons = weights.size(0)
    u, s, v = torch.svd(input)
    #with torch.no_grad():
    weights = (u[:numneurons,:numneurons].mm(torch.diag(1/torch.sqrt(s[:numneurons]))).mm(v[:,:numneurons].t())).reshape(numneurons, *weights.shape[1:])
    #TODO: normalize if needed
    return weights

