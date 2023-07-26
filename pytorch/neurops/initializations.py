import torch
import torch.nn as nn
import math

"""
Reimplements kaiming_uniform directly from pytorch, making it usuable for initializing new neurons within an existing layer.
"""
def kaiming_uniform(tensor: torch.Tensor, a: float = math.sqrt(5.0), mode: str = 'fan_in', nonlinearity: str = 'leaky_relu', fan: int = -1, input: torch.Tensor = None, output_normalize: bool = False):
    if fan == -1:
        fan = nn.init._calculate_correct_fan(tensor, mode)
    bound = nn.init.calculate_gain(nonlinearity, a) * math.sqrt(3.0/fan)
    with torch.no_grad():
        tensor = tensor.uniform_(-bound, bound)
    if output_normalize and input is not None:
        if len(tensor.shape) == 4:
            output = torch.nn.functional.conv2d(input, tensor, stride=1)
            tensor = tensor / torch.norm(output, dim=(0,2,3)).reshape(-1,1,1,1)
        else:
            output = input.mm(tensor.t())
            tensor = tensor / torch.norm(output, dim=0).reshape(-1,1)
    return tensor


"""
Implementation of iterative orthogonal initialization from Daneshmand et al. (2021)
"""
def iterative_orthogonalization(weights: torch.Tensor, input: torch.Tensor, stride: int = 1, output_normalize: bool = False):
    shape = weights.shape
    if len(weights.shape) == 4:
        input = torch.nn.functional.unfold(input,kernel_size=weights.size(2),stride=stride)
        input = input.transpose(1,2)
        input = input.reshape(input.size(0)*input.size(1),input.size(2))
    if len(input.shape) == 4 and len(weights.shape) == 2:
        input = input.flatten(start_dim=1)
    numneurons = weights.size(0)
    u, s, v = torch.svd(input)
    weights = (u[:numneurons,:numneurons].mm(torch.diag(1/torch.sqrt(s[:numneurons]))).mm(v[:,:numneurons].t())).reshape(-1, *weights.shape[1:])
    if weights.shape != shape:
        weights = torch.cat((weights, kaiming_uniform(torch.zeros(*shape))[weights.shape[0]:]), dim=0)
    if output_normalize:
        if len(weights.shape) == 4:
            output = torch.nn.functional.conv2d(input, weights, stride=stride)
            weights = weights / torch.norm(output, dim=(0,2,3)).reshape(-1,1,1,1)
        else:
            output = input.mm(weights.t())
            weights = weights / torch.norm(output, dim=0).reshape(-1,1)
    return weights

def north_select(new_weights: torch.Tensor, old_weights: torch.Tensor, input: torch.Tensor, candidates: int = -1, 
                 output_normalize: bool = False, init = "kaiming_uniform", stride=1, limit_ratio=-1, nonlinearity="relu"):
    if candidates == -1:
        candidates = new_weights.size(0)*100
    if input is None:
        new_weights = kaiming_uniform(new_weights)
        return new_weights
    if len(input.shape) == 4 and len(new_weights.shape) == 2:
        input = input.flatten(start_dim=1)
    candidate_weights = torch.zeros(candidates, *new_weights.shape[1:])
    if init == "autoinit":
        candidate_weights = autoinit(candidate_weights, input, output_normalize=output_normalize)
    elif init == "iterative_orthogonalization":
        candidate_weights = iterative_orthogonalization(candidate_weights, input, stride=stride, output_normalize=output_normalize)
    else:
        candidate_weights = kaiming_uniform(candidate_weights, input=input, output_normalize=output_normalize)
    if len(new_weights.shape) == 4:
        activations = torch.nn.functional.conv2d(input, torch.cat((candidate_weights, old_weights), dim=0), stride=stride)
        activations = torch.transpose(torch.transpose(activations, 0, 1).reshape(activations.shape[1], -1), 0, 1)
        if limit_ratio > 0 and activations.shape[0]/activations.shape[1] > limit_ratio:
            sampleindices = torch.randperm(activations.shape[0])[:activations.shape[1]*limit_ratio]
            activations = activations[sampleindices]
    else:
        activations = input.mm(torch.cat((candidate_weights, old_weights), dim=0).t())
    if nonlinearity == "relu":
        activations = torch.relu(activations)
    scores = torch.zeros(candidates)
    pruned_activations = activations[:, candidates:]
    proj =  pruned_activations @ torch.pinverse(pruned_activations)
    for neuron in range(candidates):
        scores[neuron] = torch.norm(activations[:, neuron] - proj @ activations[:, neuron])
    _, indices = torch.sort(scores, descending=True)
    new_weights = candidate_weights[indices[:new_weights.size(0)]]
    if output_normalize:
        if len(new_weights.shape) == 4:
            output = torch.nn.functional.conv2d(input, new_weights, stride=stride)
            new_weights = new_weights / torch.norm(output, dim=(0,2,3)).reshape(-1,1,1,1)
        else:
            output = input.mm(new_weights.t())
            new_weights = new_weights / torch.norm(output, dim=0).reshape(-1,1)
    return new_weights

"""
Implementation of AutoInit (zero mean and unit variance given inputs) from Bingham & Miikkulainen 2023
"""
def autoinit(weights: torch.Tensor, input: torch.Tensor, uniform=True, output_normalize: bool = False, stride=1):
    denominator = math.sqrt(input.shape[1]*(input.mean().item()**2+input.var().item()))
    if uniform:
        weights = weights.uniform_(-math.sqrt(3)/denominator, math.sqrt(3)/denominator)
    else:
        weights = weights.normal_(0, 1/denominator)
    if output_normalize:
        if len(weights.shape) == 4:
            output = torch.nn.functional.conv2d(input, weights, stride=stride)
            weights = weights / torch.norm(output, dim=(0,2,3)).reshape(-1,1,1,1)
        else:
            output = input.mm(weights.t())
            weights = weights / torch.norm(output, dim=0).reshape(-1,1)
    return weights