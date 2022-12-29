import torch
import torch.nn as nn

"""
Measure l1 or l2 (etc.) sum of weights for each neuron in a layer
L1 sum of weights is used by Li et al (2019) to measure neuron importance

weights: weights of a layer
p: p-norm to use (int, inf, -inf, "fro", "nuc")
fanin: whether to measure w.r.t. fan-in weights (so output length is # of output neurons) or fan-out weights
"""
def weight_sum(weights: torch.Tensor = None, p = 1, fanin: bool = True):
    if weights is None:
        return None
    if fanin:
        weights = weights.t()
    if len(weights.shape) > 2:
        weights = weights.reshape(weights.shape[0], -1)

    return torch.norm(weights, p=p, dim=0)

"""
Measure variance of activations for each neuron in a layer, used by Polyak 
and Wolf (2015) to measure neuron importance
"""
def activation_variance(activations: torch.Tensor = None):
    if activations is None:
        return None
    if len(activations.shape) > 2:
        activations = activations.reshape(activations.shape[0], -1)
    return torch.var(activations, dim=0)

"""
Measure effective rank of whole layer via thresholding singular values of 
activations (or weights)
"""
def effective_rank(tensor: torch.Tensor = None, threshold: float = 0.01, 
                 partial: bool = False, scale: bool = True):
    if tensor is None:
        return None
    if len(tensor.shape) > 2:
        tensor = tensor.reshape(tensor.shape[0], -1) #TODO: check if this is correct for activations and weights
    if scale:
        tensor = tensor.clone() / tensor.shape[1]**0.5
    _, S, _ = torch.svd(tensor, compute_uv=False)
    effdim = torch.count_nonzero(S > threshold)
    if partial:
        effdim = effdim.float() + torch.sum(S)
    return effdim

"""
Measure orthogonality gap of activations
Used by Daneshmand et al. (2021)
"""
def orthogonality_gap(activations: torch.Tensor = None):
    if activations is None:
        return None
    if len(activations.shape) > 2:
       activations = activations.reshape(activations.shape[0], -1)
    cov = activations @ activations.t()
    return torch.norm(cov*torch.trace(cov) - torch.eye(activations.shape[0]).to(cov.device)/activations.shape[0], p='fro')


"""
Measure effective rank per neuron when that neuron is left out of the 
computation
Used by Maile et al. (2022) for selection of neurogenesis initialization candidates
"""
def svd_score(tensor: torch.Tensor = None, threshold: float = 0.01, addwhole: bool = False, 
             scale: bool = True):
    if tensor is None:
        return None
    scores = torch.zeros(tensor.shape[1])
    for neuron in range(tensor.shape[1]):
        prunedtensor = torch.cat((tensor[:, :neuron], tensor[:, neuron+1:]), dim=1)
        if len(prunedtensor.shape) > 2:
            prunedtensor = prunedtensor.reshape(tensor.shape[0], -1)
        if scale:
            prunedtensor /= prunedtensor.shape[1]**0.5
        _, S, _ = torch.svd(prunedtensor, compute_uv=False)
        effdim = torch.sum(S)
        if addwhole:
            effdim += torch.count_nonzero(S > threshold).float()
        scores[neuron] = effdim
    return scores

"""
Measure nuclear norm (sum of singular values) of activations per neuron when that neuron is left out
of the computation
Average version used by Sui et al. (2021) for channel pruning
"""
def nuclear_score(activations: torch.Tensor = None, average: bool = False):
    if activations is None:
        return None
    scores = torch.zeros(activations.shape[1])
    if average and len(activations.shape) > 2:
        activations = activations.reshape(activations.shape[0], activations.shape[1], -1) 
    for neuron in range(activations.shape[1]):
        pruned_activations = torch.cat((activations[:, :neuron], activations[:, neuron+1:]), dim=1)
        if not average: 
            if len(pruned_activations.shape) > 2:
                pruned_activations = pruned_activations.reshape(activations.shape[0], -1)
            scores[neuron] = torch.norm(pruned_activations, p='nuc')
        else:
            scores[neuron] = torch.mean(torch.norm(pruned_activations, p='nuc', dim=(1,2)))
    return scores

"""
Measure fisher information of mask gradients: assume 0th dim is batch dim and rest are weight dims

Used by Kwon et al. (2022)
"""
def fisher_info(maskgrads: torch.Tensor = None):
    if maskgrads is None:
        return None
    return maskgrads.pow(2).sum(dim=0)