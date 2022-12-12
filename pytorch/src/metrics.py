import torch
import torch.nn as nn

"""
Measure l1 or l2 (etc.) sum of weights for each neuron in a layer
L1 sum of weights is used by Li et al (2019) to measure neuron importance

weights: weights of a layer
p: p-norm to use (int, inf, -inf, "fro", "nuc")
fanin: whether to measure w.r.t. fan-in or fan-out weights
"""
def weightsum(weights: torch.Tensor = None, p = 1, fanin: bool = True):
    if weights is None:
        return None
    if not fanin:
        weights = weights.t()
    if len(weights.shape) > 2:
        weights = weights.reshape(weights.shape[0], -1)
    return torch.norm(weights, p=p, dim=0)

"""
Measure variance of activations for each neuron in a layer, used by Polyak 
and Wolf (2015) to measure neuron importance
"""
def actvar(acts: torch.Tensor = None):
    if acts is None:
        return None
    if len(acts.shape) > 2:
        acts = acts.reshape(acts.shape[0], -1)
    return torch.var(acts, dim=0)

"""
Measure effective rank of whole layer via thresholding singular values of 
activations (or weights)
"""
def effectivesvd(tensor: torch.Tensor = None, threshold: float = 0.01, 
                 partial: bool = False, scale: bool = True):
    if tensor is None:
        return None
    if len(tensor.shape) > 2:
        tensor = tensor.reshape(tensor.shape[0], -1) #TODO: check if this is correct for acts and weights
    if scale:
        tensor /= tensor.shape[1]**0.5
    _, S, _ = torch.svd(tensor, compute_uv=False)
    effdim = torch.count_nonzero(S > threshold)
    if partial:
        effdim = effdim.float() + torch.sum(S)
    return effdim

"""
Measure effective rank per neuron when that neuron is left out of the 
computation
"""
def svdscore(tensor: torch.Tensor = None, threshold: float = 0.01, addwhole: bool = False, 
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
