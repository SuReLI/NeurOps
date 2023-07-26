import torch

"""
Whole layer metrics
"""

"""
Measure effective rank of whole layer via thresholding singular values of 
activations (or weights) (high score = low redundancy)
"""
def effective_rank(tensor: torch.Tensor = None, threshold: float = 0.01, 
                 partial: bool = False, scale: bool = True, limit_ratio = -1):
    if tensor is None:
        return None
    if len(tensor.shape) > 2:
        tensor = torch.transpose(torch.transpose(tensor, 0, 1).reshape(tensor.shape[1], -1), 0, 1)
        if limit_ratio > 0 and tensor.shape[0]/tensor.shape[1] > limit_ratio:
            sampleindices = torch.randperm(tensor.shape[0])[:tensor.shape[1]*limit_ratio]
            tensor = tensor[sampleindices]
    if scale:
        tensor = tensor.clone() / tensor.shape[1]**0.5
    _, S, _ = torch.svd(tensor, compute_uv=False)
    effdim = torch.count_nonzero(S > threshold)
    if partial:
        effdim = effdim.float() + torch.sum(S)
    return effdim

"""
Measure orthogonality gap of activations. Score of 0 means completely orthogonal, score of 1 means completely redundant
Used by Daneshmand et al. (2021) (theoretical version is covariance-based, implementation is SVD-based)
"""
def orthogonality_gap(activations: torch.Tensor = None, svd = True, norm_neurons: bool = False):
    if activations is None:
        return None
    if norm_neurons:
        activations = activations / torch.norm(activations, dim=1, keepdim=True) #TEST
    if len(activations.shape) > 2:
        activations = activations.reshape(activations.shape[0], -1)
    cov = activations @ activations.t()
    return torch.norm(cov/(torch.norm(activations)**2) - torch.eye(activations.shape[0]).to(cov.device)/activations.shape[0], p='fro')





"""
Per-neuron metrics
"""

"""
Measure l1 or l2 (etc.) sum of weights for each neuron in a layer
L1 sum of weights is used by Li et al (2019) to measure neuron importance (high score = high importance)

weights: weights of a layer
p: p-norm to use (int, inf, -inf, "fro", "nuc")
fanin: whether to measure w.r.t. fan-in weights (so output length is # of output neurons) or fan-out weights
"""
def weight_sum(weights: torch.Tensor = None, p = 1, fanin: bool = True, conversion_factor: int = -1):
    if weights is None:
        return None
    if not fanin:
        weights = torch.transpose(weights, 0, 1)
    if len(weights.shape) > 2:
        weights = weights.reshape(weights.shape[0], -1)
    if conversion_factor != -1:
        weights = weights.reshape(-1, conversion_factor, *weights.shape[1:])
        weights = weights.reshape(weights.shape[0], -1)
    return torch.norm(weights, p=p, dim=1)

"""
Measure variance of activations for each neuron in a layer, used by Polyak 
and Wolf (2015) to measure neuron importance (high score = high importance)
"""
def activation_variance(activations: torch.Tensor = None):
    if activations is None:
        return None
    if len(activations.shape) > 2:
        activations = torch.transpose(torch.transpose(activations, 0, 1).reshape(activations.shape[1], -1), 0, 1)
    return torch.var(activations, dim=0)

"""
Measure effective rank per neuron when that neuron is left out of the 
computation (high score = high redundancy)
Variation of method used by Maile et al. (2022) for selection of neurogenesis initialization candidates
"""
def svd_score(tensor: torch.Tensor = None, threshold: float = 0.01, addwhole: bool = False, 
             scale: bool = True, difference: bool = False, limit_ratio = -1):
    if tensor is None:
        return None
    if difference:
        _, S, _ = torch.svd(tensor, compute_uv=False)
        baseeffdim = torch.sum(S)
        if addwhole:
            baseeffdim += torch.count_nonzero(S > threshold).float()
    else:
        baseeffdim = 0
    scores = torch.zeros(tensor.shape[1])
    for neuron in range(tensor.shape[1]):
        prunedtensor = torch.cat((tensor[:, :neuron], tensor[:, neuron+1:]), dim=1)
        if len(prunedtensor.shape) > 2:
            prunedtensor = prunedtensor.reshape(tensor.shape[0], -1)
            if limit_ratio > 0 and prunedtensor.shape[1]/tensor.shape[0] > limit_ratio:
                sampleindices = torch.randperm(prunedtensor.shape[1])[:prunedtensor.shape[0]*limit_ratio]
                prunedtensor = prunedtensor[:, sampleindices]
        if scale:
            prunedtensor /= prunedtensor.shape[1]**0.5
        _, S, _ = torch.svd(prunedtensor, compute_uv=False)
        effdim = torch.sum(S)
        if addwhole:
            effdim += torch.count_nonzero(S > threshold).float()
        scores[neuron] = effdim-baseeffdim
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
Measure correlation between activations of each neuron (dim 1) and the rest of the layer
(high score = high redundance)

Used by Suau et al 2018
"""
def correlation_score(activations: torch.Tensor = None, crosscorr: bool = True):
    if activations is None:
        return None
    if len(activations.shape) > 2:
        activations = torch.transpose(torch.transpose(activations, 0, 1).reshape(activations.shape[1], -1), 0, 1)
    if crosscorr:
        corr = torch.nan_to_num(torch.corrcoef(activations.t()),0)
    else:
        corr = activations.t() @ activations
    return torch.sum(torch.square(corr), dim=1)

"""
Measure correlation between activations when each neuron is left out of the computation: 
(high score = low redundance)
"""
def dropped_corr_score(activations: torch.Tensor = None):
    if activations is None:
        return None
    if len(activations.shape) > 2:
        activations = torch.transpose(torch.transpose(activations, 0, 1).reshape(activations.shape[1], -1), 0, 1)
    scores = torch.zeros(activations.shape[1])
    for neuron in range(activations.shape[1]):
        pruned_activations = torch.cat((activations[:, :neuron], activations[:, neuron+1:]), dim=1)
        corr = pruned_activations.t() @ pruned_activations
        scores[neuron] = torch.sum(torch.sum(corr,dim=1)/(torch.diagonal(corr)+1e-8))
    return scores

"""
Measure Average Percentage of Zeros (APoZ) per neurons (high score = low importance)
Used by Hu et al 2016
"""
def apoz_score(activations: torch.Tensor = None):
    if activations is None:
        return None
    if len(activations.shape) > 2:
        activations = torch.transpose(torch.transpose(activations, 0, 1).reshape(activations.shape[1], -1), 0, 1)
    return torch.mean((torch.abs(activations) < 1e-8).float(), dim=0)

"""
Measure reconstruction error of each neuron when that neuron is projected onto the rest of the layer. 
(high score = low redundancy) 
Used by Berg 2022
"""
def reconstruction_score(activations: torch.Tensor = None, limit_ratio = -1):
    if activations is None:
        return None
    if len(activations.shape) > 2:
        activations = torch.transpose(torch.transpose(activations, 0, 1).reshape(activations.shape[1], -1), 0, 1)
        if limit_ratio > 0 and activations.shape[0]/activations.shape[1] > limit_ratio:
            sampleindices = torch.randperm(activations.shape[0])[:activations.shape[1]*limit_ratio]
            activations = activations[sampleindices]
    scores = torch.zeros(activations.shape[1])
    for neuron in range(activations.shape[1]):
        pruned_activations = torch.cat((activations[:, :neuron], activations[:, neuron+1:]), dim=1)
        scores[neuron] = torch.norm(activations[:, neuron] - pruned_activations @ torch.pinverse(pruned_activations) @ activations[:, neuron])
    return scores


"""
Measure fisher information of mask gradients: assume 0th dim is batch dim and rest are weight dims

Used by Kwon et al. (2022)
"""
def fisher_info(maskgrads: torch.Tensor = None):
    if maskgrads is None:
        return None
    return maskgrads.pow(2).sum(dim=0)