import torch
import torch.nn as nn

"""
Measure effective rank via thresholding singular values of activations (or weights)
"""
def effectivesvd(tensor: torch.Tensor = None, threshold: float = 0.01, partial: bool = False, scale: bool = True):
    if tensor is None:
        return None
    if len(tensor.shape) > 2:
        tensor = tensor.reshape(tensor.shape[0], -1) #TODO: check if this is correct for acts and weights
    if scale:
        tensor = tensor.clone() / tensor.shape[1]**0.5
    _, S, _ = torch.svd(tensor, compute_uv=False)
    effdim = torch.count_nonzero(S > threshold)
    if partial:
        effdim = effdim.float() + torch.sum(S)
    return effdim

"""
Measure effective rank when each neuron is left out of the computation
"""
def svdscore(tensor: torch.Tensor = None, threshold: float = 0.01, addwhole: bool = False, scale: bool = True):
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
