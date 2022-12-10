import torch
import torch.nn as nn

"""
Measure effective rank via thresholding singular values of activations (or weights)
"""
def effectivesvd(layer: nn.Module = None, tensor: torch.Tensor = None, threshold: float = 0.01, partial: bool = False, useacts: bool = True):
    if layer is not None:
        if useacts:
            tensor = layer.activations
        else:
            tensor = layer.weight
    if tensor is None:
        return None
    if len(tensor.shape) > 2:
        tensor = tensor.reshape(tensor.shape[0], -1) #TODO: check if this is correct for acts and weights
    _, S, _ = torch.svd(tensor)
    effdim = torch.count_nonzero(S > threshold)
    if partial:
        effdim = effdim.float() + torch.sum(S)
    return effdim
