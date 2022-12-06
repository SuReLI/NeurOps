import torch
import torch.nn as nn

def kaiming_uniform(tensor: torch.Tensor, a: float = 0, mode: str = 'fan_in', nonlinearity: str = 'leaky_relu', fan: int = -1):
    if fan == -1:
        fan = torch.nn.init._calculate_correct_fan(tensor, mode)
    gain = torch.nn.init.calculate_gain(nonlinearity, a)
    std = gain / torch.sqrt(fan)
    bound = torch.sqrt(3.0) * std  
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)



