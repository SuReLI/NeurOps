import torch
import torch.nn as nn

from transformers import prune_layer

from functools import partial
from collections import defaultdict

from .layers import ModLinear, ModConv2d

"""
A modifiable sequential container that allows for masking, pruning, and growing of layers.

"""
class ModSequential(nn.Sequential):
    def __init__(self,  *args, track_activations: bool = False):
        super(ModSequential, self).__init__(*args)
        self.track_activations = track_activations
        if self.track_activations:
            self.activations = defaultdict(torch.Tensor)
            for name, module in self.named_modules():
                if isinstance(module, ModLinear) or isinstance(module, ModConv2d):
                    module.register_forward_hook(partial(self._acthook, name))

    def _acthook(self, name, module, input, output):
        self.activations[name] = torch.cat((self.activations[name], output.detach()), dim=0)
        if self.activations[name].shape[0] > 2*self.activations[name].shape[1]:
            #self.activations[name] = self.activations[name][min(-2*self.activations[name].shape[1], output.shape[1]):]
            self.activations[name] = self.activations[name][-2*self.activations[name].shape[1]:]
    
    def forward(self, x):
        for module in self._modules.values():
            x = module(x)
        return x

    def limited_forward(self, x, layer_index: int):
        for i, module in enumerate(self._modules.values()):
            x = module(x)
            if i == layer_index:
                return x
        return x

    def mask(self, layer_index: int, neurons: list = [], clear_activations: bool = False):
        for i, module in enumerate(self._modules.values()):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.mask(neurons, [])
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.mask([], neurons)
        if clear_activations and self.track_activations:
            for index in (layer_index+1, layer_index):
                self.activations[str(index)] = torch.Tensor()

    
    def unmask(self, layer_index: int, neurons: list = [], clear_activations: bool = False):
        for i, module in enumerate(self._modules.values()):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.unmask(neurons, [])
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.unmask([], neurons)
        if clear_activations and self.track_activations:
            for index in (layer_index+1, layer_index):
                self.activations[str(index)] = torch.Tensor()
            

    def prune(self, layer_index: int, neurons: list = [], optimizer=None, clear_activations: bool = False):
        for i, module in enumerate(self._modules.values()):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.prune(neurons, [], optimizer=optimizer)
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.prune([], neurons, optimizer=optimizer)
        for index in (layer_index+1, layer_index):
            if clear_activations and self.track_activations:
                self.activations[str(index)] = torch.Tensor()
            elif index == layer_index and self.track_activations and len(self.activations[str(index)].shape) >= 2:
                neurons_to_keep = range(self.activations[str(index)].shape[1])
                neurons_to_keep = [
                    ntk for ntk in neurons if ntk not in neurons]
                self.activations[str(index)] = self.activations[str(index)][:, neurons_to_keep]


    def grow(self, layer_index: int, newneurons: int = 0, fanin_weights=None, fanout_weights=None, 
             optimizer=None, clear_activations: bool = False, sendactivations: bool = False):
        for i, module in enumerate(self._modules.values()):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.grow(newneurons, 0, fanin_weights = fanin_weights, optimizer=optimizer, 
                activations=self.activations[str(layer_index-1)] if sendactivations else None)
                if self.track_activations:
                    if clear_activations:
                        self.activations[str(layer_index)] = torch.Tensor()
                    else:
                        self.activations[str(layer_index)] = torch.cat(
                            (self.activations[str(layer_index)], 
                             torch.zeros(self.activations[str(layer_index)].shape[0], newneurons)), dim=1)
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.grow(0, newneurons, fanout_weights = fanout_weights, optimizer=optimizer, 
                activations=self[layer_index](self.activations[str(layer_index-1)]) if sendactivations else None)
                if clear_activations and self.track_activations:
                    self.activations[str(layer_index+1)] = torch.Tensor()
           

"""
A wrapper for the HuggingFace Transformer model that allows for masking of attention heads and of 
hidden neurons in the FFN layer of each transformer block.
"""
class ModTransformer(nn.Module):
    def __init__(self, model, track_activations: bool=False):
        super(ModTransformer, self).__init__()
        self.model = model
        self.config = model.config
        for param in model.parameters():
            param.requires_grad_(False)
        self.head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads).to(model.device)
        self.neuron_mask = torch.ones(model.config.num_hidden_layers, model.config.intermediate_size).to(model.device)
        self.handles = self.register_neuron_mask()
        self.track_activations = track_activations
        if track_activations:
            model.config.output_attentions = True
            self.head_activations = defaultdict(torch.Tensor)
            self.neuron_activations = defaultdict(torch.Tensor)
            for index, layer in enumerate(getattr(self.model, self.model.base_model_prefix).encoder.layer):
                layer.output.register_forward_pre_hook(partial(self._neuron_act_pre_hook, index)) 
                layer.register_forward_hook(partial(self._head_act_hook, index)) 

    def register_neuron_mask(self):
        handles = []
        for index, layer in enumerate(getattr(self.model, self.model.base_model_prefix).encoder.layer):
            hook = lambda module, inputs, outputs: outputs * self.neuron_mask[index]
            handles.append(layer.intermediate.register_forward_hook(hook))
        return handles
    
    def unregister_neuron_mask(self):
        for handle in self.handles:
            handle.remove()

    def _head_act_hook(self, index, module, input, output):
        self.head_activations[index] = torch.cat((self.head_activations[index], output[1].detach()), dim=0)
        if self.head_activations[index].shape[0] > 2*self.head_activations[index].shape[1]:
            self.head_activations[index] = self.head_activations[index][-2*self.head_activations[index].shape[1]:]
    
    def _neuron_act_pre_hook(self, index, module, input):
        self.neuron_activations[index] = torch.cat((self.neuron_activations[index], input[0].detach()), dim=0)
        if self.neuron_activations[index].shape[0] > 2*self.neuron_activations[index].shape[1]:
            self.neuron_activations[index] = self.neuron_activations[index][-2*self.neuron_activations[index].shape[1]:]
    
    def forward(self, x):
        return self.model(x, head_mask=self.head_mask)

    def mask_neurons(self, layer_index: int, neurons: list):
        self.neuron_mask[layer_index][neurons] = 0

    def mask_heads(self, layer_index: int, neurons: list):
        self.head_mask[layer_index][neurons] = 0

    def unmask_neurons(self, layer_index: int, neurons: list):
        self.neuron_mask[layer_index][neurons] = 1

    def unmask_heads(self, layer_index: int, neurons: list):
        self.head_mask[layer_index][neurons] = 1
    
    
