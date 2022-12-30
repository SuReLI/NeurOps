import torch
import torch.nn as nn

from functools import partial
from collections import defaultdict
from math import prod

from .layers import ModLinear, ModConv2d


"""
A modifiable sequential container that allows for masking, pruning, and growing of layers.
"""
class ModSequential(nn.Sequential):
    def __init__(self,  *args, track_activations: bool = False, track_auxiliary_gradients: bool = False, input_features: int = None, input_shape: tuple = None):
        
        super(ModSequential, self).__init__(*args)

        self.track_activations = track_activations
        self.track_auxiliary_gradients = track_auxiliary_gradients

        if self.track_activations:
            self.activations = defaultdict(torch.Tensor)
            for name, module in self.named_modules():
                if isinstance(module, ModLinear) or isinstance(module, ModConv2d):
                    module.register_forward_hook(partial(self._act_hook, name))

        modules = list(self._modules.values())
        self.conversion_layer = -1
        self.conversion_factor = 1
        if input_shape is not None:
            for i in range(len(modules)-1):
                if isinstance(modules[i+1], nn.Linear) and isinstance(modules[i], nn.Conv2d):
                    self.conversion_layer = i
                    self.conversion_factor = prod(self(torch.zeros(1, *input_shape), layer_index=i).shape[1:])   

        if input_features is None:
            if isinstance(modules[0], nn.Linear):
                input_features = modules[0].in_features
            elif isinstance(modules[0], nn.Conv2d):
                input_features = modules[0].in_channels
        self.input_features = input_features

        if self.track_auxiliary_gradients:
            self.auxiliaries = []
            for i in range(1, len(modules)):
                if isinstance(modules[i-1], nn.Linear):
                    aux = torch.zeros(modules[i].out_features, modules[i-1].in_features)
                elif isinstance(modules[i], nn.Conv2d):
                    aux = torch.zeros(modules[i].out_channels, modules[i-1].in_channels, 
                                      modules[i-1].kernel_size[0]+modules[i].kernel_size[0]-1, 
                                      modules[i-1].kernel_size[1]+modules[i].kernel_size[1]-1)
                else:
                    aux = torch.zeros(modules[i].out_features, modules[i-1].in_channels, 
                                      modules[i-1].kernel_size[0], modules[i-1].kernel_size[1])
                self.auxiliaries.append(nn.parameter.Parameter(aux))

    def _act_hook(self, name, module, input, output):
        self.activations[name] = torch.cat((self.activations[name], output.detach()), dim=0)
        if self.activations[name].shape[0] > 2*self.activations[name].shape[1]:
            #self.activations[name] = self.activations[name][min(-2*self.activations[name].shape[1], output.shape[1]):]
            self.activations[name] = self.activations[name][-2*self.activations[name].shape[1]:]
    
    def _act_shape_hook(self, module, input, output):
        self.conv_output_shape = output.shape[1:]

    def parameter_count(self, masked: bool = False):
        count = 0
        for i in range(len(self)):
            if isinstance(self[i], ModLinear) or isinstance(self[i], ModConv2d):
                count += self[i].parameter_count(masked=masked, previous_mask = self[i-1].mask_vector if i > 0 else None)
            else:
                count += sum(p.numel() for p in self[i].parameters())
        return count

    def forward(self, x, auxiliaries: list = None, layer_index: int = -1):
        old_x = x
        modules = list(self._modules.values())
        for i, module in enumerate(modules):
            if i == 0 or auxiliaries is None:
                x = module(x)
            else:
                x_copy = x
                x = module(x, auxiliaries[i-1], old_x, modules[i-1])
                old_x = x_copy * modules[i-1].mask_vector.view(1, -1)
            if i == layer_index:
                return x
        return x

    def mask(self, layer_index: int, neurons: list = [], clear_activations: bool = False):
        for i, module in enumerate(self._modules.values()):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.mask(neurons)
        if clear_activations and self.track_activations:
            for index in (layer_index+1, layer_index):
                self.activations[str(index)] = torch.Tensor()

    
    def unmask(self, layer_index: int, neurons: list = [], optimizer=None, clear_activations: bool = False):
        for i, module in enumerate(self._modules.values()):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.unmask(neurons, [], optimizer=optimizer)
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                if i == self.conversion_layer:
                    converted_neurons = sum([[*range(neuron*self.conversion_factor, (neuron+1)*self.conversion_factor)] for neuron in neurons], [])
                else:
                    converted_neurons = neurons
                module.unmask([], neurons, optimizer=optimizer)
        if clear_activations and self.track_activations:
            for index in (layer_index+1, layer_index):
                self.activations[str(index)] = torch.Tensor()
            

    def prune(self, layer_index: int, neurons: list = [], optimizer=None, clear_activations: bool = False):
        for i, module in enumerate(self._modules.values()):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.prune(neurons, [], optimizer=optimizer)
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                if i == self.conversion_layer:
                    converted_neurons = sum([[*range(neuron*self.conversion_factor, (neuron+1)*self.conversion_factor)] for neuron in neurons], [])
                else:
                    converted_neurons = neurons
                module.prune([], converted_neurons, optimizer=optimizer)
        for index in (layer_index+1, layer_index):
            if clear_activations and self.track_activations:
                self.activations[str(index)] = torch.Tensor()
            elif index == layer_index and self.track_activations and len(self.activations[str(index)].shape) >= 2:
                neurons_to_keep = range(self.activations[str(index)].shape[1])
                neurons_to_keep = [
                    ntk for ntk in neurons if ntk not in neurons]
                self.activations[str(index)] = self.activations[str(index)][:, neurons_to_keep]
        if self.track_auxiliary_gradients:
            self.auxiliaries[layer_index-1] = self.auxiliaries[layer_index-1][:, neurons_to_keep]
            self.auxiliaries[layer_index-2] = self.auxiliaries[layer_index-2][neurons_to_keep]



    def grow(self, layer_index: int, newneurons: int = 0, fanin_weights=None, fanout_weights=None, 
             optimizer=None, clear_activations: bool = False, send_activations: bool = False):
        for i, module in enumerate(self._modules.values()):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.grow(newneurons, 0, fanin_weights = fanin_weights, optimizer=optimizer, 
                activations=self.activations[str(layer_index-1)] if send_activations else None)
                if self.track_activations:
                    if clear_activations:
                        self.activations[str(layer_index)] = torch.Tensor()
                    else:
                        self.activations[str(layer_index)] = torch.cat(
                            (self.activations[str(layer_index)], 
                             torch.zeros(self.activations[str(layer_index)].shape[0], newneurons)), dim=1)
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                if i == self.conversion_layer:
                    converted_newneurons = newneurons*self.conversion_factor
                else:
                    converted_newneurons = newneurons
                module.grow(0, converted_newneurons, fanout_weights = fanout_weights, optimizer=optimizer, 
                            activations=self[layer_index](self.activations[str(layer_index-1)]) if send_activations else None)
                if clear_activations and self.track_activations:
                    self.activations[str(layer_index+1)] = torch.Tensor()
        if self.track_auxiliary_gradients:
            self.auxiliaries[layer_index-1] = torch.cat(
                (self.auxiliaries[layer_index-1], torch.zeros(self.auxiliaries[layer_index-1].shape[0], newneurons, self.auxiliaries[layer_index-1].shape[2:])), dim=1)
            self.auxiliaries[layer_index-2] = torch.cat(
                (self.auxiliaries[layer_index-2], torch.zeros(newneurons, self.auxiliaries[layer_index-1].shape[1:])), dim=0)
           

"""
A wrapper for the HuggingFace Transformer model that allows for masking of attention heads and/or of 
hidden neurons in the FFN layer of each transformer block.
"""
class ModTransformer(nn.Module):
    def __init__(self, model, track_activations: bool=False, track_auxiliary_gradients: bool=False):
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

        # in current version, only auxiliary gradients across neurons are supported
        self.track_auxiliary_gradients = track_auxiliary_gradients
        if track_auxiliary_gradients:
            self.auxiliaries = []
            for index, layer in enumerate(getattr(self.model, self.model.base_model_prefix).encoder.layer):
                self.auxiliaries.append(torch.zeros(layer.output.dense.out_features, layer.intermediate.dense.in_features, requires_grad=True).to(model.device))
                layer.intermediate.register_forward_hook(self._aux_intermediate_hook)
                layer.output.dense.register_forward_hook(self._aux_output_hook(index))
               
    def register_neuron_mask(self):
        handles = []
        for index, layer in enumerate(getattr(self.model, self.model.base_model_prefix).encoder.layer):
            hook = lambda module, inputs, outputs: outputs * self.neuron_mask[index]
            handles.append(layer.intermediate.register_forward_hook(hook))
        return handles
    
    def unregister_neuron_mask(self):
        for handle in self.handles:
            handle.remove()

    def _aux_intermediate_hook(self, module, input, output):
        self.intermediate_input = input[0]

    def _aux_output_hook(self, index):
        def hook(module, input, output):
            return output + nn.functional.linear(self.intermediate_input, self.auxiliaries[index])
        return hook

    def _head_act_hook(self, index, module, input, output):
        self.head_activations[index] = torch.cat((self.head_activations[index], output[1].detach()), dim=0)
        if self.head_activations[index].shape[0] > 2*self.head_activations[index].shape[1]:
            self.head_activations[index] = self.head_activations[index][-2*self.head_activations[index].shape[1]:]
    
    def _neuron_act_pre_hook(self, index, module, input):
        self.neuron_activations[index] = torch.cat((self.neuron_activations[index], input[0].detach()), dim=0)
        if self.neuron_activations[index].shape[0] > 2*self.neuron_activations[index].shape[1]:
            self.neuron_activations[index] = self.neuron_activations[index][-2*self.neuron_activations[index].shape[1]:]
    
    def forward(self, x, **kwargs):
        return self.model(x, head_mask=self.head_mask, **kwargs)
        
    def mask_neurons(self, layer_index: int, neurons: list):
        self.neuron_mask[layer_index][neurons] = 0

    def mask_heads(self, layer_index: int, neurons: list):
        self.head_mask[layer_index][neurons] = 0

    def unmask_neurons(self, layer_index: int, neurons: list):
        self.neuron_mask[layer_index][neurons] = 1

    def unmask_heads(self, layer_index: int, neurons: list):
        self.head_mask[layer_index][neurons] = 1
    
    
