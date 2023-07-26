import torch
import torch.nn as nn

from functools import partial
from collections import defaultdict
from math import prod

from .layers import ModLinear, ModConv2d


"""
A modifiable sequential container that allows for masking, pruning, and growing of layers.

track_activations: don't track activations if false or 0, else keep input & activation buffer with size 
                   equal to track_activations (2 if true) times dimensionality of layer
track_auxiliary_gradients: don't track auxiliary gradients if false or 0, else compute auxiliary matrices for aux gradients
input_shape: shape of input to model, used to compute conversion factor for linear layers
"""
class ModSequential(nn.Sequential):
    def __init__(self,  *args, track_activations: False, track_auxiliary_gradients: bool = False, input_shape: tuple = None):
        
        super(ModSequential, self).__init__(*args)

        self.test = False

        self.track_activations = track_activations
        self.track_auxiliary_gradients = track_auxiliary_gradients

        if self.track_activations:
            self.multiplier = 2 if isinstance(self.track_activations, bool) else self.track_activations
            self.activations = defaultdict(torch.Tensor)
            for name, module in self.named_modules():
                if isinstance(module, ModLinear) or isinstance(module, ModConv2d):
                    module.register_forward_hook(partial(self._act_hook, name))
            self[0].register_forward_pre_hook(self._input_hook)

        self.conversion_layer = -1
        self.conversion_factor = 1
        self.input_shape = input_shape
        
        for i in range(len(self)-1):
            if isinstance(self[i+1], nn.Linear) and isinstance(self[i], nn.Conv2d):
                self.conversion_layer = i
                self[i+1].preflatten = nn.Flatten(start_dim=1)
                if input_shape is not None:
                    self.conversion_factor = prod(self(torch.zeros(1, *input_shape), layer_index=i).shape[2:])  
                else:
                    self.conversion_factor = self[i+1].in_features // self[i].out_channels

        if self.track_auxiliary_gradients:
            self.auxiliaries = []
            for i in range(1, len(self)):
                if isinstance(self[i-1], nn.Linear):
                    aux = torch.zeros(self[i].out_features, self[i-1].in_features)
                elif isinstance(self[i], nn.Conv2d):
                    aux = torch.zeros(self[i].out_channels, self[i-1].in_channels, 
                                      self[i-1].kernel_size[0]+self[i].kernel_size[0]-1, 
                                      self[i-1].kernel_size[1]+self[i].kernel_size[1]-1)
                else:
                    aux = torch.zeros(self[i].out_features // self.conversion_factor, self[i-1].in_channels, 
                                      self[i-1].kernel_size[0], self[i-1].kernel_size[1])
                self.auxiliaries.append(nn.parameter.Parameter(aux))

    """
    Redefines `parameters` to exclude the auxiliary matrices and optionally the masks.
    """
    def parameters(self, recurse: bool = True, include_mask = False):
        return (p for name, p in self.named_parameters(recurse=recurse) if name != 'auxiliaries' and ("mask" not in name or include_mask))

    """
    Saves the activations of a layer to the activations dictionary.
    """
    def _act_hook(self, name, module, input, output):
        if not self.test:
            self.activations[name] = torch.cat((self.activations[name], output.cpu()), dim=0)
            if self.activations[name].shape[0] > self.multiplier*self.activations[name].shape[1]:
                self.activations[name] = self.activations[name][-self.multiplier*self.activations[name].shape[1]:]
    
    """
    Saves the input to the first layer to the activations dictionary.
    """
    def _input_hook(self, module, input):
        if not self.test:
            self.activations["-1"] = torch.cat((self.activations["-1"], input[0].cpu()), dim=0)
            if self.activations["-1"].shape[0] > self.multiplier*self.activations["-1"].shape[1]:
                self.activations["-1"] = self.activations["-1"][-self.multiplier*self.activations["-1"].shape[1]:]
    
    def _act_shape_hook(self, module, input, output):
        self.conv_output_shape = output.shape[1:]

    """
    Returns the number of "effective" parameters in the model, optionally excluding the masks.
    """
    def parameter_count(self, masked: bool = False):
        count = 0
        for i in range(len(self)):
            if isinstance(self[i], ModLinear) or isinstance(self[i], ModConv2d):
                count += self[i].parameter_count(masked=masked, previous_mask = None if i == 0 or not self[i-1].masked else 
                                                 self[i-1].mask_vector if i-1 != self.conversion_layer else 
                                                 self[i-1].mask_vector.view(1,-1).tile(self.conversion_factor,1).view(-1))
            else:
                count += sum(p.numel() for p in self[i].parameters())
        return count
    
    def FLOPs_count(self, input = None, masked: bool = False, verbose: bool = False):
        count = 0
        x = torch.zeros_like(input)
        for i in range(len(self)):
            if isinstance(self[i], ModLinear) or isinstance(self[i], ModConv2d):
                FLOPs, x = self[i].FLOPs_count(x, masked=masked, previous_mask = None if i == 0 or not self[i-1].masked else 
                                                 self[i-1].mask_vector if i-1 != self.conversion_layer else 
                                                 self[i-1].mask_vector.view(1,-1).tile(self.conversion_factor,1).view(-1))
                count += FLOPs
                if verbose:
                    print(f"Layer {i}: {FLOPs} FLOPs")
            else:
                x = self[i](x)
        return count

    def clear_activations(self):
        self.activations = defaultdict(torch.Tensor)

    def forward(self, x, auxiliaries: list = None, layer_index: int = -1):
        old_x = x
        for i, module in enumerate(self):
            if i == 0 or auxiliaries is None:
                x = module(x)
            else:
                x_copy = x
                x = module(x, auxiliaries[i-1], old_x, self[i-1])
                old_x = x_copy * self[i-1].mask_vector.view(1, -1, *[1 for dim in x_copy.shape[2:]])
            if i == layer_index:
                return x
        return x

    """
    Masks neurons of a given layer, optionally clearing the activations of the layer and the following layer.
    """
    def mask(self, layer_index: int, neurons: list = [], clear_activations: bool = False):
        for i, module in enumerate(self):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.mask(neurons)
        if clear_activations and self.track_activations:
            for index in (layer_index+1, layer_index):
                self.activations[str(index)] = torch.Tensor().to(self.activations[str(index)].device)

    """
    Unmasks neurons of a given layer, optionally updating an optimizer and/or clearing the activations of the 
    layer and the following layer.
    """
    def unmask(self, layer_index: int, neurons: list = [], optimizer=None, clear_activations: bool = False):
        for i, module in enumerate(self):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.unmask(neurons, [], optimizer=optimizer)
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                if layer_index == self.conversion_layer:
                    converted_neurons = sum([[*range(neuron*self.conversion_factor, (neuron+1)*self.conversion_factor)] for neuron in neurons], [])
                else:
                    converted_neurons = neurons
                module.unmask([], converted_neurons, optimizer=optimizer)
        if clear_activations and self.track_activations:
            for index in (layer_index+1, layer_index):
                self.activations[str(index)] = torch.Tensor().to(self.activations[str(index)].device)
            
    """
    Prunes neurons of a given layer, optionally updating an optimizer and/or clearing the activations of the
    layer and the following layer.
    """
    def prune(self, layer_index: int, neurons: list = [], optimizer=None, clear_activations: bool = False):
        for i, module in enumerate(self):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.prune(neurons, [], optimizer=optimizer)
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                if i-1 == self.conversion_layer:
                    converted_neurons = sum([[*range(neuron*self.conversion_factor, (neuron+1)*self.conversion_factor)] for neuron in neurons], [])
                else:
                    converted_neurons = neurons
                module.prune([], converted_neurons, optimizer=optimizer)
        for index in (layer_index+1, layer_index):
            if clear_activations and self.track_activations:
                self.activations[str(index)] = torch.Tensor().to(self.activations[str(index)].device)
            elif index == layer_index and self.track_activations and len(self.activations[str(index)].shape) >= 2:
                neurons_to_keep = range(self.activations[str(index)].shape[1])
                neurons_to_keep = [
                    ntk for ntk in neurons_to_keep if ntk not in neurons]
                self.activations[str(index)] = self.activations[str(index)][:, neurons_to_keep]
        if self.track_auxiliary_gradients:
            neurons_to_keep = range(self.auxiliaries[layer_index-1].shape[1])
            neurons_to_keep = [
                ntk for ntk in neurons if ntk not in neurons]
            self.auxiliaries[layer_index-1] = self.auxiliaries[layer_index-1][:, neurons_to_keep]
            self.auxiliaries[layer_index-2] = self.auxiliaries[layer_index-2][neurons_to_keep]
                
    """
    Grows a given layer by a given number of neurons, optionally updating an optimizer and/or clearing the activations of the
    layer and the following layer.
    """
    def grow(self, layer_index: int, newneurons: int = 0, fanin_weights=None, fanout_weights=None, 
             optimizer=None, clear_activations: bool = False, send_activations: bool = False):
        for i, module in enumerate(self):
            if i == layer_index and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                module.grow(newneurons, 0, fanin_weights = fanin_weights, optimizer=optimizer, 
                            activations=self.activations[str(layer_index-1)] if send_activations or fanin_weights == "iterative_orthogonalization" else None)
                if self.track_activations:
                    if clear_activations:
                        self.activations[str(layer_index)] = torch.Tensor().to(self.activations[str(layer_index)].device)
                    else:
                        self.activations[str(layer_index)] = torch.cat(
                            (self.activations[str(layer_index)], 
                             torch.zeros(self.activations[str(layer_index)].shape[0], newneurons, *self.activations[str(layer_index)].shape[2:]).to(self.activations[str(layer_index)].device)), dim=1)
            elif i == layer_index+1 and (isinstance(module, ModLinear) or isinstance(module, ModConv2d)):
                if layer_index == self.conversion_layer:
                    converted_newneurons = newneurons*self.conversion_factor
                else:
                    converted_newneurons = newneurons
                module.grow(0, converted_newneurons, fanout_weights = fanout_weights, optimizer=optimizer, 
                            activations=self.activations[str(layer_index)] if send_activations or fanout_weights == "iterative_orthogonalization" else None)
                if clear_activations and self.track_activations:
                    self.activations[str(layer_index+1)] = torch.Tensor().to(self.activations[str(layer_index+1)].device)
        if self.track_auxiliary_gradients:
            self.auxiliaries[layer_index-1] = torch.cat(
                (self.auxiliaries[layer_index-1], torch.zeros(self.auxiliaries[layer_index-1].shape[0], newneurons, *self.auxiliaries[layer_index-1].shape[2:])), dim=1)
            self.auxiliaries[layer_index-2] = torch.cat(
                (self.auxiliaries[layer_index-2], torch.zeros(newneurons, *self.auxiliaries[layer_index-2].shape[1:])), dim=0)
           

"""
A wrapper for the HuggingFace Transformer model that allows for masking of attention heads and/or of 
hidden neurons in the FFN layer of each transformer block, via hooks on top of the existing model class.
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
    
    
