import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from .initializations import kaiming_uniform


"""
    A modifiable version of Conv2D that can increase or decrease channel count and/or be masked
"""


class ModLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, masked: bool = False,
                 learnablemask: bool = False, nonlinearity: str = 'relu', prebatchnorm: bool = False):

        super().__init__(in_features, out_features, bias)

        self.masked = masked

        if masked:
            self.masktensor = Parameter(torch.ones(
                self.out_features, self.in_features), requires_grad=learnablemask)
            self.maskvector = Parameter(torch.ones(
                self.out_features), requires_grad=learnablemask)

        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == '':
            self.nonlinearity = nn.Identity()
        else:
            raise ValueError('Nonlinearity not supported')

        if prebatchnorm:
            self.batchnorm = nn.BatchNorm1d(self.in_features)
        else:
            self.batchnorm = nn.Identity()
    
    def weightparameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        else:
            return [self.weight]

    def forward(self, x: torch.Tensor):
        return self.nonlinearity(nn.functional.linear(self.batchnorm(x), self.masktensor * self.weight if self.masked else self.weight,
                                                      self.maskvector * self.bias if self.masked else self.bias))

    """
        Mask fanin weights of neurons of this layer that have indices in fanin and fanout weights 
        of neurons of the previous layer that have indices in fanout.

        fanin: list of indices of neurons of this layer
        fanout: list of indices of neurons of the previous layer
    """

    def mask(self, fanin=[], fanout=[]):
        if self.masked:
            self.masktensor.data[fanin, :] = 0
            self.maskvector.data[fanin] = 0
            self.masktensor.data[:, fanout] = 0
        else:
            print("No mask found")

    """
        Unmask fanin weights of neurons of this layer that have indices in fanin and fanout weights 
        of neurons of the previous layer that have indices in fanout.

        fanin: list of indices of neurons of this layer
        fanout: list of indices of neurons of the previous layer
    """

    def unmask(self, fanin=[], fanout=[]):
        if self.masked:
            self.masktensor.data[fanin, :] = 1
            self.maskvector.data[fanin] = 1
            self.masktensor.data[:, fanout] = 1
            if not isinstance(self.batchnorm, nn.Identity):
                if self.batchnorm.running_mean is not None:
                    self.batchnorm.running_mean[fanout] = 0
                    self.batchnorm.running_var[fanout] = 1
                if self.batchnorm.weight is not None:
                    self.batchnorm.weight.data[fanout] = 1
                    self.batchnorm.bias.data[fanout] = 0  
        else:
            print("No mask found")  

    """
        Remove fanin weights of neurons (of this layer) in list fanintoprune from the layer, and 
        fanout weights of neurons (of previous layer) in list fanouttoprune.

        fanintoprune: list of neurons to remove from this layer
        fanouttoprune: list of neurons to remove from previous layer
        optimizer: optimizer to update to new shape of the layer
    """

    def prune(self, fanintoprune=[], fanouttoprune=[], optimizer=None):
        fanintokeep = range(self.out_features)
        fanintokeep = [
            fitk for fitk in fanintokeep if fitk not in fanintoprune]

        fanouttokeep = range(self.in_features)
        fanouttokeep = [
            fotk for fotk in fanouttokeep if fotk not in fanouttoprune]

        if self.masked:
            self.masktensor.data = self.masktensor.data[fanintokeep,
                                                        :][:, fanouttokeep]
            self.maskvector.data = self.maskvector.data[fanintokeep]

        with torch.no_grad():
            newweight = Parameter(self.weight[fanintokeep, :][:, fanouttokeep])
            if self.bias is not None:
                newbias = Parameter(self.bias[fanintokeep])
            if not isinstance(self.batchnorm, nn.Identity) and self.batchnorm.weight is not None:
                newbnweight = Parameter(self.batchnorm.weight[fanouttokeep])
                newbnbias = Parameter(self.batchnorm.bias[fanouttokeep])

        if optimizer is not None:
            for group in optimizer.param_groups:
                for (i, param) in enumerate(group['params']):
                    if param is self.weight:
                        for (_, v) in optimizer.state[param].items():
                            if isinstance(v, torch.Tensor) and v.shape == self.weight.shape:
                                v.data = v.data[fanintokeep,
                                                :][:, fanouttokeep]
                        optimizer.state[newweight] = optimizer.state[param]
                        group['params'][i] = newweight
                    if self.bias is not None and param is self.bias:
                        for (_, v) in optimizer.state[param].items():
                            if isinstance(v, torch.Tensor) and v.shape == self.bias.shape:
                                v.data = v.data[fanintokeep]
                        optimizer.state[newbias] = optimizer.state[param]
                        group['params'][i] = newbias
                    if not isinstance(self.batchnorm, nn.Identity):
                        if self.batchnorm.weight is not None and param is self.batchnorm.weight:
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.batchnorm.weight.shape:
                                    v.data = v.data[fanouttokeep]
                            optimizer.state[newbnweight] = optimizer.state[param]
                            group['params'][i] = newbnweight
                        if self.batchnorm.bias is not None and param is self.batchnorm.bias:
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.batchnorm.bias.shape:
                                    v.data = v.data[fanouttokeep]
                            optimizer.state[newbnbias] = optimizer.state[param]
                            group['params'][i] = newbnbias

        self.weight = newweight
        if self.bias is not None:
            self.bias = newbias

        self.out_features = len(fanintokeep)
        self.in_features = len(fanouttokeep)

        if not isinstance(self.batchnorm, nn.Identity):
            if self.batchnorm.running_mean is not None:
                self.batchnorm.running_mean = self.batchnorm.running_mean[fanouttokeep]
                self.batchnorm.running_var = self.batchnorm.running_var[fanouttokeep]
            if self.batchnorm.weight is not None:
                self.batchnorm.weight = newbnweight
                self.batchnorm.bias = newbnbias

    """
        Add newoutfeatures new neurons to the layer (and newinfeatures new inputs to the layer), with 
        weights faninweights and fanoutweights respectively.

        If faninweights and/or fanoutweights are None, they are initialized with zeros.

        If faninweights and/or fanoutweights are 1D tensors, they are expanded to 2D tensors
        with the appropriate number of neurons/inputs.

        If faninweights and/or fanoutweights is "kaiming", they are initialized with the
        Kaiming initialization.

        newoutfeatures: number of neurons to add to this layer
        newinfeatures: number of inputs to add to this layer
        faninweights: weights of the new neurons
        fanoutweights: weights of the new inputs (adding neurons to the previous layer)
        optimizer: optimizer to update to new shape of the layer
    """

    def grow(self, newoutfeatures=0, newinfeatures=0, faninweights=None, fanoutweights=None, optimizer=None):
        if newinfeatures > 0:
            if fanoutweights is None:
                fanoutweights = torch.zeros(self.out_features, newinfeatures)
            elif fanoutweights == "kaiming":
                fanoutweights = kaiming_uniform(torch.zeros(self.out_features,self.in_features+newinfeatures))[:, :newinfeatures]
            elif len(fanoutweights.shape) == 1:
                fanoutweights = fanoutweights.unsqueeze(0)

            with torch.no_grad():
                newweight = Parameter(
                    torch.cat((self.weight.data, fanoutweights), dim=1))
                if not isinstance(self.batchnorm, nn.Identity) and self.batchnorm.weight is not None:
                    newbnweight = Parameter(
                        torch.cat((self.batchnorm.weight.data, torch.ones(newinfeatures)), dim=0))
                    newbnbias = Parameter(
                        torch.cat((self.batchnorm.bias.data, torch.zeros(newinfeatures)), dim=0))

            if optimizer is not None:
                for group in optimizer.param_groups:
                    for (i, param) in enumerate(group['params']):
                        if param is self.weight:
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.weight.shape:
                                    v.data = torch.cat(
                                        (v.data, torch.zeros_like(fanoutweights)), dim=1)
                            optimizer.state[newweight] = optimizer.state[param]
                            group['params'][i] = newweight
                        if not isinstance(self.batchnorm, nn.Identity):
                            if self.batchnorm.weight is not None and param is self.batchnorm.weight:
                                for (_, v) in optimizer.state[param].items():
                                    if isinstance(v, torch.Tensor) and v.shape == self.batchnorm.weight.shape:
                                        v.data = torch.cat(
                                            (v.data, torch.ones(newinfeatures)), dim=1)
                                optimizer.state[newbnweight] = optimizer.state[param]
                                group['params'][i] = newbnweight
                            if self.batchnorm.bias is not None and param is self.batchnorm.bias:
                                for (_, v) in optimizer.state[param].items():
                                    if isinstance(v, torch.Tensor) and v.shape == self.batchnorm.bias.shape:
                                        v.data = torch.cat(
                                            (v.data, torch.ones(newinfeatures)), dim=1)
                                optimizer.state[newbnbias] = optimizer.state[param]
                                group['params'][i] = newbnbias

            self.weight = newweight
            if self.masked:
                self.masktensor.data = torch.cat(
                    (self.masktensor.data, torch.ones(self.out_features, newinfeatures)), dim=1)

            self.in_features = self.in_features + newinfeatures

            if not isinstance(self.batchnorm, nn.Identity):
                if self.batchnorm.running_mean is not None:
                    self.batchnorm.running_mean = torch.cat(
                        (self.batchnorm.running_mean, torch.zeros(newinfeatures)))
                    self.batchnorm.running_var = torch.cat(
                        (self.batchnorm.running_var, torch.ones(newinfeatures)))
                if self.batchnorm.weight is not None:
                    self.batchnorm.weight = newbnweight
                    self.batchnorm.bias = newbnbias

        if newoutfeatures > 0:
            if faninweights is None:
                faninweights = torch.zeros(newoutfeatures, self.in_features)
            elif faninweights == "kaiming":
                faninweights = kaiming_uniform(torch.zeros(newoutfeatures+self.out_features, self.in_features))[:newoutfeatures, :]
            elif len(faninweights.shape) == 1:
                faninweights = faninweights.unsqueeze(1)

            with torch.no_grad():
                newweight = Parameter(
                    torch.cat((self.weight.data, faninweights), dim=0))
                if self.bias is not None:
                    newbias = Parameter(
                        torch.cat((self.bias.data, torch.zeros(newoutfeatures))))

            if optimizer is not None:
                for group in optimizer.param_groups:
                    for (i, param) in enumerate(group['params']):
                        if param is self.weight:
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.weight.shape:
                                    v.data = torch.cat(
                                        (v.data, torch.zeros_like(faninweights)), dim=0)
                            optimizer.state[newweight] = optimizer.state[param]
                            group['params'][i] = newweight
                        if self.bias is not None and param is self.bias:
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.bias.shape:
                                    v.data = torch.cat(
                                        (v.data, torch.zeros(newoutfeatures)))
                            optimizer.state[newbias] = optimizer.state[param]
                            group['params'][i] = newbias

            self.weight = newweight
            if self.bias is not None:
                self.bias = newbias
            if self.masked:
                self.masktensor.data = torch.cat(
                    (self.masktensor.data, torch.ones(newoutfeatures, self.in_features)), dim=0)
                self.maskvector.data = torch.cat(
                    (self.maskvector.data, torch.ones(newoutfeatures)))

            self.out_features = self.out_features + newoutfeatures


"""
    A modifiable version of Conv2D that can increase or decrease channel count and/or be masked
"""
class ModConv2d(nn.Conv2d):
    def __init__(self, masked: bool = False, bias: bool = True, learnablemask: bool = False, nonlinearity: str = 'relu',
                 prebatchnorm: bool = False, *args, **kwargs):

        super().__init__(bias=bias, *args, **kwargs)

        self.masked = masked
        self.learnablemask = learnablemask

        if masked:
            self.masktensor = Parameter(torch.ones(self.out_channels, self.in_channels,
                                        self.kernel_size[0], self.kernel_size[1]), requires_grad=learnablemask)
            self.maskvector = Parameter(torch.ones(
                self.out_channels), requires_grad=learnablemask)

        if nonlinearity == 'relu':
            self.nonlinearity = nn.ReLU()
        elif nonlinearity == '':
            self.nonlinearity = nn.Identity()
        else:
            raise ValueError('Nonlinearity not supported')

        if prebatchnorm:
            self.batchnorm = nn.BatchNorm2d(self.in_channels)
        else:
            self.batchnorm = nn.Identity()

    def weightparameters(self):
        if self.bias is not None:
            return [self.weight, self.bias]
        else:
            return [self.weight]
            
    def forward(self, x):
        return self.nonlinearity(nn.functional.conv2d(self.batchnorm(x), self.masktensor * self.weight if self.masked else self.weight,
                                                      self.maskvector * self.bias if self.masked else self.bias, self.stride, self.padding,
                                                      self.dilation, self.groups))

    """
        Mask fanin weights of neurons of this layer that have indices in fanin and fanout weights 
        of neurons of the previous layer that have indices in fanout.

        fanin: list of indices of neurons of this layer
        fanout: list of indices of neurons of the previous layer
    """

    def mask(self, fanin=[], fanout=[]):
        self.masktensor.data[fanin, :, :, :] = 0
        self.maskvector.data[fanin] = 0
        self.masktensor.data[:, fanout, :, :] = 0

    """
        Unmask fanin weights of channels of this layer that have indices in fanin and fanout weights 
        of channels of the previous layer that have indices in fanout.

        fanin: list of indices of channels of this layer
        fanout: list of indices of channels of the previous layer
    """

    def unmask(self, fanin=[], fanout=[]):
        self.masktensor.data[fanin, :, :, :] = 1
        self.maskvector.data[fanin] = 1
        self.masktensor.data[:, fanout, :, :] = 1
        if not isinstance(self.batchnorm, nn.Identity):
            if self.batchnorm.running_mean is not None:
                self.batchnorm.running_mean[fanout] = 0
                self.batchnorm.running_var[fanout] = 1
            if self.batchnorm.weight is not None:
                self.batchnorm.weight.data[fanout] = 1
                self.batchnorm.bias.data[fanout] = 0

    """
        Remove fanin weights of channels (of this layer) in list fanintoprune from the layer, and 
        fanout weights of channels (of previous layer) in list fanouttoprune.

        fanintoprune: list of channels to remove from this layer
        fanouttoprune: list of channels to remove from previous layer
    """

    def prune(self, fanintoprune=[], fanouttoprune=[], optimizer=None):
        fanintokeep = range(self.out_channels)
        fanintokeep = [
            fitk for fitk in fanintokeep if fitk not in fanintoprune]

        fanouttokeep = range(self.in_channels)
        fanouttokeep = [
            fotk for fotk in fanouttokeep if fotk not in fanouttoprune]

        with torch.no_grad():
            newweight = Parameter(self.weight[fanintokeep, :][:, fanouttokeep])
            if self.bias is not None:
                newbias = Parameter(self.bias[fanintokeep])
            if not isinstance(self.batchnorm, nn.Identity) and self.batchnorm.weight is not None:
                newbnweight = Parameter(self.batchnorm.weight[fanouttokeep])
                newbnbias = Parameter(self.batchnorm.bias[fanouttokeep])


        if optimizer is not None:
            for group in optimizer.param_groups:
                for (i, param) in enumerate(group['params']):
                    if param is self.weight:
                        for (_, v) in optimizer.state[param].items():
                            if isinstance(v, torch.Tensor) and v.shape == self.weight.shape:
                                v.data = v.data[fanintokeep,
                                                :][:, fanouttokeep]
                        optimizer.state[newweight] = optimizer.state[param]
                        group['params'][i] = newweight
                    if self.bias is not None and param is self.bias:
                        for (_, v) in optimizer.state[param].items():
                            if isinstance(v, torch.Tensor) and v.shape == self.bias.shape:
                                v.data = v.data[fanintokeep]
                        optimizer.state[newbias] = optimizer.state[param]
                        group['params'][i] = newbias
                    if not isinstance(self.batchnorm, nn.Identity):
                        if self.batchnorm.weight is not None and param is self.batchnorm.weight:
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.batchnorm.weight.shape:
                                    v.data = v.data[fanouttokeep]
                            optimizer.state[newbnweight] = optimizer.state[param]
                            group['params'][i] = newbnweight
                        if self.batchnorm.bias is not None and param is self.batchnorm.bias:
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.batchnorm.bias.shape:
                                    v.data = v.data[fanouttokeep]
                            optimizer.state[newbnbias] = optimizer.state[param]
                            group['params'][i] = newbnbias

        self.weight = newweight
        if self.bias is not None:
            self.bias = newbias

        if self.masked:
            self.masktensor.data = self.masktensor.data[fanintokeep,
                                                        :][:, fanouttokeep]
            self.maskvector.data = self.maskvector.data[fanintokeep]

        self.out_channels = len(fanintokeep)
        self.in_channels = len(fanouttokeep)

        if not isinstance(self.batchnorm, nn.Identity):
            if self.batchnorm.running_mean is not None:
                self.batchnorm.running_mean = self.batchnorm.running_mean[fanouttokeep]
                self.batchnorm.running_var = self.batchnorm.running_var[fanouttokeep]
            if self.batchnorm.weight is not None:
                self.batchnorm.weight = newbnweight
                self.batchnorm.bias = newbnbias

    """
        Add newoutchannels new channels to the layer (and newinchannels new inputs to the layer), with 
        weights faninweights and fanoutweights respectively.

        If faninweights and/or fanoutweights are None, they are initialized with zeros.

        If faninweights and/or fanoutweights are 1D tensors, they are reshaped to 4D tensors
        with the appropriate number of channels/inputs.

        newoutchannels: number of channels to add to this layer
        newinchannels: number of inputs to add to this layer
        faninweights: weights of the new channels
        fanoutweights: weights of the new inputs (channels of previous layer)
    """

    def grow(self, newoutchannels=0, newinchannels=0, faninweights=None, fanoutweights=None, optimizer=None):
        if newinchannels > 0:
            if fanoutweights is None:
                fanoutweights = torch.zeros(
                    self.out_channels, newinchannels, self.kernel_size[0], self.kernel_size[1])
            elif fanoutweights == "kaiming":
                fanoutweights = kaiming_uniform(torch.zeros(self.out_channels,self.in_channels+newinchannels, self.kernel_size[0], self.kernel_size[1]))[:, :newinchannels]
            elif len(fanoutweights.shape) == 1:
                fanoutweights = torch.reshape(
                    fanoutweights, (self.out_channels, newinchannels, self.kernel_size[0], self.kernel_size[1]))
            elif len(fanoutweights.shape) == 3:
                fanoutweights = fanoutweights.unsqueeze(0)

            with torch.no_grad():
                newweight = Parameter(
                    torch.cat((self.weight.data, fanoutweights), dim=1))
                if not isinstance(self.batchnorm, nn.Identity) and self.batchnorm.weight is not None:
                    newbnweight = Parameter(
                        torch.cat((self.batchnorm.weight.data, torch.ones(newinchannels)), dim=0))
                    newbnbias = Parameter(
                        torch.cat((self.batchnorm.bias.data, torch.zeros(newinchannels)), dim=0))


            if optimizer is not None:
                for group in optimizer.param_groups:
                    for (i, param) in enumerate(group['params']):
                        if param is self.weight:  # note: p will automatically be updated in optimizer.param_groups
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.weight.shape:
                                    v.data = torch.cat(
                                        (v.data, torch.zeros_like(fanoutweights)), dim=1)
                            optimizer.state[newweight] = optimizer.state[param]
                            group['params'][i] = newweight
                        if not isinstance(self.batchnorm, nn.Identity):
                            if self.batchnorm.weight is not None and param is self.batchnorm.weight:
                                for (_, v) in optimizer.state[param].items():
                                    if isinstance(v, torch.Tensor) and v.shape == self.batchnorm.weight.shape:
                                        v.data = torch.cat(
                                            (v.data, torch.ones(newinchannels)), dim=1)
                                optimizer.state[newbnweight] = optimizer.state[param]
                                group['params'][i] = newbnweight
                            if self.batchnorm.bias is not None and param is self.batchnorm.bias:
                                for (_, v) in optimizer.state[param].items():
                                    if isinstance(v, torch.Tensor) and v.shape == self.batchnorm.bias.shape:
                                        v.data = torch.cat(
                                            (v.data, torch.ones(newinchannels)), dim=1)
                                optimizer.state[newbnbias] = optimizer.state[param]
                                group['params'][i] = newbnbias


            self.weight = newweight
            if self.masked:
                self.masktensor.data = torch.cat((self.masktensor, torch.ones(
                    self.out_channels, newinchannels, self.kernel_size[0], self.kernel_size[1])), dim=1)

            self.in_channels = self.in_channels + newinchannels

            if not isinstance(self.batchnorm, nn.Identity):
                if self.batchnorm.running_mean is not None:
                    self.batchnorm.running_mean = torch.cat(
                        (self.batchnorm.running_mean, torch.zeros(newinchannels)))
                    self.batchnorm.running_var = torch.cat(
                        (self.batchnorm.running_var, torch.ones(newinchannels)))
                if self.batchnorm.weight is not None:
                    self.batchnorm.weight = newbnweight
                    self.batchnorm.bias = newbnbias


        if newoutchannels > 0:
            if faninweights is None:
                faninweights = torch.zeros(
                    newoutchannels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
            elif faninweights == "kaiming":
                faninweights = kaiming_uniform(torch.zeros(newoutchannels+self.out_channels, self.in_channels, 
                                                           self.kernel_size[0], self.kernel_size[1]))[:newoutchannels]
            elif len(faninweights.shape) == 1:
                faninweights = torch.reshape(
                    faninweights, (newoutchannels, self.in_channels, self.kernel_size[0], self.kernel_size[1]))
            elif len(faninweights.shape) == 3:
                faninweights = faninweights.unsqueeze(1)

            newweight = nn.Parameter(
                torch.cat((self.weight.data, faninweights), dim=0))
            if self.bias is not None:
                newbias = nn.Parameter(
                    torch.cat((self.bias.data, torch.zeros(newoutchannels)), dim=0))

            if optimizer is not None:
                for group in optimizer.param_groups:
                    for (i, param) in enumerate(group['params']):
                        if param is self.weight:
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.weight.shape:
                                    v.data = torch.cat(
                                        (v.data, torch.zeros_like(faninweights)), dim=0)
                            optimizer.state[newweight] = optimizer.state[param]
                            group['params'][i] = newweight
                        if self.bias is not None and param is self.bias:
                            for (_, v) in optimizer.state[param].items():
                                if isinstance(v, torch.Tensor) and v.shape == self.bias.shape:
                                    v.data = torch.cat(
                                        (v.data, torch.zeros(newoutchannels)))
                            optimizer.state[newbias] = optimizer.state[param]
                            group['params'][i] = newbias

            self.weight = newweight
            if self.bias is not None:
                self.bias = newbias
            if self.masked:
                self.masktensor.data = torch.cat((self.masktensor, torch.ones(
                    newoutchannels, self.in_channels, self.kernel_size[0], self.kernel_size[1])), dim=0)
                self.maskvector.data = torch.cat(
                    (self.maskvector, torch.ones(newoutchannels)))

            self.out_channels = self.out_channels + newoutchannels
