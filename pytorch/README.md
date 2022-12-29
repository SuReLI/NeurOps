This directory contains the PyTorch implementation of NeurOps. 

Source code is found in the 'src' directory:
-'layers.py': `ModLinear` and `ModConv2d` respectfully extend standard `nn.Linear` and `nn.Conv2d` to also enable masking, growing, and pruning, including auxiliary gradient matrix calculation, activation tracking, pre/post layer operations such as nonlinearities and layer normalization. 
-'models.py': `ModSequential` extends standard `nn.Sequential` to enable these neural operations to a sequence of `ModConv2d` and `ModLinear` layers, including the transition from convolutional to dense layers. `ModTransformer` extends architectures from `transformers` to also enable masking of attention heads and/or hidden neurons in each block.
-'metrics.py': Many heuristics useful for growing and pruning that utilize weight, activation, and gradient information.
-'initializations.py': Initializations for new layers and growing new neurons, including random initialization (Kaimeng) with proper scale with the existing layer size as well as orthogonalized initialization.
