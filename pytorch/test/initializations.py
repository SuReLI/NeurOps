import unittest
import torch
import sys
sys.path.append('.')
from pytorch.neurops.layers import *
from pytorch.neurops.models import *
from pytorch.neurops.initializations import *

class TestInitializations(unittest.TestCase):
    def test_iterative_orthogonalization(self):
        model = ModSequential(
            ModLinear(5, 6, masked=True),
            ModLinear(6, 4, masked=True),
            ModLinear(4, 4, masked=True),
            track_activations=True
        )
        x = torch.randn(10, 5)
        H = x
        for layer in model:
            original = layer.weight.data
            layer.weight.data = iterative_orthogonalization(layer.weight.data, H)
            self.assertFalse(torch.allclose(original, layer.weight.data))
            H = layer(H)
        original1 = model[1].weight.data
        original2 = model[2].weight.data
        model.grow(1,2,"iterative_orthogonalization","iterative_orthogonalization", send_activations = True)
        self.assertTrue(model[1].weight.size(0) == 6)
        self.assertTrue(torch.allclose(original1, model[1].weight.data[:4]))
        self.assertFalse(torch.allclose(model[1].weight.data[4:], torch.zeros([2,6])))
        self.assertTrue(model[2].weight.size(1) == 6)
        self.assertTrue(torch.allclose(original2, model[2].weight.data[:,:4]))
        self.assertFalse(torch.allclose(model[2].weight.data[:,4:], torch.zeros([4,2])))

        model = ModSequential(
            ModConv2d(in_channels=1, out_channels=8, kernel_size=3, masked=True),
            ModConv2d(in_channels=8, out_channels=1, kernel_size=3, masked=True),
            track_activations=True
        )
        x = torch.randn(10, 1, 5, 5)
        H = x
        for layer in model:
            original = layer.weight.data
            layer.weight.data = iterative_orthogonalization(layer.weight.data, H)
            self.assertFalse(torch.allclose(original, layer.weight.data))
            H = layer(H)
        original1 = model[0].weight.data
        original2 = model[1].weight.data
        model.grow(0,2,"iterative_orthogonalization", None, send_activations = True)
        self.assertTrue(model[0].weight.size(0) == 10)
        self.assertTrue(torch.allclose(original1, model[0].weight.data[:8]))
        self.assertFalse(torch.allclose(model[0].weight.data[8:], torch.zeros([2,8,3,3])))
        self.assertTrue(model[1].weight.size(1) == 10)
        self.assertTrue(torch.allclose(original2, model[1].weight.data[:,:8]))
        self.assertTrue(torch.allclose(model[1].weight.data[:,8:], torch.zeros([1,2,3,3])))

        model = ModSequential(
            ModConv2d(in_channels=1, out_channels=16, kernel_size=7, masked=True, padding=1, prebatchnorm=True, learnable_mask=True),
            ModConv2d(in_channels=16, out_channels=16, kernel_size=5, masked=True, prebatchnorm=True, learnable_mask=True),
            ModLinear(64, 10, masked=True, learnable_mask=True),
            track_activations=True,
            track_auxiliary_gradients=True,
            input_shape = (1, 10, 10)
        )
        x = torch.randn(10, 1, 10, 10)
        out = model(x)
        H = x
        for layer in model:
            original = layer.weight.data
            layer.weight.data = iterative_orthogonalization(layer.weight.data, H)
            self.assertFalse(torch.allclose(original, layer.weight.data))
            H = layer(H)
        original1 = model[1].weight.data
        original2 = model[2].weight.data
        model.grow(1,2,"iterative_orthogonalization","iterative_orthogonalization", send_activations = True)
        self.assertTrue(model[1].weight.size(0) == 18)
        self.assertTrue(torch.allclose(original1, model[1].weight.data[:16]))
        self.assertFalse(torch.allclose(model[1].weight.data[16:], torch.zeros([2,16,5,5])))
        self.assertTrue(model[2].weight.size(1) == 72)
        self.assertTrue(torch.allclose(original2, model[2].weight.data[:,:64]))
        self.assertFalse(torch.allclose(model[2].weight.data[:,64:], torch.zeros([10,8])))
        self.assertTrue(model[2].weight.size(0) == 10)

        model = ModSequential(
            ModConv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, 
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            ModConv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            ModConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            ModConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,
                      postpool=torch.nn.Sequential(torch.nn.MaxPool2d(2,2), 
                                                   torch.nn.AdaptiveAvgPool2d((2, 2)))), 
            ModLinear(128, 128, preflatten=True),
            ModLinear(128, 2, nonlinearity=''),
            track_activations=2,
            input_shape = (3, 64, 64)
        )
        x = torch.randn(128, 3, 64, 64)
        out = model(x)
        H = x
        for layer in model:
            original = layer.weight.data
            layer.weight.data = iterative_orthogonalization(layer.weight.data, H)
            self.assertFalse(torch.allclose(original, layer.weight.data))
            H = layer(H)
        for i, layer in enumerate(model):
            model.grow(i,2,"iterative_orthogonalization","iterative_orthogonalization", send_activations = True)


    def test_kaiming_uniform(self):
        model = ModSequential(
            ModLinear(16, 128, masked=True),
            ModLinear(128, 128, masked=True),
            ModLinear(128, 16, masked=True),
            track_activations=True
        )
        for layer in model:
            original = layer.weight.data.clone()
            layer.weight.data = kaiming_uniform(layer.weight.data)
            self.assertFalse(torch.allclose(original, layer.weight.data))
            self.assertTrue(torch.allclose(original.std(), layer.weight.data.std(), atol=1e-1, rtol=1e-1))
        original1 = model[1].weight.data
        original2 = model[2].weight.data
        model.grow(1,2,"kaiming","kaiming")
        self.assertTrue(model[1].weight.size(0) == 130)
        self.assertTrue(torch.allclose(original1, model[1].weight.data[:-2]))
        self.assertFalse(torch.allclose(model[1].weight.data[-2:], torch.zeros([2,128])))
        self.assertTrue(model[2].weight.size(1) == 130)
        self.assertTrue(torch.allclose(original2, model[2].weight.data[:,:-2]))
        self.assertFalse(torch.allclose(model[2].weight.data[:,-2:], torch.zeros([16,2])))

    def test_autoinit(self):
        model = ModSequential(
            ModLinear(16, 128, masked=True),
            ModLinear(128, 128, masked=True),
            ModLinear(128, 16, masked=True, nonlinearity=""),
            track_activations=True
        )
        x = torch.randn(2048, 16)
        out = model(x)
        H = x
        for layer in model:
            original = layer.weight.data.clone()
            layer.weight.data = autoinit(layer.weight.data, H)
            self.assertFalse(torch.allclose(original, layer.weight.data))
            output = torch.nn.functional.linear(H, layer.weight, layer.bias)
            orig_output = torch.nn.functional.linear(H, original, layer.bias)
            H = layer(H)
        original1 = model[1].weight.data
        original2 = model[2].weight.data
        model.grow(1, 2, "autoinit", None, send_activations = True)
        self.assertTrue(model[1].weight.size(0) == 130)
        self.assertTrue(torch.allclose(original1, model[1].weight.data[:-2]))
        self.assertFalse(torch.allclose(model[1].weight.data[-2:], torch.zeros([2,128])))
        self.assertTrue(model[2].weight.size(1) == 130)
        self.assertTrue(torch.allclose(original2, model[2].weight.data[:,:-2]))
        self.assertTrue(torch.allclose(model[2].weight.data[:,-2:], torch.zeros([16,2])))

    def test_northselect(self):
        model = ModSequential(
            ModLinear(16, 128, masked=True),
            ModLinear(128, 128, masked=True),
            ModLinear(128, 16, masked=True, nonlinearity=""),
            track_activations=True
        )
        x = torch.randn(2048, 16)
        out = model(x)
        original1 = model[1].weight.data
        original2 = model[2].weight.data
        model.grow(1, 2, "north_select", None, send_activations = True)
        self.assertTrue(model[1].weight.size(0) == 130)
        self.assertTrue(torch.allclose(original1, model[1].weight.data[:-2]))
        self.assertFalse(torch.allclose(model[1].weight.data[-2:], torch.zeros([2,128])))
        self.assertTrue(model[2].weight.size(1) == 130)
        self.assertTrue(torch.allclose(original2, model[2].weight.data[:,:-2]))
        self.assertTrue(torch.allclose(model[2].weight.data[:,-2:], torch.zeros([16,2])))

        model = ModSequential(
            ModConv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1, 
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            ModConv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            ModConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,
                      postpool=torch.nn.MaxPool2d(2,2)),
            ModConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            ModConv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1,
                      postpool=torch.nn.Sequential(torch.nn.MaxPool2d(2,2), 
                                                   torch.nn.AdaptiveAvgPool2d((2, 2)))), 
            ModLinear(128, 128, preflatten=True),
            ModLinear(128, 2, nonlinearity=''),
            track_activations=2,
            input_shape = (3, 64, 64)
        )

        x = torch.randn(2048, 3, 64, 64)
        out = model(x)
        original1 = model[1].weight.data
        original2 = model[2].weight.data
        model.grow(1, 2, "north_select", None, send_activations = True)
        self.assertTrue(model[1].weight.size(0) == 18)
        self.assertTrue(torch.allclose(original1, model[1].weight.data[:-2]))
        self.assertFalse(torch.allclose(model[1].weight.data[-2:], torch.zeros([2,8,3,3])))
        self.assertTrue(model[2].weight.size(1) == 18)
        self.assertTrue(torch.allclose(original2, model[2].weight.data[:,:-2]))
        self.assertTrue(torch.allclose(model[2].weight.data[:,-2:], torch.zeros([16,2,3,3])))
        original8 = model[8].weight.data
        original9 = model[9].weight.data
        model.grow(8, 2, "north_select", None, send_activations = True)
        self.assertTrue(model[8].weight.size(0) == 130)
        self.assertTrue(torch.allclose(original8, model[8].weight.data[:-2]))
        self.assertFalse(torch.allclose(model[8].weight.data[-2:], torch.zeros([2,128])))
        self.assertTrue(model[9].weight.size(1) == 130)
        self.assertTrue(torch.allclose(original9, model[9].weight.data[:,:-2]))
        self.assertTrue(torch.allclose(model[9].weight.data[:,-2:], torch.zeros([2,2])))
        


if __name__ == '__main__':
    unittest.main()
