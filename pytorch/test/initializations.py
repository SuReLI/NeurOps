import unittest
import torch
import sys
sys.path.append('.')
from pytorch.src.layers import *
from pytorch.src.models import *
from pytorch.src.initializations import *

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
        model.grow(1,2,"iterative_orthogonalization","iterative_orthogonalization", sendactivations = True)
        self.assertTrue(model[1].weight.size(0) == 6)
        self.assertTrue(torch.allclose(original1, model[1].weight.data[:4]))
        self.assertFalse(torch.allclose(model[1].weight.data[4:], torch.zeros([2,6])))
        self.assertTrue(model[2].weight.size(1) == 6)
        self.assertTrue(torch.allclose(original2, model[2].weight.data[:,:4]))
        self.assertFalse(torch.allclose(model[2].weight.data[:,4:], torch.zeros([4,2])))




if __name__ == '__main__':
    unittest.main()
