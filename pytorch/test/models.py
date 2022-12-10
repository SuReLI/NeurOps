import unittest
import torch
import sys
sys.path.append('.')
from pytorch.src.models import *
from pytorch.src.layers import *


class TestModels(unittest.TestCase):
    def test_modsequential(self):
        model = ModSequential(
            ModLinear(2, 3, masked=True),
            ModLinear(3, 4, masked=True),
            ModLinear(4, 5, masked=True),
        )
        x = torch.randn(10, 2)
        y = model(x)
        self.assertEqual(y.shape, (10, 5))
        self.assertEqual(model.activations['0'].shape[1], 3)
        self.assertEqual(model.activations['1'].shape[1], 4)

        model.mask(1, [1])
        y = model(x)
        self.assertEqual(y.shape, (10, 5))
        self.assertEqual(model.activations['1'][-1,1], 0)

        model.unmask(1, [1], clearacts=True)
        self.assertTrue(len(model.activations['1'].shape) < 2)
        y = model(x)
        self.assertEqual(y.shape, (10, 5))

        model.prune(2, [0], clearacts=True)
        y = model(x)
        self.assertEqual(y.shape, (10, 4))
        
        model.grow(1, 2)
        y = model(x)
        self.assertEqual(model.activations['1'].shape[1], 6)
        




if __name__ == '__main__':
    unittest.main()
