import unittest
import torch
import transformers
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
            track_activations=True
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

        model.unmask(1, [1], clear_acts=True)
        self.assertTrue(len(model.activations['1'].shape) < 2)
        y = model(x)
        self.assertEqual(y.shape, (10, 5))

        model.prune(2, [0], clear_acts=True)
        y = model(x)
        self.assertEqual(y.shape, (10, 4))
        
        model.grow(1, 2)
        y = model(x)
        self.assertEqual(model.activations['1'].shape[1], 6)

    def test_modtransformer(self):
        config = transformers.BertConfig()
        model = transformers.BertForSequenceClassification(config)
        modmodel = ModTransformer(model, track_activations=True)
        x = torch.randint(0, 100, (10, 20))
        _ = modmodel(x)
        self.assertEqual(modmodel.neuron_activations[2].shape[2], config.intermediate_size)
        self.assertEqual(modmodel.head_activations[3].shape[1], config.num_attention_heads)
        for layer in range(12):
            modmodel.mask_heads(layer, range(2))
            modmodel.mask_neurons(layer, range(128))
        _ = modmodel(x)
        self.assertTrue(torch.allclose(modmodel.head_activations[2][10:20,:2,0,0], torch.zeros(10,2)))
        #print(modmodel.neuron_activations[2][10:20,0,:10])
        self.assertTrue(torch.allclose(modmodel.neuron_activations[2][10:20,:,:128], torch.zeros(10,20,128)))
        

        




if __name__ == '__main__':
    unittest.main()
