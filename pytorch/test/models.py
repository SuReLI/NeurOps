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

        model.unmask(1, [1], clear_activations=True)
        self.assertTrue(len(model.activations['1'].shape) < 2)
        y = model(x)
        self.assertEqual(y.shape, (10, 5))

        model.prune(2, [0], clear_activations=True)
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
        

    def test_auxiliary(self):
        layer1 = ModConv2d(in_channels=6, out_channels=2, kernel_size=3, masked=True)
        layer2 = ModLinear(2, 3, masked=True, preflatten=True)  
        model = ModSequential(layer1, layer2, track_activations=True,
                              track_auxiliary_gradients=True)
        loss = torch.nn.CrossEntropyLoss()
        ytrue = torch.randn(10, 3)
        x = torch.randn(10, 6, 3, 3)
        y = model(x, model.auxiliaries)
        loss = loss(y, ytrue)
        loss.backward()
        self.assertEqual(y.shape, (10, 3)) 
        self.assertEqual(len(model.auxiliaries), 1)
        self.assertFalse(torch.allclose(model.auxiliaries[0].grad, torch.zeros(3, 6, 3, 3)))

        config = transformers.BertConfig()
        model = transformers.BertForSequenceClassification(config)
        modmodel = ModTransformer(model, track_activations=True,
                                  track_auxiliary_gradients=True)
        x = torch.randint(0, 100, (10, 20))
        y = modmodel(x, labels=torch.randint(0, 2, (10,)))
        y.loss.backward()
        self.assertEqual(len(modmodel.auxiliaries), 12)
        self.assertFalse(torch.allclose(modmodel.auxiliaries[0].grad, torch.zeros_like(modmodel.auxiliaries[0].grad)))
        





if __name__ == '__main__':
    unittest.main()
