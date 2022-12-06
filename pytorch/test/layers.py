import unittest
import torch
import sys
sys.path.append('.')
from pytorch.src.layers import *
print("test")


class TestLayers(unittest.TestCase):
    def test_linear(self):
        data = torch.randn(6)
        layer = ModLinear(6, 4, masked=True)

        out = layer(data)
        self.assertTrue(out.shape == torch.Size([4]))

        layer.mask([0, 1])
        maskedout = layer(data)
        self.assertTrue(torch.allclose(maskedout[0:2], torch.zeros(2)))

        layer.unmask([0, 1])
        newout = layer(data)
        self.assertTrue(torch.allclose(newout, out))

        layer.prune([3])
        self.assertEqual(layer.weight.shape, torch.Size([3, 6]))
        self.assertEqual(layer.bias.shape, torch.Size([3]))
        prunedout = layer(data)
        self.assertEqual(prunedout.shape, torch.Size([3]))
        self.assertTrue(torch.allclose(prunedout, torch.cat((newout[0:3],newout[4:]))))

        layer.grow(2)
        self.assertEqual(layer.weight.shape, torch.Size([5, 6]))
        self.assertEqual(layer.bias.shape, torch.Size([5]))
        addedout = layer(data)
        self.assertEqual(addedout.shape, torch.Size([5]))
        self.assertTrue(torch.allclose(addedout[:-2], prunedout))
        self.assertTrue(torch.allclose(addedout[-2:], torch.zeros(2)))

        layer.grow(1, faninweights=torch.ones(1, 6))
        self.assertEqual(layer.weight.shape, torch.Size([6, 6]))
        self.assertEqual(layer.bias.shape, torch.Size([6]))
        newaddedout = layer(data)
        self.assertEqual(newaddedout.shape, torch.Size([6]))
        self.assertTrue(torch.allclose(newaddedout[:-1], addedout))
        self.assertNotEqual(newaddedout[-1], torch.zeros(1))

        layer.mask([], [1])
        newmaskedout = layer(data)

        layer.prune([], [1])
        self.assertEqual(layer.weight.shape, torch.Size([6, 5]))
        self.assertEqual(layer.bias.shape, torch.Size([6]))
        newprunedout = layer(torch.cat((data[0:1],data[2:])))
        self.assertTrue(torch.allclose(newprunedout, newmaskedout))


    def test_conv2d(self):
        data = torch.randn(8,4,8,8)
        layer = ModConv2d(masked=True, in_channels=4, out_channels=6, kernel_size=3, padding = 1)

        out = layer(data)
        self.assertTrue(out.shape == torch.Size([8,6,8,8]))

        layer.mask([0, 1])
        maskedout = layer(data)
        self.assertTrue(torch.allclose(maskedout[:,0:2,:,:], torch.zeros(8,2,8,8)))

        layer.unmask([0, 1])
        newout = layer(data)
        self.assertTrue(torch.allclose(newout, out))

        layer.prune([3])
        self.assertEqual(layer.weight.shape, torch.Size([5, 4, 3, 3]))
        self.assertEqual(layer.bias.shape, torch.Size([5]))
        prunedout = layer(data)
        self.assertEqual(prunedout.shape, torch.Size([8,5,8,8]))
        self.assertTrue(torch.allclose(prunedout[:,0:3,:,:], newout[:,0:3,:,:]))
        self.assertTrue(torch.allclose(prunedout[:,3:,:,:], newout[:,4:,:,:]))

        layer.grow(2)
        self.assertEqual(layer.weight.shape, torch.Size([7, 4, 3, 3]))
        self.assertEqual(layer.bias.shape, torch.Size([7]))
        addedout = layer(data)
        self.assertEqual(addedout.shape, torch.Size([8,7,8,8]))
        self.assertTrue(torch.allclose(addedout[:,:-2,:,:], prunedout))
        self.assertTrue(torch.allclose(addedout[:,-2:,:,:], torch.zeros(8,2,8,8)))  

        layer.grow(1, faninweights=torch.ones(1, 4, 3, 3))
        self.assertEqual(layer.weight.shape, torch.Size([8, 4, 3, 3]))
        self.assertEqual(layer.bias.shape, torch.Size([8]))
        newaddedout = layer(data)
        self.assertEqual(newaddedout.shape, torch.Size([8,8,8,8]))
        self.assertTrue(torch.allclose(newaddedout[:,:-1,:,:], addedout[:,:,:,:]))
        self.assertFalse(torch.allclose(newaddedout[:,-1,:,:], torch.zeros(8,1,8,8)))

        layer.mask([], [1])
        newmaskedout = layer(data)

        layer.prune([], [1])
        self.assertEqual(layer.weight.shape, torch.Size([8, 3, 3, 3]))
        self.assertEqual(layer.bias.shape, torch.Size([8]))
        newprunedout = layer(torch.cat((data[:,0:1,:,:],data[:,2:,:,:]), dim=1))
        self.assertTrue(torch.allclose(newprunedout, newmaskedout))




        


    
if __name__ == '__main__':
    unittest.main()
