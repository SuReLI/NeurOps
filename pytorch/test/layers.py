import unittest
import torch
import sys
sys.path.append('.')
from pytorch.neurops.layers import *


class TestLayers(unittest.TestCase):
    def test_linear(self):
        data = torch.randn(6)
        layer = ModLinear(6, 4, masked=True, nonlinearity='')

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

        layer.grow(1, fanin_weights=torch.ones(1, 6))
        self.assertEqual(layer.weight.shape, torch.Size([6, 6]))
        self.assertEqual(layer.bias.shape, torch.Size([6]))
        newaddedout = layer(data)
        self.assertEqual(newaddedout.shape, torch.Size([6]))
        self.assertTrue(torch.allclose(newaddedout[:-1], addedout))
        self.assertNotEqual(newaddedout[-1], torch.zeros(1))

        layer.prune([], [1])
        self.assertEqual(layer.weight.shape, torch.Size([6, 5]))
        self.assertEqual(layer.bias.shape, torch.Size([6]))


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

        layer.grow(1, fanin_weights=torch.ones(1, 4, 3, 3))
        self.assertEqual(layer.weight.shape, torch.Size([8, 4, 3, 3]))
        self.assertEqual(layer.bias.shape, torch.Size([8]))
        newaddedout = layer(data)
        self.assertEqual(newaddedout.shape, torch.Size([8,8,8,8]))
        self.assertTrue(torch.allclose(newaddedout[:,:-1,:,:], addedout[:,:,:,:]))
        self.assertFalse(torch.allclose(newaddedout[:,-1,:,:], torch.zeros(8,1,8,8)))

        layer.prune([], [1])
        self.assertEqual(layer.weight.shape, torch.Size([8, 3, 3, 3]))
        self.assertEqual(layer.bias.shape, torch.Size([8]))


    def test_optimizer(self):
        data = [torch.randn(8, 6) for _ in range(5)] 
        labels = [torch.randn(8, 2) for _ in range(5)]
        layer = ModLinear(6, 4, masked=True)
        layer2 = ModLinear(4, 2, masked=True)
        model = nn.Sequential(layer, torch.nn.ReLU(), layer2)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        lossfunction = nn.CrossEntropyLoss()

        out = model(data[0])
        self.assertTrue(out.shape == torch.Size([8, 2]))

        layer.mask([0, 1])
        for batch, label in zip(data, labels):
            optimizer.zero_grad()
            maskedout = model(batch)
            loss = lossfunction(maskedout, label)
            loss.backward()
            optimizer.step()

        layer.prune([2], optimizer=optimizer)
        layer2.prune([],[2], optimizer=optimizer)
        self.assertEqual(layer.weight.shape, torch.Size([3, 6]))
        self.assertEqual(layer.bias.shape, torch.Size([3]))
        self.assertTrue(layer.weight in optimizer.param_groups[0]['params'])
        self.assertTrue(len(optimizer.state[layer.weight]) != 0)

        for batch, label in zip(data, labels):
            optimizer.zero_grad()
            maskedout = model(batch)
            loss = lossfunction(maskedout, label)
            loss.backward()
            optimizer.step()

        layer.grow(2, fanin_weights="kaiming", optimizer=optimizer)
        layer2.grow(0, 2, fanout_weights="kaiming", optimizer=optimizer)
        self.assertEqual(layer.weight.shape, torch.Size([5, 6]))
        self.assertEqual(layer.bias.shape, torch.Size([5]))
        self.assertTrue(layer.weight in optimizer.param_groups[0]['params'])
        self.assertTrue(len(optimizer.state[layer.weight]) != 0)
        for batch, label in zip(data, labels):
            optimizer.zero_grad()
            maskedout = model(batch)
            loss = lossfunction(maskedout, label)
            loss.backward()
            optimizer.step()

        layer.mask([4])
        layer.unmask([0,4], [], optimizer=optimizer)
        layer2.unmask([], [0,4], optimizer=optimizer)
        self.assertEqual(layer.weight.shape, torch.Size([5, 6]))
        self.assertEqual(layer.bias.shape, torch.Size([5]))

        data = [torch.randn(8, 4, 4, 4) for _ in range(5)]
        labels = [torch.randn(8, 1, 1, 1) for _ in range(5)]
        layer = ModConv2d(masked=True, in_channels=4, out_channels=6, kernel_size=3)
        layer2 = ModConv2d(masked=True, in_channels=6, out_channels=1, kernel_size=2)
        model = nn.Sequential(layer, torch.nn.ReLU(), layer2) 
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        lossfunction = nn.CrossEntropyLoss()

        out = model(data[0])
        self.assertTrue(out.shape == torch.Size([8, 1, 1, 1]))

        layer.mask([0, 1])
        for batch, label in zip(data, labels):
            optimizer.zero_grad()
            maskedout = model(batch)
            loss = lossfunction(maskedout, label)
            loss.backward()
            optimizer.step()

        layer.prune([2], optimizer=optimizer)
        layer2.prune([],[2], optimizer=optimizer)
        self.assertEqual(layer.weight.shape, torch.Size([5, 4, 3, 3]))
        self.assertEqual(layer.bias.shape, torch.Size([5]))
        self.assertTrue(layer.weight in optimizer.param_groups[0]['params'])
        self.assertTrue(len(optimizer.state[layer.weight]) != 0)

        for batch, label in zip(data, labels):
            optimizer.zero_grad()
            maskedout = model(batch)
            loss = lossfunction(maskedout, label)
            loss.backward()
            optimizer.step()
        
        layer.grow(2, optimizer=optimizer)
        layer2.grow(0, 2, optimizer=optimizer)
        self.assertEqual(layer.weight.shape, torch.Size([7, 4, 3, 3]))
        self.assertEqual(layer.bias.shape, torch.Size([7]))
        self.assertTrue(layer.weight in optimizer.param_groups[0]['params'])
        self.assertTrue(len(optimizer.state[layer.weight]) != 0)

    def test_batchnorm(self):
        layer = ModLinear(6, 4, masked=True, prebatchnorm=True)

        data = torch.randn(8, 6)
        out = layer(data)
        self.assertTrue(out.shape == torch.Size([8, 4]))
        self.assertFalse(torch.allclose(layer.batchnorm.running_mean, torch.zeros(6)))
        self.assertFalse(torch.allclose(layer.batchnorm.running_var, torch.ones(6)))

        layer.mask([1, 2])
        maskedout = layer(data)
        self.assertTrue(maskedout.shape == torch.Size([8, 4]))
        self.assertTrue(torch.allclose(maskedout[:, 1:3], torch.zeros(8, 2)))

        layer.unmask([2], [1])
        self.assertTrue(torch.allclose(layer.batchnorm.running_mean[2], torch.zeros(1)))
        self.assertTrue(torch.allclose(layer.batchnorm.running_var[2], torch.ones(1)))
        unmaskedout = layer(data)
        self.assertTrue(unmaskedout.shape == torch.Size([8, 4]))
        rm = layer.batchnorm.running_mean

        layer.prune([], [0])
        self.assertTrue(torch.allclose(layer.batchnorm.running_mean, rm[1:]))

        layer.grow(0, 1)
        self.assertTrue(torch.allclose(layer.batchnorm.running_mean[:-1], rm[1:]))
        self.assertTrue(torch.allclose(layer.batchnorm.running_mean[-1], torch.zeros(1)))

        layer = ModConv2d(in_channels = 6, out_channels = 4, kernel_size = 3, masked=True, prebatchnorm=True)
        data = torch.randn(8, 6, 4, 4)

        out = layer(data)
        self.assertTrue(out.shape == torch.Size([8, 4, 2, 2]))
        self.assertFalse(torch.allclose(layer.batchnorm.running_mean, torch.zeros(6)))
        self.assertFalse(torch.allclose(layer.batchnorm.running_var, torch.ones(6)))

        layer.mask([1, 2])
        maskedout = layer(data)
        self.assertTrue(maskedout.shape == torch.Size([8, 4, 2, 2]))
        self.assertTrue(torch.allclose(maskedout[:, 1:3], torch.zeros(8, 2, 2, 2)))

        layer.unmask([2], [1])
        self.assertTrue(torch.allclose(layer.batchnorm.running_mean[1], torch.zeros(1)))
        self.assertTrue(torch.allclose(layer.batchnorm.running_var[1], torch.ones(1)))
        unmaskedout = layer(data)
        self.assertTrue(unmaskedout.shape == torch.Size([8, 4, 2, 2]))
        rm = layer.batchnorm.running_mean

        layer.prune([], [0])
        self.assertTrue(torch.allclose(layer.batchnorm.running_mean, rm[1:]))

        layer.grow(0, 1)
        self.assertTrue(torch.allclose(layer.batchnorm.running_mean[:-1], rm[1:]))
        self.assertTrue(torch.allclose(layer.batchnorm.running_mean[-1], torch.zeros(1)))

    def test_auxiliary(self):
        layer1 = ModLinear(6, 4, masked=True)
        layer2 = ModLinear(4, 3, masked=True)
        data = torch.randn(8, 6)
        output = torch.randn(8, 3)
        aux = torch.nn.parameter.Parameter(torch.zeros(3, 6))
        lossfunction = nn.CrossEntropyLoss()
        
        inter = layer1(data)
        out = layer2(inter, aux, data, layer1)
        loss = lossfunction(out, output)
        loss.backward()
        self.assertTrue(out.shape == torch.Size([8, 3]))
        self.assertTrue(aux.grad.shape == torch.Size([3, 6]))
        oldauxgrad = aux.grad.clone()
        self.assertFalse(torch.allclose(aux.grad, torch.zeros(3, 6)))
        aux.grad = torch.zeros(3, 6)

        inter = layer1(data)
        out = layer2(inter, aux, data, layer1)
        loss = lossfunction(out, output)
        loss.backward()
        self.assertTrue(torch.allclose(aux.grad, oldauxgrad))
        aux.grad = torch.zeros(3, 6)

        layer1.mask([0, 1])

        inter = layer1(data)
        out = layer2(inter, aux, data, layer1)
        loss = lossfunction(out, output)
        loss.backward()
        auxgrad = aux.grad.clone()
        self.assertTrue(out.shape == torch.Size([8, 3]))
        self.assertTrue(aux.grad.shape == torch.Size([3, 6]))
        self.assertFalse(torch.allclose(auxgrad, oldauxgrad))
        aux.grad = torch.zeros(3, 6)

        layer1.prune([0,1])
        layer2.prune([], [0,1])

        inter = layer1(data)
        out = layer2(inter, aux, data, layer1)
        loss = lossfunction(out, output)
        loss.backward()
        self.assertTrue(torch.allclose(aux.grad, auxgrad))
        aux.grad = torch.zeros(3, 6)

        layer1.grow(1, fanin_weights="kaiming")
        layer2.grow(0, 1, fanout_weights="kaiming")

        inter = layer1(data)
        out = layer2(inter, aux, data, layer1)
        loss = lossfunction(out, output)
        loss.backward()
        self.assertFalse(torch.allclose(aux.grad, auxgrad))
        aux.grad = torch.zeros(3, 6)

        layer1 = ModConv2d(in_channels=6, out_channels=2, kernel_size=3, masked=True)
        layer2 = ModLinear(2, 3, masked=True, preflatten=True)
        data = torch.randn(8, 6, 3, 3)
        output = torch.randn(8, 3)
        aux = torch.nn.parameter.Parameter(torch.zeros(3, 6, 3, 3))
        lossfunction = nn.CrossEntropyLoss()

        inter = layer1(data)
        out = layer2(inter, aux, data, layer1)
        loss = lossfunction(out, output)
        loss.backward()
        self.assertTrue(out.shape == torch.Size([8, 3]))
        self.assertTrue(aux.grad.shape == torch.Size([3, 6, 3, 3]))
        oldauxgrad = aux.grad.clone()
        self.assertFalse(torch.allclose(aux.grad, torch.zeros(3, 6, 3, 3)))
        aux.grad = torch.zeros(3, 6, 3, 3)

        layer1 = ModConv2d(in_channels=6, out_channels=2, kernel_size=3, padding = 1, masked=True)
        layer2 = ModConv2d(in_channels=2, out_channels=3, kernel_size=3, masked=True)
        data = torch.randn(8, 6, 3, 3)
        output = torch.randn(8, 3, 1, 1)
        aux = torch.nn.parameter.Parameter(torch.zeros(3, 6, 5, 5))
        lossfunction = nn.MSELoss()

        inter = layer1(data)
        out = layer2(inter, aux, data, layer1)
        loss = lossfunction(out, output)
        loss.backward()
        self.assertTrue(out.shape == torch.Size([8, 3, 1, 1]))
        self.assertTrue(aux.grad.shape == torch.Size([3, 6, 5, 5]))
        self.assertFalse(torch.allclose(aux.grad, torch.zeros(3, 6, 5, 5)))














if __name__ == '__main__':
    unittest.main()
