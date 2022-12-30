import unittest
import torch
import sys
sys.path.append('.')
from pytorch.src.models import *
from pytorch.src.layers import *
from pytorch.src.metrics import *


class TestMetrics(unittest.TestCase):
    def test_effective_rank(self):
        model = ModSequential(
            ModLinear(2, 10, masked=True),
            ModLinear(10, 1, masked=True),
            track_activations = True
        )
        data = [torch.randn(8, 2) for _ in range(5)] 
        labels = [torch.randn(8, 1) for _ in range(5)]
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        for x, y in zip(data, labels):
            optimizer.zero_grad()
            yhat = model(x)
            loss = torch.nn.functional.mse_loss(yhat, y)
            loss.backward()
            optimizer.step()
        effdim = effective_rank(tensor=model.activations["0"], threshold=0.01, partial=False)
        self.assertLessEqual(effdim, 10)

        model.mask(0, [0,1,7,8,9])
        for x, y in zip(data, labels):
            optimizer.zero_grad()
            yhat = model(x)
            loss = torch.nn.functional.mse_loss(yhat, y)
            loss.backward()
            optimizer.step()
        effdim2 = effective_rank(tensor=model.activations["0"], threshold=0.01, partial=False)
        self.assertTrue(effdim2 < effdim)
        self.assertTrue(effdim2 <= 5)
        effdim2partial = effective_rank(tensor=model.activations["0"], threshold=0.01, partial=True)
        self.assertTrue(effdim2partial > effdim2)

        lowrankdata = torch.randn(1, 2)
        lowrankdata = torch.cat([lowrankdata]*8, dim=0)
        lowranky = torch.randn(8, 1)
        for _ in range(50):
            optimizer.zero_grad()
            yhat = model(lowrankdata)
            loss = torch.nn.functional.mse_loss(yhat, lowranky)
            loss.backward()
            optimizer.step()
        effdim3 = effective_rank(tensor=model.activations["0"], threshold=0.01, partial=False)
        self.assertTrue(effdim3 < effdim2)
        effdim3partial = effective_rank(tensor=model.activations["0"], threshold=0.01, partial=True)
        self.assertTrue(effdim3partial > effdim3)

        model.unmask(0, [0,1,7,8,9], clear_activations=True)
        model.prune(0, [0,1,2])
        for x, y in zip(data, labels):
            optimizer.zero_grad()
            yhat = model(x)
            loss = torch.nn.functional.mse_loss(yhat, y)
            loss.backward()
            optimizer.step()
        effdim4 = effective_rank(tensor=model.activations["0"], threshold=0.01, partial=False)
        self.assertTrue(effdim4 <= 7)

        model = ModSequential(
                ModConv2d(in_channels=1, out_channels=8, kernel_size=7, masked=True, padding=1),
                ModConv2d(in_channels=8, out_channels=16, kernel_size=5, masked=True, padding = 1),
                ModLinear(1024, 256, masked=True),
                ModLinear(256, 10, masked=True),
                track_activations=True,
                track_auxiliary_gradients=True,
                input_shape=(1, 14, 14)
            )
            
        data = [torch.randn(16, 1, 14, 14) for _ in range(10)]
        labels = [torch.randint(0, 10, (16,)) for _ in range(10)]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for i in range(10):
            optimizer.zero_grad()
            x = data[i]
            ytrue = labels[i]
            y = model(x, auxiliaries=model.auxiliaries)
            loss = criterion(y, ytrue)
            loss.backward()
            optimizer.step()

        effdim = effective_rank(tensor=model.activations["0"], threshold=0.01, partial=False)
        self.assertTrue(len(effdim) == 8)
        
    def test_weight_sum(self):
        layer = ModLinear(2, 10, masked=True)
        layer.weight.data = torch.ones(10, 2)
        fanout = weight_sum(layer.weight, fanin=False)
        fanin = weight_sum(layer.weight, fanin=True)
        self.assertTrue(len(fanout) == 2)
        self.assertTrue(len(fanin) == 10)
        self.assertTrue(torch.sum(fanout) == 20)
        self.assertTrue(torch.sum(fanin) == 20)
        self.assertTrue(fanout[0] == 10)
        self.assertTrue(fanin[0] == 2)
        fanin_fro = weight_sum(layer.weight, fanin=True, p="fro")
        self.assertTrue(fanin_fro[0] < fanin[0])

        model = ModSequential(
                ModConv2d(in_channels=1, out_channels=8, kernel_size=7, masked=True, padding=1),
                ModConv2d(in_channels=8, out_channels=16, kernel_size=5, masked=True, padding = 1),
                ModLinear(1024, 256, masked=True),
                ModLinear(256, 10, masked=True),
                track_activations=True,
                track_auxiliary_gradients=True,
                input_shape=(1, 14, 14)
            )
            
        data = [torch.randn(16, 1, 14, 14) for _ in range(10)]
        labels = [torch.randint(0, 10, (16,)) for _ in range(10)]
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        for i in range(10):
            optimizer.zero_grad()
            x = data[i]
            ytrue = labels[i]
            y = model(x, auxiliaries=model.auxiliaries)
            loss = criterion(y, ytrue)
            loss.backward()
            optimizer.step()

        fanout = weight_sum(model[1].weight, fanin=False)
        fanin = weight_sum(model[0].weight, fanin=True)
        self.assertTrue(len(fanout) == 8)
        self.assertTrue(len(fanin) == 8)
        fanout2 = weight_sum(model[2].weight, fanin=False, conversion_factor=model.conversion_factor)
        fanin2 = weight_sum(model[1].weight, fanin=True)
        self.assertTrue(len(fanin2) == 16)
        self.assertTrue(len(fanout2) == 16)
    
    def test_activation_variance(self):
        model = ModSequential(
            ModLinear(2, 10, masked=True),
            ModLinear(10, 1, masked=True),
            track_activations = True
        )
        data = [torch.randn(8, 2) for _ in range(5)] 
        for x in data:
            _ = model(x)
        var = activation_variance(activations=model.activations["0"])

        model.mask(0, [0,1,7,8,9])
        for x in data:
            _ = model(x)
        var2 = activation_variance(activations=model.activations["0"])
        self.assertTrue(all(var2 <= var))
        self.assertTrue(var2[0] == 0)

        lowvardata = torch.randn(1, 2)
        lowvardata = torch.cat([lowvardata]*20, dim=0)
        _ = model(lowvardata)
        var3 = activation_variance(activations=model.activations["0"])
        self.assertTrue(torch.allclose(var3, torch.zeros_like(var3)))
    
    def test_orthogonality_gap(self):
        model = ModSequential(
            ModLinear(2, 20, masked=True),
            ModLinear(20, 1, masked=True),
            track_activations = True
        )
        data = [torch.randn(8, 2) for _ in range(5)] 
        for x in data:
            _ = model(x)
        gap = orthogonality_gap(activations=model.activations["0"])

        model.mask(0, [0,1,17,18,19])
        for x in data:
            _ = model(x)
        gap2 = orthogonality_gap(activations=model.activations["0"])
        self.assertTrue(gap2 <= gap)

    def test_svd_score(self):
        model = ModSequential(
            ModLinear(2, 10, masked=True),
            ModLinear(10, 1, masked=True),
            track_activations = True
        )
        data = [torch.randn(8, 2) for _ in range(5)] 
        for x in data:
            _ = model(x)
        scores = svd_score(tensor=model.activations["0"])

        model.mask(0, [0,1,7,8,9])
        for x in data:
            _ = model(x)
        scores2 = svd_score(tensor=model.activations["0"])
        self.assertTrue(torch.allclose(scores2[0], scores2[1]))
        self.assertTrue(scores2.mean()<scores.mean())

        for i in range(1, 6):
            model[0].weight.data[i, :] = model[0].weight.data[0, :]
            model[0].bias.data[i] = model[0].bias.data[0]
        for x in data:
            _ = model(x)
        scores3 = svd_score(tensor=model.activations["0"])
        self.assertTrue(scores3[4]==scores3[5])

    def test_nuclear_score(self):
        model = ModSequential(
            ModLinear(2, 10, masked=True),
            ModLinear(10, 1, masked=True),
            track_activations = True
        )
        data = [torch.randn(8, 2) for _ in range(5)] 
        for x in data:
            _ = model(x)
        scores = nuclear_score(model.activations["0"])

        model.mask(0, [0,1,7,8,9])
        for x in data:
            _ = model(x)
        scores2 = nuclear_score(model.activations["0"])
        self.assertTrue(torch.allclose(scores2[0], scores2[1]))
        self.assertTrue(scores2.mean()<scores.mean())

        for i in range(1, 6):
            model[0].weight.data[i, :] = model[0].weight.data[0, :]
            model[0].bias.data[i] = model[0].bias.data[0]
        for x in data:
            _ = model(x)
        scores3 = nuclear_score(model.activations["0"])
        self.assertTrue(scores3[4]==scores3[5])

    def test_fisher_info(self):
        model = ModSequential(
            ModLinear(2, 10, masked=True, learnable_mask=True),
            ModLinear(10, 1, masked=True),
            track_activations = True
        )
        data = [torch.randn(8, 2) for _ in range(5)] 
        labels = [torch.randn(8, 1) for _ in range(5)]
        mask_grads = torch.empty(0, *model[0].mask_vector.shape)
        for x,y in zip(data, labels):
            model[0].mask_vector.grad = None
            yhat = model(x)
            loss = torch.nn.functional.mse_loss(yhat, y)
            loss.backward()
            mask_grads = torch.cat([mask_grads, model[0].mask_vector.grad.unsqueeze(0)])

        scores = fisher_info(mask_grads)
        self.assertTrue(len(scores)==10)

        model.mask(0, [0,1,7,8,9])
        mask_grads = torch.empty(0, *model[0].mask_vector.shape)
        for x,y in zip(data, labels):
            model[0].mask_vector.grad = None
            yhat = model(x)
            loss = torch.nn.functional.mse_loss(yhat, y)
            loss.backward()
            mask_grads = torch.cat([mask_grads, model[0].mask_vector.grad.unsqueeze(0)])

        scores2 = fisher_info(mask_grads)
        self.assertTrue(torch.allclose(scores2[-3:], torch.zeros_like(scores2[-3:])))


if __name__ == '__main__':
    unittest.main()

