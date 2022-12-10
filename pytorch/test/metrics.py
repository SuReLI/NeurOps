import unittest
import torch
import sys
sys.path.append('.')
from pytorch.src.models import *
from pytorch.src.layers import *
from pytorch.src.metrics import *


class TestMetrics(unittest.TestCase):
    def test_effectivesvd(self):
        model = ModSequential(
            ModLinear(2, 10, masked=True),
            ModLinear(10, 1, masked=True),
            trackacts = True
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
        effdim = effectivesvd(tensor=model.activations["0"], threshold=0.01, partial=False)
        self.assertLessEqual(effdim, 10)

        model.mask(0, [0,1,7,8,9])
        for x, y in zip(data, labels):
            optimizer.zero_grad()
            yhat = model(x)
            loss = torch.nn.functional.mse_loss(yhat, y)
            loss.backward()
            optimizer.step()
        effdim2 = effectivesvd(tensor=model.activations["0"], threshold=0.01, partial=False)
        self.assertTrue(effdim2 < effdim)
        self.assertTrue(effdim2 <= 5)
        effdim2partial = effectivesvd(tensor=model.activations["0"], threshold=0.01, partial=True)
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
        effdim3 = effectivesvd(tensor=model.activations["0"], threshold=0.01, partial=False)
        self.assertTrue(effdim3 < effdim2)
        effdim3partial = effectivesvd(tensor=model.activations["0"], threshold=0.01, partial=True)
        self.assertTrue(effdim3partial > effdim3)

        model.unmask(0, [0,1,7,8,9], clearacts=True)
        model.prune(0, [0,1,2])
        for x, y in zip(data, labels):
            optimizer.zero_grad()
            yhat = model(x)
            loss = torch.nn.functional.mse_loss(yhat, y)
            loss.backward()
            optimizer.step()
        effdim4 = effectivesvd(tensor=model.activations["0"], threshold=0.01, partial=False)
        self.assertTrue(effdim4 <= 7)




        


if __name__ == '__main__':
    unittest.main()

