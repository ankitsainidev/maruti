import torch
from .utils import _limit_string


def progressive_mean(new, old, iteration):
    return ((old * iteration) + new) / (iteration + 1)


reduction_dict = {
    'mean': {'function': progressive_mean, 'starting_value': 0}
}


class BaseMetric:
    """Base Class for metrics for pytorch"""

    def __init__(self, name=None, starting_value=0):
        self.iteration = 0
        self.starting_value = starting_value
        self.value = starting_value
        self.name = name

    def loss(self, ypred, y):
        raise NotImplementedError

    def reducer(self, newVal, oldVal, iteration):
        return progressive_mean(newVal, oldVal, iteration)

    def __call__(self, ypred, y):
        newLoss = self.loss(ypred, y)
        self.update(newLoss)
        return self.value

    def update(self, newVal):
        self.value = self.reducer(newVal, self.value, self.iteration)
        self.iteration += 1

    def reset(self):
        self.value = self.starting_value
        self.iteration = 0

    def __str__(self):
        return _limit_string(round(self.value, 5), 12)


class Accuracy(BaseMetric):
    def __init__(self, name='accuracy'):
        super().__init__(name=name, reduction='mean', starting_value=0)

    def loss(self, ypred, y):
        assert ypred.shape == y.shape, 'invalid dimensions'
        loss = (ypred == y).float().mean()
        return loss.item()


class BinaryAccuracy(Accuracy):
    def loss(self, ypred, y):
        assert ypred.shape == y.shape, 'invalid dimensions'
        ypred = (ypred > 0.5).float()
        return super().loss(ypred, y)


class MultiClassAccuracy(Accuracy):
    def loss(self, ypred, y):
        assert ypred.shape[:1] == y.shape, 'invalid dimensions'
        ypred = ypred.max(1)[1]
        return super().loss(ypred, y)
