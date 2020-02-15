import torch.nn as nn
import torchvision
from ..torch.utils import freeze


def resnext50(pretrained=False, feature=False, hidden_size=64):
    """feature:int weather to return hidden_size sized feature vector or binary classification"""
    model = torchvision.models.resnext50_32x4d(pretrained)
    model.fc = nn.Sequential(nn.Linear(2048, hidden_size),
                             nn.BatchNorm1d(64), nn.Linear(64, 1))
    if feature:
        model.fc[2] = nn.Identity()
    freeze(model)
    return model


class MultiFrame(nn.Module):
    """
    (model
    model
    model     -----   linar batchnorm linear -> output
    .
    .
    .)
    total frames
    """

    def __init__(self, model, frames, feature_size=64):
        super().__init__()
        self.feature = model
        self.frames = frames
        self.feature_size = feature_size
        self.fc = nn.Sequential(nn.Linear(self.feature_size * self.frames, 16),
                                nn.BatchNorm1d(16), nn.Linear(16, 1))

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1:] == (self.frames, 3, 224, 224)
        x = self.feature(x.view(-1, 3, 224, 224))
        x = x.view(batch_size, -1)
        return self.fc(x)
