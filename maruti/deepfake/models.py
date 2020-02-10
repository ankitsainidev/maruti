import torch.nn as nn


class MultiFrame(nn.Module):
    def __init__(self, model, frames, feature_size=64):
        super().__init__()
        self.feature = model
        self.frames = frames
        self.fc = nn.Sequential(nn.Linear(
            self.feature_size * self.frames, 32), nn.Linear(32, 4), nn.Linear(32, 1))

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1:] == (self.frames, 3, 224, 224)
#         output = []
        x = model(x.view(-1, 3, 224, 224))
        x = x.view(batch_size, -1)
#         for i,sample in enumerate(x):
#             output.append(model(sample).view(-1))
        return self.fc(x)
