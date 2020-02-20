import torch.nn as nn
import torchvision
from ..torch.utils import freeze
from torch.nn.utils.rnn import PackedSequence
import itertools


def resnext50(feature=False, pretrained=False):
    model = torchvision.models.resnext50_32x4d(pretrained)
    if feature:
        model.fc = nn.Identity()
    else:
        model.fc = nn.Linear(2048, 1)
    return model


class ResLSTM(nn.Module):

    def __init__(self, hidden_size=128):
        super().__init__()
        # resnext
        self.feature_model = resnext50(True)

        # lstm
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(2048, self.hidden_size, bidirectional=False)
        self.classifier = nn.Linear(self.hidden_size, 1)

    def forward(self, x):
        # indices
        unsorted_indices = x.unsorted_indices

        # prediction on all images from each batch
        x_data = self.feature_model(x.data)

        # converting again to PackedSequence
        x = PackedSequence(x_data, x.batch_sizes)

        # lstm
        out, (h, c) = self.lstm(x)
        batch_size = h.shape[1]

        # treat each batch differently instaed of lstm layer
        split_on_batch = h.permute(1, 0, 2)

        # reshape to make each bach flat
        combining_passes = split_on_batch.reshape(batch_size, -1)

        # classify
        val = self.classifier(combining_passes).squeeze(1)
        return val[unsorted_indices]

    def param(self, i=-1):
        # all
        if i == -1:
            return self.parameters()

        # grouped
        if i == 0:
            return itertools.chain(self.feature_model.conv1.parameters(),
                                   self.feature_model.bn1.parameters(),
                                   self.feature_model.layer1.parameters(),)
        if i == 1:
            return itertools.chain(self.feature_model.layer2.parameters(),
                                   self.feature_model.layer3.parameters())
        if i == 2:
            return itertools.chain(self.feature_model.layer4.parameters(),
                                   self.feature_model.fc.parameters())
        if i == 3:
            return itertools.chain(self.lstm.parameters(),
                                   self.classifier.parameters())
        else:
            print('there are only 4 param groups -> 0,1,2,3')
