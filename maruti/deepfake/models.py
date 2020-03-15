import torch.nn as nn
import torchvision
from ..torch.utils import freeze
from torch.nn.utils.rnn import PackedSequence, pack_sequence
import itertools
from .dataset import group_transform


def resnext50(feature=False, pretrained=False):
    model = torchvision.models.resnext50_32x4d(pretrained)
    if feature:
        model.fc = nn.Identity()
    else:
        model.fc = nn.Linear(2048, 1)
    return model


def binaryClassifier(input_features):
    return nn.Sequential(
        nn.Linear(input_features, input_features // 2),
        nn.ReLU(),
        nn.BatchNorm1d(input_features // 2),

        nn.Linear(input_features // 2, input_features // 2),
        nn.ReLU(),
        nn.BatchNorm1d(input_features // 2),

        nn.Linear(input_features // 2, 128),
        nn.ReLU(),
        nn.Dropout(),

        nn.Linear(128, 1),
        nn.Flatten())


class ResLSTM(nn.Module):

    def __init__(self, pretrained=False, hidden_size=512, num_layers=1, bidirectional=True, dropout=0.5):
        super().__init__()
        # resnext
        self.feature_model = resnext50(True, pretrained)

        # lstm
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(2048, hidden_size=hidden_size, num_layers=num_layers,
                            bidirectional=bidirectional, dropout=dropout)
        classifier_features = hidden_size * num_layers
        if bidirectional:
            classifier_features *= 2
        self.classifier = binaryClassifier(classifier_features)

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


class ReslstmNN(nn.Module):
    def __init__(self, num_sets=6, pretrained=False, hidden_size=512, num_layers=1, bidirectional=True, dropout=0.5):
        super().__init__()
        self.feature = ResLSTM(pretrained=pretrained, hidden_size=hidden_size,
                               num_layers=num_layers, bidirectional=bidirectional, dropout=dropout)
        self.feature.classifier[9] = nn.Identity()
        self.feature.classifier[10] = nn.Identity()
        self.classifier = binaryClassifier(128 * num_sets)

    def forward(x):
        preds = []
        for vid_set in x:
            preds.append(self.feature(vid_set))
        preds = torch.cat(preds, dim=1)
        preds = self.classifier(preds)
        return preds.squeeze(dim=1)

    @staticmethod
    def transform(vid_sets):
        transformed = []
        for vid in vid_sets:
            transformed.append(group_transform(vid))
        return transformed

    @staticmethod
    def collate(batches):
        ps_list = []
        for set_idx in range(len(batches[0][0])):
            vids = [batch[set_idx] for batch, target in batches]
            ps = pack_sequence(vids, False)
            ps_list.append(ps)
        return ps, torch.tensor([target for _, target in batches])
