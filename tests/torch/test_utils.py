import unittest
import tempfile
import os
import torchvision
import torch
from maruti.torch import utils


class TorchUtilsTest(unittest.TestCase):
    def setUp(self):
        self.model = torchvision.models.resnet18(False)

    def tearDown(self):
        self.model = None

    def test_freeze_unfreeze(self):
        utils.freeze(self.model)
        for param in self.model.parameters():
            self.assertFalse(param.requires_grad)
        utils.unfreeze(self.model)
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

    def test_layer_freeze_unfreeze(self):
        layers = ['fc.weight', 'layer1.0', 'layer2', 'layer1', 'layer3.0']
        utils.freeze_layers(self.model, layers)
        for name, layer in self.model.named_parameters():
            tested = False
            for to_freeze in layers:
                if name.startswith(to_freeze):
                    tested = True
                    self.assertFalse(layer.requires_grad)
            if not tested:
                self.assertTrue(layer.requires_grad)
        utils.unfreeze_layers(self.model, layers)
        for param in self.model.parameters():
            self.assertTrue(param.requires_grad)

    def test_children_names(self):
        names = utils.children_names(self.model)
        layers = {'fc', 'layer1', 'layer2', 'layer3',
                  'layer4', 'conv1', 'bn1', 'relu', 'maxpool', 'avgpool'}
        self.assertEquals(len(names), len(layers))
        for name in names:
            self.assertTrue(name in layers)
