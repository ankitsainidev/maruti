from .general import *
from .general import __all__ as gen_all
import torch
import torch.nn as nn
import torchvision.transforms as torch_transforms
import torchvision
import maruti.torch as mtorch
import maruti
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils import data
import torch.optim as optim
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
__all__ = gen_all + ['pd','device','plt','np','torch', 'nn', 'torch_transforms',
                     'torchvision', 'mtorch', 'maruti', 'data', 'optim']
