import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import WeightedRandomSampler
import torch.nn as nn

import utils
from torchsampler import ImbalancedDatasetSampler


train_path_feat_x = '/home/leo101/Work/Harp/dino/trainfeatx.npy'
train_path_feat_y = '/home/leo101/Work/Harp/dino/trainfeaty.npy'
train_path_feat_z = '/home/leo101/Work/Harp/dino/trainfeatz.npy'
train_path_labels = '/home/leo101/Work/Harp/dino/trainlabelsy.npy'
batch_size = 32

X = torch.cat((torch.tensor(np.load(train_path_feat_y)), torch.tensor(np.load(train_path_feat_z)), torch.tensor(np.load(train_path_feat_x))), dim=-1)
y = torch.tensor(np.load(train_path_labels))

X = torch.tensor(StandardScaler().fit_transform(X, y))


correct = 0

for idx in range(len(X)):
    train_X = torch.cat([X[:idx, :], X[idx + 1:, :]], dim=0)
    train_y = torch.cat([y[:idx], y[idx + 1:]], dim=0)
    model = KNeighborsClassifier(20).fit(train_X, train_y)
    pred = model.predict(X[idx].reshape(1, -1))
    correct = correct + (int(pred) == y[idx].item())

print(correct)
print(correct / len(X))