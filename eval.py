import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.manifold import TSNE
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

def train_test_split(set_X, set_y, st, ed, seed=100):
    ''' train/test split to ensure class distribution is represented properly for small dataset'''
    random_state = np.random.RandomState(seed)
    by_class = [[] for _ in range(3)]
    for it, item in enumerate(set_y):
        by_class[item].append(it)
    index_train = []; index_test = []
    for each in by_class:
        #random_state.shuffle(each)
        begin = int(len(each) * st)
        until = int(len(each) * ed)
        index_train.append(each[:begin] + each[until:])
        index_test.append(each[begin:until])
    index_train = np.concatenate(index_train)
    index_test = np.concatenate(index_test)
    X_train = set_X[index_train]
    X_test = set_X[index_test]
    y_train = set_y[index_train]
    y_test = set_y[index_test]
    return X_train, X_test, y_train, y_test


X = torch.cat((torch.tensor(np.load(train_path_feat_y)), torch.tensor(np.load(train_path_feat_z)), torch.tensor(np.load(train_path_feat_x))), dim=-1)
print(X.shape)
y = torch.tensor(np.load(train_path_labels))
X_train, X_test, y_train, y_test = train_test_split(X, y, 0.8, 1.0)

train_set = TensorDataset(X_train, y_train)
test_set = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_set, batch_size=batch_size, sampler=ImbalancedDatasetSampler(train_set))
test_loader= DataLoader(test_set, batch_size=batch_size, shuffle=False)

def train_one_epoch(model, loader, loss_fun, optimizer, epoch, epoch_total, is_train):
    if is_train:
        model.train()
    else:
        model.eval()

    loss_count = 0.0
    correct = 0; total = 0

    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.set_grad_enabled(is_train):
            output = model(images)
            optimizer.zero_grad()
            loss = loss_fun(output, labels)
            if is_train:
                loss.backward()
                optimizer.step()

        _, prediction = torch.max(output.data, 1)
        correct = correct + (prediction == labels).sum().item()
        total = total + images.shape[0]
        loss_count = loss_count + loss.item()
    
    loss_count = loss_count / len(loader)
    if not is_train:
        print(f'epoch: {epoch} / {epoch_total}, is_train: {is_train}, loss: {loss_count: .5f}, accuracy: {(correct / total): .5f}')
    return loss_count, correct / total

class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.fcx = nn.Linear(384, 10)
        self.fcy = nn.Linear(384, 10)
        self.fcz = nn.Linear(384, 10)
        self.fc = nn.Linear(30, 3)
        self.activate = nn.Identity()
        #self.activate = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x1 = x[:, :384]
        x2 = x[:, 384: 384 * 2]
        x3 = x[:, 384 * 2:]
        x1 = self.activate(self.fcy(x1))
        x2 = self.activate(self.fcz(x2))
        x3 = self.activate(self.fcx(x3))
        x = torch.cat((x1, x2, x3), dim=-1)
        x = self.fc(x)
        return x

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

LR = 0.0001
EPOCH = 10000
model = LinearModel()
model.cuda()
loss_fun = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=0.01)
stopper = EarlyStopper(patience=500)

epoch = 0
while True:
    train_one_epoch(model=model, loader=train_loader,
                    loss_fun=loss_fun, optimizer=optimizer,
                    epoch=epoch, epoch_total=EPOCH, is_train=True)
    loss_count, accuracy = train_one_epoch(model=model, loader=test_loader,
                    loss_fun=loss_fun, optimizer=optimizer,
                    epoch=epoch, epoch_total=EPOCH, is_train=False)

    if stopper.early_stop(loss_count):
        break
    epoch = epoch + 1