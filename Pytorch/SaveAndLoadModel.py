import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import numpy as np
from pathlib import Path
from resnet import resnet18
from train import Trainer
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.io


model = resnet18(num_classes=4)
# model.load_state_dict(torch.load('./models/model_epoch_99.pth')["weight"])
model.load_state_dict(torch.load('./models/model_epoch_99.pth'))
model.fc = nn.Sequential()
model = model.eval()


print(model)


train_dir = './Data/train_data/'
test_dir = './Data/test_data/'
batch_size = 1

deg = random.random() * 10
to_normalized_tensor = [transforms.CenterCrop(224),transforms.ColorJitter(),transforms.RandomRotation(deg),transforms.ToTensor(),transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])]


traindir = train_dir
testdir = test_dir
train = datasets.ImageFolder(traindir,transforms.Compose(to_normalized_tensor))
test = datasets.ImageFolder(testdir, transforms.Compose(to_normalized_tensor))


train_loader = DataLoader(train, batch_size = batch_size, shuffle=False, num_workers=8)
test_loader = DataLoader(test, batch_size=batch_size,shuffle=False, num_workers=8)


output_train = []
for data, target in train_loader:
    tmp = model(data)
    output_train.append(tmp)


output_test = []
for data, target in test_loader:
    tmp1 = model(data)
    output_test.append(tmp1)


result_test = {'features': output_test}
result_train = {'features',output_train}
scipy.io.mat(result_test)
scipy.io.mat(result_train)
