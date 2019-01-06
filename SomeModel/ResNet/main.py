import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import se_module
from pathlib import Path
from se_resnet import se_resnet50
from train import Trainer
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

def get_dataloader(batch_size, train_dir, test_dir):
    root = './mnist/'
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    train_set = datasets.MNIST(root=root, train=True, transform=trans, download=False)
    test_set = datasets.MNIST(root=root, train=False, transform=trans)
    #train_set = datasets.MNIST(root=root, train=True, download=False)
    #test_set = datasets.MNIST(root=root, train=False)


    train_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=train_set,batch_size=batch_size,shuffle=True)
    return train_loader, test_loader

def main(batch_size, train_dir, test_dir):
    train_loader, test_loader = get_dataloader(batch_size, train_dir, test_dir)
    #multi GPU
    #se_resnet = nn.DataParallel(se_resnet18(num_classes=2),
    #                            device_ids=list(range(torch.cuda.device_count())))
    se_resnet = se_resnet50(num_classes=2)
    optimizer = optim.SGD(params=se_resnet.parameters(), lr=0.6, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    trainer = Trainer(se_resnet, optimizer, F.cross_entropy, save_dir="./")
    trainer.loop(100, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    train_dir = '/media/haoran/Data1/LivenessDetection/Data/ReplayAttack/Train/'
    test_dir = '/media/haoran/Data1/LivenessDetection/Data/ReplayAttack/Test/'
    batch_size = 16

    main(batch_size, train_dir, test_dir)
