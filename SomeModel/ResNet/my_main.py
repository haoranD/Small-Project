import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
import se_module
import numpy as np
from pathlib import Path
from se_resnet import se_resnet50
from resnet import resnet50
from train import Trainer
import random
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import datasets, transforms

def get_dataloader(batch_size, train_dir, test_dir):
    deg = random.random() * 10
    to_normalized_tensor = [transforms.CenterCrop(224),
                            transforms.ColorJitter(),
                            transforms.RandomRotation(deg),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), ]
    #mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    #data_augmentation = [transforms.RandomResizedCrop(224),
    #                     transforms.RandomHorizontalFlip(), ]

    #transform = transforms.Compose([
    #    transforms.RandomHorizontalFlip(),  # The order seriously matters: RandomHorizontalFlip, ToTensor, Normalize
    #    transforms.ColorJitter(),
    #    transforms.RandomRotation(deg),
    #    transforms.ToTensor(),
    #    transforms.Normalize(),
    #]))

    traindir = train_dir
    testdir = test_dir
    train = datasets.ImageFolder(traindir,transforms.Compose(to_normalized_tensor))
    test = datasets.ImageFolder(testdir, transforms.Compose(to_normalized_tensor))
    #train = datasets.ImageFolder(traindir)
    #test = datasets.ImageFolder(testdir)

    train_loader = DataLoader(
        train, batch_size = batch_size, shuffle=True, num_workers=8)
    test_loader = DataLoader(
        test, batch_size=batch_size,shuffle=True, num_workers=8)
    return train_loader, test_loader

def main(batch_size, train_dir, test_dir):
    use_gpu = torch.cuda.is_available()
    train_loader, test_loader = get_dataloader(batch_size, train_dir, test_dir)
    
    resnet = se_resnet50(num_classes=10)
    optimizer = optim.SGD(params=resnet.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 30, gamma=0.1)
    trainer = Trainer(resnet, optimizer, torch.nn.CrossEntropyLoss(), save_dir="./")
    trainer.loop(100, train_loader, test_loader, scheduler)


if __name__ == '__main__':
    train_dir = ''
    test_dir = ''
    batch_size = 16

    main(batch_size, train_dir, test_dir)
