# -*- coding: utf-8 -*-
"""PartA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15TrHis7V54Ndx11GYkFIYH-0V6OrMFgm
"""

import statistics
from torch import nn
from torch.nn import functional as Funn
from torchvision import datasets, transforms
import pytorch_lightning as pl
import pandas as pd
import os
import torch
import torchvision
from torch.utils.data import Dataset ,DataLoader
from torchmetrics import Accuracy
from pytorch_lightning.callbacks import ModelCheckpoint
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as Fnn
from torch.utils.data import DataLoader, sampler, random_split
from torchvision import models
from pytorch_lightning.loggers import WandbLogger
import matplotlib.pyplot as plt
from PIL import Image
import os
import random

import argparse
import wandb

wandb.login(key="ed57c3903aa24b40dc30a68b77aad62d1489535b")
pName = "CS6910 - Assignment 2"
run_obj=wandb.init( project=pName)

parser = argparse.ArgumentParser()
parser.add_argument('-wp','--wandb_project',default ='myprojectname',metavar="",required = False,type=str,help = "Project name used to track experiments in Weights & Biases dashboard" )
parser.add_argument('-we','--wandb_entity',default ='myname',metavar="",required = False,type=str,help = "Wandb Entity used to track experiments in the Weights & Biases dashboard." )
parser.add_argument('-e','--epochs',default=10,metavar="",required = False,type=int,help = "Number of epochs to train neural network.")
parser.add_argument('-do','--drop_OUT',default=0.3,metavar="",required = False,type=float,help = "Dropout")
parser.add_argument('-lr','--learning_rate',default=0.0001,metavar="",required = False,type=float,help = "Learning rate used to optimize model parameters")
parser.add_argument('-a','--activation_FUN',default='GELU',metavar="",required = False, help = "Activation Function", type=str,choices= ["SiLU", "Mish", "GELU", "ReLU"])
parser.add_argument('-bn','--batch_NORM',default='No',metavar="",required = False,type=str, help = "batch normalization", choices= ["Yes", "No"])
parser.add_argument('-da','--data_AUG',default='No',metavar="",required = False, type=str,help = "data augmentation", choices= ["Yes", "No"])
parser.add_argument('-fz','--filter_size',default=64,metavar="",required = False, type=int,help = "filter_size")
parser.add_argument('-fo','--filter_ORG',default="same",metavar="",required = False,type=str, help = "filter_organisation", choices= ["same","half","double"])

arg=parser.parse_args()

# Define data transformations
transform = transforms.Compose([
    transforms.Resize((256,256)),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

transform_augmented = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.AutoAugment(),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
])

#train_data_dir = 'path/to/your/training/data'
# Load dataset from directory
#if(arg.data_AUG=="No"):
#    dataset = datasets.ImageFolder(root=r'C:\Users\samruddhi\Downloads\nature_12K\inaturalist_12K\train', transform=transform)
#else:
#    dataset = datasets.ImageFolder(root=r'C:\Users\samruddhi\Downloads\nature_12K\inaturalist_12K\train', transform=transform_augmented)

if(arg.data_AUG=="No"):
    dataset = datasets.ImageFolder(root=r'C:\Users\samruddhi\Downloads\nature_12K\inaturalist_12K\train', transform=transform)
else:
    dataset = datasets.ImageFolder(root=r'C:\Users\samruddhi\Downloads\nature_12K\inaturalist_12K\train', transform=transform_augmented)

test_dataset = datasets.ImageFolder(root=r'C:\Users\samruddhi\Downloads\nature_12K\inaturalist_12K\val', transform=transform)


#test_dataset = datasets.ImageFolder(root=r'C:\Users\samruddhi\Downloads\nature_12K\inaturalist_12K\val', transform=transform)
#train_dataset = datasets.ImageFolder('/content/inaturalist_12K/train', transform=transform)
#test_dataset = datasets.ImageFolder('/content/inaturalist_12K/val', transform=transform)


# Split dataset into training and testing sets
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [int(0.8*len(dataset)), len(dataset)-int(0.8*len(dataset))])

# Create data loader objects for training and testing sets
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
#test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

class CNN_Train(pl.LightningModule):

  def __init__(self,activation_FUN,batch_NORM,data_AUG,filter_ORG,drop_OUT):


    self.activation_FUN=activation_FUN
    self.batch_NORM=batch_NORM


    #syntax:The syntax is torch.nn.Conv2d(in_channels, out_channels, kernel_size)

    super(CNN_Train,self).__init__()

    self.convo_1 = torch.nn.Conv2d(3,filter_ORG[0],3)
    self.convo_2 = torch.nn.Conv2d(filter_ORG[0], filter_ORG[1], 3)
    self.convo_3 = torch.nn.Conv2d(filter_ORG[1], filter_ORG[2], 3)
    self.convo_4 = torch.nn.Conv2d(filter_ORG[2], filter_ORG[3], 3)
    self.convo_5 = torch.nn.Conv2d(filter_ORG[3], filter_ORG[4], 3)

    if(activation_FUN=="ReLU"):
              self.activation_FUN=nn.ReLU()
    elif(activation_FUN=="GELU"):
             self.activation_FUN=nn.GELU()
    elif(activation_FUN=="SiLU"):
              self.activation_FUN=nn.SiLU()
    elif(activation_FUN=="Mish"):
             self.activation_FUN=nn.Mish()
    else:
             print("ERROR")

    stride=2
    input_dimension=256

    DenseLayerDimension = input_dimension
    for filter in filter_ORG:
      DenseLayerDimension = (DenseLayerDimension-4)//stride + 1


    self.f_batch = nn.BatchNorm1d(DenseLayerDimension*DenseLayerDimension*filter_ORG[4])
    self.maxpool= nn.MaxPool2d(2)
    self.flatten= nn.Flatten()
    self.fc_Layer= nn.Linear(DenseLayerDimension*DenseLayerDimension*filter_ORG[4],10)
    #self.fc_Layer = nn.Linear(filter_ORG[4] * input_dimension * input_dimension, 10)
    self.softmax= nn.Softmax()
    self.learning_rate=0.001
    self.s_dropout= nn.Dropout(p=drop_OUT)
    self.save_hyperparameters()

  def forward(self,x):

    output = self.activation_FUN(self.convo_1(x))
    output = self.maxpool(output)

    output = self.activation_FUN(self.convo_2(output))  # Apply activation after the convolution
    output = self.maxpool(output)

    output = self.activation_FUN(self.convo_3(output))  # Apply activation after the convolution
    output = self.maxpool(output)

    output = self.activation_FUN(self.convo_4(output))  # Apply activation after the convolution
    output = self.maxpool(output)

    output = self.activation_FUN(self.convo_5(output))  # Apply activation after the convolution
    output = self.maxpool(output)

    output = self.flatten(output)

    if(self.batch_NORM == "Yes"):
        output = self.f_batch(output)

    output = self.s_dropout(output)

    output = self.activation_FUN(self.fc_Layer(output))  # Apply activation to the fully connected layer
    return output

  def training_step(self, batch, batch_index):
    x, y = batch
    un_logits = self(x)
    loss = Funn.cross_entropy(un_logits,y)
    #self.train_loss.append(loss)
    accuracy = (un_logits.argmax(dim=1) == y).float().mean()
    #self.train_accuracy.append(accuracy)
    return loss

  def validation_step(self, batch, batch_index):
    x, y = batch
    un_logits = self(x)
    loss = Funn.cross_entropy(un_logits,y)
    #self.val_loss.append(loss)
    accuracy = (un_logits.argmax(dim=1) == y).float().mean()
    #self.val_accuracy.append(accuracy)


  def test_step(self, batch, batch_index):
    x, y = batch
    un_logits = self(x)
    loss = Funn.cross_entropy(un_logits,y)
    #self.log("test loss",loss, prog_bar=True)
    accuracy = (un_logits.argmax(dim=1) == y).float().mean()
    #self.log("test accuracy",accuracy,prog_bar=True)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    return optimizer

filter_size=64
if(arg.filter_ORG=="same"):
    filter_organisation1 = [arg.filter_size]*5
elif(arg.filter_ORG=="half"):
    filter_organisation1=[arg.filter_size,arg.filter_size//2,arg.filter_size//4,arg.filter_size//8,arg.filter_size//16]    
elif(arg.filter_ORG=="double"):
    filter_organisation1=[arg.filter_size,arg.filter_size*2,arg.filter_size*4,arg.filter_size*8,arg.filter_size*16]

obj = CNN_Train(arg.activation_FUN,arg.batch_NORM,arg.data_AUG,filter_organisation1,arg.drop_OUT)

trainer = pl.Trainer(max_epochs=arg.epochs) #, accelerator="gpu", devices=1)

trainer.fit(model=obj,train_dataloaders=train_dataloader,val_dataloaders=val_dataloader)


wandb.finish()
