#Code base on this repo https://github.com/aksh-ai/neuralBlack/tree/28701d5f02d5a3584541258829c0ed1054d8d4e2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import os
import random
import numpy as np
import pandas as pd
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, jaccard_similarity_score
from Dataproc import ABIDE
import tqdm

import argparse


# Arguments
parser = argparse.ArgumentParser(description='Autism classification')
parser.add_argument('--mode', type=str, default='test', choices=['test', 'demo'],
                    help ='model segmentation for testing (default: resnet50')

parser.add_argument('--img', type=str, default=0,
                    help='Number of image')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA testing')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

std = [0.229, 0.224, 0.225]
mean = [0.485, 0.456, 0.406]


transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ])
'''transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
                transforms.Scale(256,256),
                transforms.CenterCrop(224,224)
            ])'''
targetTransform = transforms.Compose([
                transforms.ToTensor()
            ])
            
def eval_metric(lbl_pred, lbl_true):

        # Over-all accuracy
        # TODO: per-class accuracy
        accu = []
        for lt, lp in zip(lbl_true, lbl_pred):
            accu.append(np.mean(lt == lp))
        return np.mean(accu)

resnet_model = models.resnet50(pretrained=True)

# set all paramters as trainable
for param in resnet_model.parameters():
    param.requires_grad = True

# get input of fc layer
n_inputs = resnet_model.fc.in_features

# redefine fc layer / top layer/ head for our classification problem
resnet_model.fc = nn.Sequential(nn.Linear(n_inputs, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2048),
                                nn.SELU(),
                                nn.Dropout(p=0.4),
                                nn.Linear(2048, 2),
                                nn.LogSigmoid())

# set all paramters of the model as trainable
#for name, child in resnet_model.named_children():
 # for name2, params in child.named_parameters():
  #  params.requires_grad = True

# set model to run on GPU or CPU absed on availibility
resnet_model.cuda()

# print the trasnfer learning NN model's architecture
resnet_model
test_dataset = ABIDE(split='test', transform=transform)
if args.mode =='test':
    test_gen = DataLoader(test_dataset, batch_size=64, shuffle=True, pin_memory=True, num_workers=8)
if args.mode =='demo':
    test_gen = DataLoader(test_dataset[int(args.img)], batch_size=64, shuffle=True, pin_memory=True, num_workers=8)

resnet_model.load_state_dict(torch.load('/media/disk1/user_home1/bacuestas/intento/bt_resnet50_model4.pt'))
train_gen = None
valid_gen = None
train_set = None
valid_set = None

# set model to evaluation mode
resnet_model.eval()
criterion = nn.CrossEntropyLoss().cuda()
activation = nn.Softmax()
# perform no gradient updates
with torch.no_grad():
    # soem metrics storage for visualization and analysis
    correct = 0
    test_loss = []
    test_corr = []
    labels = []
    pred = []
    # perform test set evaluation batch wise
    for (X, y) in test_gen:
        # set label to use CUDA if available
        X, y = X.cuda(), y.cuda()
        target = y
        X = torch.permute(X,(0,3,1,2))
        # forward pass image
        #y_val = resnet_model(X.view(-1, 3, 512, 512))
        y_val = resnet_model(X.float())
        # append original labels
        labels.append(target)
        y = nn.functional.one_hot(y, num_classes=2)
        # perform forward pass
        #y_val = resnet_model(X.view(-1, 3, 512, 512))

        # get argmax of predicted values, which is our label
        #predicted = torch.argmax(y_val, dim=1).data
        lbl_pred = y_val.data.max(1)[1].cpu().numpy()
        lbl_pred = lbl_pred.squeeze()
        lbl_true = target.data.cpu()
        lbl_true = np.squeeze(lbl_true.numpy())
        # append predicted label
        pred.append(lbl_pred)

        # calculate loss
        #loss = criterion(y_val.float(), torch.argmax(y.view(10 * 8, 4), dim=1).long())
        loss = criterion(activation(y_val.squeeze()), y.float())
        # increment correct with correcly predicted labels per batch
        #correct += (predicted == torch.argmax(y.view(10 * 8, 4), dim=1)).sum()
        correct += (lbl_pred == lbl_true).sum()
        # append correct samples labels and losses
        test_corr.append(correct)
        test_loss.append(loss)
        
print(f"Test Loss: {test_loss[-1].item():.4f}")
ACC=eval_metric(pred[0],labels[0].cpu().numpy())
print(f'Test accuracy: {ACC}')
#breakpoint()
'''
#labels = torch.stack(labels)
#pred = torch.stack(pred)

LABELS = ['Autism', 'No-Autism']

arr = confusion_matrix(pred[0], labels[0].cpu().numpy())
df_cm = pd.DataFrame(arr, LABELS, LABELS)
#plt.figure(figsize = (9,6))
plt.figure()
sns.heatmap(df_cm, annot=True, fmt="d", cmap='viridis')
plt.xlabel("Prediction")
plt.ylabel("Target")
plt.savefig('/media/disk1/user_home1/bacuestas/intento/confusion matriz.png')

print(f"Clasification Report\n\n{classification_report(pred.view(-1).cpu(), labels.view(-1).cpu())}")'''
