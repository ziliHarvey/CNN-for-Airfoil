#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:00:13 2018

@author: zili
"""
########################################################################################################
###This file loads parsed_data, builds a 6-layer convolutional neural network, and prints the results###
###Data is obtained by running raw_data_parsing.py on airfoil figures and CFD-calculated cl/cd values###
###Data: this file used a sample 1_300.mat, which means #1~#300 airfoil data, which includes:        ###
###data_x ( a 6855*16384 binary matrix), data_y (a 6855*1 matrix), and a normalization factor (309)  ###
###Structure: a well-tuned 4-conv-layer followed by 2-fc-layer network, with trick of batch norm, etc###
###Training: train and test with GPU on Alienware with GTX 1080 Ti graphics                          ###
###MSE result: train loss is 0.06415, validation/test loss is 0.36484 after 200 epochs               ###
###With a well-trained cnn, cl/cd prediction speed can be 5k X faster than matured CFD software      ###
###Please modify loading for more data (totally around 1550 foil types, only 1~300 is shown here)    ###
########################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import scipy.io
import time


data = scipy.io.loadmat('../data/parsed_data/1_300.mat')
data_x, data_y, rNorm = data['data_x'], data['data_y'], data['Normalization_Factor']
num_data = np.shape(data_x)[0]
train_x, train_y = data_x[:int(0.7*num_data)], data_y[:int(0.7*num_data)]
valid_x,valid_y = data_x[int(0.2*num_data):int(0.9*num_data)], data_y[int(0.2*num_data):int(0.9*num_data)]
test_x, test_y = data_x[int(0.9*num_data):], data_y[int(0.9*num_data):]


class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
       
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,13),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,7),
            #nn.Dropout2d(0.5),
            nn.BatchNorm2d(20),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20,40,7),
            nn.BatchNorm2d(40),
            #nn.Dropout2d(0.5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(40,80,5),
            nn.BatchNorm2d(80),
            #nn.Dropout2d(0.5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(720,400),
            nn.ReLU(),
            #nn.Dropout(0.5)
        )
        self.fc2 = nn.Linear(400,1)
    
        

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f4_flat = f4.view(f4.size(0), -1)
        f5 = self.fc1(f4_flat)
        output = self.fc2(f5)
        return output 
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#set up parameters
batch_size = 50
learning_rate = 0.00001
num_epochs = 30

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
valid_x = torch.from_numpy(valid_x).float()
valid_y = torch.from_numpy(valid_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()


train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
valid_dataset = torch.utils.data.TensorDataset(valid_x, valid_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)

train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
valid_dataloader = torch.utils.data.DataLoader(
    dataset=valid_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset
)
neural_net = Net1().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(neural_net.parameters(),learning_rate)
lossList = []
accList = []
valid_lossList = []
valid_accList = []

for epoch in range(num_epochs):
    loss_sum_train = 0
    loss_sum_valid = 0
    acc_sum = 0
    
    for iteration, (images, labels) in enumerate(train_dataloader):
        x_batch = torch.autograd.Variable(images)
        x_batch = x_batch.reshape(-1,1,128,128)
        y_batch = torch.autograd.Variable(labels)
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        
        output = neural_net(x_batch)
        loss = criterion(output,y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = abs((output.cpu() - labels)/labels).mean()
        acc_sum += acc
        loss_sum_train += loss.cpu().data.numpy()
    acc_epoch = acc_sum/len(train_dataloader.dataset)
    lossList.append(loss_sum_train)
    accList.append(acc_epoch)
    
    for (images, labels) in valid_dataloader:
        labels = torch.autograd.Variable(labels)
        images = torch.autograd.Variable(images)
        images = images.reshape(-1,1,128,128)
        images = images.to(device)
        labels = labels.to(device)
        output_valid = neural_net(images)
        loss = criterion(output_valid,labels)
        loss_sum_valid += loss.cpu().data.numpy()
    valid_lossList.append(loss_sum_valid)
    print('Epoch: ', epoch, '| train loss: %.6f | valid loss: %.6f  ' % (loss_sum_train,loss_sum_valid))

    


plt.figure()
line1, = plt.plot(range(num_epochs),lossList,label='Train Loss')
line2, = plt.plot(range(num_epochs),valid_lossList,label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend([line1,line2],['Train Loss','Validation Loss'])
plt.title('Train & Valid Loss v/s Epoch')

#test
start = time.time()        
num = len(test_dataloader)
predyList = []
testyList = []
for (images, labels) in test_dataloader:
    images = torch.autograd.Variable(images)
    images = images.reshape(-1,1,128,128)
    pred = neural_net(images.to(device))
    predy = pred
    predyList.append(predy)
    testy = labels
    testyList.append(testy)
end = time.time()
elapsed = end -start
print('The time elapsed %.6f'%elapsed)

#denormalize
predyList = [x*rNorm[0,0] for x in predyList]
testyList = [x*rNorm[0,0] for x in testyList]
#plot result
plt.figure()
line3, = plt.plot(range(len(predyList)), predyList, alpha = 0.8,label = 'Predicted')
line4, = plt.plot(range(len(testyList)), testyList, label = 'GroundTruth')
plt.ylim(-100,150)
plt.legend([line3,line4],['Predicted','GroundTruth'])
plt.title(' Test & Predicted Cl/Cd Ratio')

#plot result(zoom in)
plt.figure()
line5, = plt.plot(range(len(predyList)), predyList, alpha = 0.8,label = 'Predicted')
line6, = plt.plot(range(len(testyList)), testyList, label = 'GroundTruth')
plt.ylim(-100,150)
plt.xlim(400,500)
plt.legend([line5,line6],['Predicted','GroundTruth'])
plt.title(' Test & Predicted Cl/Cd Ratio (Zoom In)')
#confusion matrix
pre = [x.cpu().data.numpy() for x in predyList]
test = [x.cpu().data.numpy() for x in testyList]
plt.figure()
plt.scatter(pre, test,s=1)
plt.plot([-150,150],[-150,150], ls="--",c=".3")
plt.plot([-150,135],[-135,150], ls="--",c=".3")
plt.plot([-135,150],[-150,135], ls="--",c=".3")
plt.xlabel('Predicted Cl/Cd Ratio')
plt.ylabel('Actual Cl/Cd Ratio')
plt.xlim(-50,150)
plt.ylim(-50,150)
plt.title(' Test & Predicted confusion matrix')
plt.show()
