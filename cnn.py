#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 14:00:13 2018

@author: zili
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 17 16:09:33 2018

@author: zili
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
import scipy.io

#load data
#train_data = scipy.io.loadmat('../data/nist36_train.mat')
#test_data = scipy.io.loadmat('../data/nist36_test.mat')
##valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
#
#train_x, train_y = train_data['train_data'], train_data['train_labels']
#test_x, test_y = test_data['test_data'], test_data['test_labels']
##valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
#train_x = np.random.random((1000,16384))
#train_y = np.random.random((1000,1))
#test_x = train_x
#test_y = train_y

data = scipy.io.loadmat('./1_100.mat')
data_x, data_y, rNorm = data['data_x'], data['data_y'], data['Normalization_Factor']

train_x, train_y = data_x[:1780], data_y[:1780]
test_x, test_y = data_x[1780:], data_y[1780:]

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1,10,13),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10,20,7),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(20,40,7),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(40,80,5),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(2,2),
            nn.ReLU()
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(720,400),
            nn.ReLU(),
            nn.Dropout(0.5)
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
#set up parameters
batch_size = 50
learning_rate = 0.001
num_epochs = 10

train_x = torch.from_numpy(train_x).float()
train_y = torch.from_numpy(train_y).float()
test_x = torch.from_numpy(test_x).float()
test_y = torch.from_numpy(test_y).float()
print(test_y.shape)
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
train_dataloader = torch.utils.data.DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True
)
test_dataloader = torch.utils.data.DataLoader(
    dataset=test_dataset
)
neural_net = Net1()

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(neural_net.parameters(),learning_rate)
lossList = []
accList = []
for epoch in range(num_epochs):
    loss_sum_train = 0
    acc_sum = 0
    for iteration, (images, labels) in enumerate(train_dataloader):
        x_batch = torch.autograd.Variable(images)
        x_batch = x_batch.reshape(-1,1,128,128)
        y_batch = torch.autograd.Variable(labels)
#        print(y_batch.shape)
        output = neural_net(x_batch)
#        print(output.shape)
        loss = criterion(output,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = abs((output - labels)/labels).mean()
        acc_sum += acc
        loss_sum_train += loss.data.numpy()
    acc_epoch = acc_sum/len(train_dataloader.dataset)
    lossList.append(loss_sum_train)
    accList.append(acc_epoch)
    print('Epoch: ', epoch, '| train loss: %.4f' % loss_sum_train)

plt.figure()
plt.plot(range(num_epochs),lossList)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Train Loss v/s Epoch')
        
num = len(test_dataloader)
predyList = []
testyList = []
for (images, labels) in test_dataloader:
    images = torch.autograd.Variable(images)
    images = images.reshape(-1,1,128,128)
    pred = neural_net(images)
    predy = pred
    predyList.append(predy)
    testy = labels
    testyList.append(testy)

#denormalize
predyList = [x*rNorm[0,0] for x in predyList]
testyList = [x*rNorm[0,0] for x in testyList]
#plot result
plt.figure()
plt.plot(range(len(predyList)), predyList, label = 'Predict')
plt.plot(range(len(testyList)), testyList, label = 'Accurate')

#confusion matrix
pre = [x.data.numpy() for x in predyList]
test = [x.data.numpy() for x in testyList]
plt.figure()
plt.scatter(pre, test)
plt.xlabel('Predicted Cl/Cd Ratio')
plt.ylabel('Actual Cl/Cd Ratio')
