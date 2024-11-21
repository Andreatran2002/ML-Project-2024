# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:40:28 2024

@author: Group3
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self,nclass=10):
        super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 6, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # self.fc2 = nn.Linear(120, 84)
        # self.fc3 = nn.Linear(84, 10)
        self.conv0 = nn.Conv2d(3,16,5,stride=2,padding=2)
        
        self.conv1 = nn.Conv2d(16,32,3)
        self.conv2 = nn.Conv2d(32,64,5,padding=2)
        # split
        # x1
        self.conv3 = nn.Conv2d(32,32,3,padding=1,stride=2)
        self.conv4 = nn.Conv2d(32,64,3,padding=1)
        # x2
        self.conv5 = nn.Conv2d(32, 64, 5, padding=2, stride=2)
        self.conv6 = nn.Conv2d(64, 128, 3, padding=1, stride=1)
        self.conv7 = nn.Conv2d(384,512,5)
        self.conv71 = nn.Conv2d(512,512,3,stride = 2)
        self.conv8 = nn.Conv2d(512,1024,3)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(1024,100)
        self.fc2 = nn.Linear(100,nclass)
    def forward(self, x):
        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 16 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = self.fc3(x)
        x = F.relu(self.conv0(x))
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)

        s = x.size()
        x1 = x[:, :int(s[1] / 2), :, :]
        x2 = x[:, int(s[1] / 2):, :, :]
        x1 = self.conv3(x1)
        x1 = F.relu(x1)
        x1 = self.conv4(x1)
        x2 = self.conv5(x2)
        x2 = F.relu(x2)
        a = x2
        x2 = self.conv6(x2)
        cat1 = torch.cat([x1,x2],dim=1)
        plus1 = a + x1
        cat2 = torch.cat([cat1,plus1], dim=1)
        cat3 = torch.cat([x2,cat2], dim=1)
        cat3 = self.conv7(cat3)
        cat3 = F.relu(cat3)
        cat3 = F.relu(self.conv71(cat3))
        cat3 = self.conv8(cat3)
        cat3 = self.avgpool(cat3)
        cat3 = cat3.view(-1,1024)
        cat3 = self.fc1(cat3)
        cat3 = self.fc2(cat3)


        return cat3