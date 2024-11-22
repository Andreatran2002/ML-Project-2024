# -*- coding: utf-8 -*-


# Thiết kế thêm ít nhất 
# 10 conv5x5(),
# 02 maxpooling(.), 
# 01 avgpooling(.), 
# 01 torch.cat(.), 
# 02 phép cộng 2 tensors, 
# và 01 fully connected layer (FC(.)).
# Số lượng parameter từ khoảng 10M to 20M.

import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, nclass=10):
        super(Net, self).__init__()
        # Initial convolution
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
        ##
        self.conv_down_1 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.conv_down_2 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=1)

        # 10 conv3x3 layers (32 filters each)
        self.conv5_1 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2) 
        self.conv5_1_1 = nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2) 
        ####
        self.conv5_2 = nn.Conv2d(64, 96, kernel_size=5, stride=2, padding=2)
        self.conv5_2_1 = nn.Conv2d(64, 96, kernel_size=5, stride=2, padding=2)
        ####
        self.conv5_3 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        self.conv5_3_1 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        self.conv5_3_2 = nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2)
        ####
        self.conv5_4 = nn.Conv2d(192, 384, kernel_size=5, stride=1, padding=2)
        self.conv3_5 = nn.Conv2d(384, 768, kernel_size=3, stride=2, padding=2)
        self.conv3_6 = nn.Conv2d(768, 1024, kernel_size=3, stride=2, padding=2)

        # Pooling layers
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.avgpool1 = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layers
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, nclass)
        
    def forward(self, x):
        x = self.initial_conv(x)  # 32 channels [32,224,224] 
        x = self.conv_down_1(x)  # 32 channels [32,112,112]  --------- [CONV5-1/10]  
        x = self.conv_down_2(x)  # 32 channels [32,56,56]  --------- [CONV5-2/10]  
        x1 = F.relu(self.conv5_1(x))    # [32, 56, 56] --------- [CONV5-3/10]  
        x2 = F.relu(self.conv5_1_1(x))   # [32, 56, 56] --------- [CONV5-4/10] 
        x = torch.cat([x1, x2], dim=1) # [64, 56, 56] --------- [CAT-1/1]  
        x = self.maxpool1(x) # [64, 28, 28]
        path_1_s1 = F.relu(self.conv5_2(x)) # [96, 28, 28] --------- [CONV5-5/10] 
        path_2_s1 = F.relu(self.conv5_2_1(x)) # [96, 28, 28] --------- [CONV5-6/10] 
        x = path_1_s1 + path_2_s1 # [96, 28, 28] --------- [PLUS-1/2]
        x = self.maxpool2(x) # [96, 14, 14]
        path_1_s1 = F.relu(self.conv5_3(x)) # [192, 14, 14] --------- [CONV5-7/10] 
        path_2_s1 = F.relu(self.conv5_3_1(x)) # [192, 14, 14] --------- [CONV5-8/10] 
        path_2_s1 = F.relu(self.conv5_3_2(x)) # [192, 14, 14] --------- [CONV5-9/10] 
        x = path_1_s1 + path_2_s1 # [192, 14, 14] --------- [PLUS-2/2]

        x = F.relu(self.conv5_4(x)) # [384, 14, 14]  --------- [CONV5-10/10] 
        x = F.relu(self.conv3_5(x)) # [1024, 7, 7]  
        x = F.relu(self.conv3_6(x)) # [1024, 7, 7]  

        x = self.avgpool1(x)   # [1024, 1, 1] --------- [AVGPOOL-2/2]

        print(x.shape)
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = Net(3)
# net.to(device)
# from torchsummary import summary
# summary(net, (3,224,224))