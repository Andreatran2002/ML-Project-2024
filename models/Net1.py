# -*- coding: utf-8 -*-


#Thiết kế thêm ít nhất 10 conv3x3(),
# 01 maxpooling(.), 
# 02 avgpooling(.), 
# 02 torch.cat(.), 
# 02 phép cộng 2 tensors, 
# và 02 fully connected layer (FC(.)). 
# Số lượng parameter từ khoảng 10M to 20M.

import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, nclass=10):
        super(Net, self).__init__()
        # Initial convolution
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
        self.conv_down_1 = nn.Conv2d(32, 32, kernel_size=5, stride=2, padding=2)
        self.conv_down_2 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)

        # 10 conv3x3 layers (32 filters each)
        ###
        self.conv3_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) 
        self.conv3_1_1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1) 
        ###
        self.conv3_2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        ##
        self.conv3_3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)

        ##
        self.conv3_4 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv5_4 = nn.Conv2d(256, 128, kernel_size=5, stride=1, padding=2)
        ##
        self.conv3_6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5_6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        #
        self.conv3_7 = nn.Conv2d(256,512 , kernel_size=3, stride=1, padding=1)
        self.conv5_7 = nn.Conv2d(256,512 , kernel_size=5, stride=1, padding=2)
        self.conv3_8 = nn.Conv2d(512,512 , kernel_size=3, stride=1, padding=1)

        # Pooling layers
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.avgpool1 = nn.AvgPool2d(2, 2)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, nclass)
        
    def forward(self, x):
        x = self.initial_conv(x)  # 32 channels [32,224,224]  
        x = self.conv_down_1(x)  # 32 channels [32,112,112]    
        x = self.conv_down_2(x)  # 32 channels [32,56,56]    --------- [CONV3-1/10] 

        x1 = F.relu(self.conv3_1(x))    # [32, 56, 56] --------- [CONV3-2/10]  
        x2 = F.relu(self.conv3_1_1(x))   # [32, 56, 56] --------- [CONV3-3/10] 
        x = torch.cat([x1, x2], dim=1) # [64, 56, 56] --------- [CAT-1/2]    
        x = self.maxpool1(x)   # [64, 28, 28]  --------- [MAX_POOL-1/1]
        x = self.conv3_2(x)  # 32 channels [128,28,28] --------- [CONV3-4/10] 
        x = self.conv3_2_2(x)  # 32 channels [128,28,28] --------- [CONV3-5/10] 

        path_1 = F.relu(self.conv3_3(x)) # [256, 28, 28] --------- [CONV3-6/10] 
        path_2 = F.relu(self.conv5_3(x)) # [256, 28, 28] 
        x = path_1 + path_2 # [256, 28, 28] --------- [PLUS-1/2]
        x = self.maxpool2(x)   # [256, 14, 14]  --------- [MAX_POOL-1/1]

        path_1 = F.relu(self.conv3_4(x)) # [128, 14, 14] --------- [CONV3-7/10] 
        path_2 = F.relu(self.conv5_4(x)) # [128, 14, 14] 
        
        x = path_1 + path_2 # [128, 14, 14] --------- [PLUS-2/2]
        x = self.avgpool1(x)  # [128, 7, 7] --------- [AVGPOOL-1/2]
        
        x1 = F.relu(self.conv3_6(x)) # [256, 7, 7]  --------- [CONV3-8/10] 
        x2 = F.relu(self.conv5_6(x)) # [256, 7, 7]  
        x1 = F.relu(self.conv3_7(x1)) # [512, 7, 7]  --------- [CONV3-9/10] 
        x1 = F.relu(self.conv3_8(x1)) # [512, 7, 7]  --------- [CONV3-9/10] 
        x2 = F.relu(self.conv5_7(x2)) # [512, 7, 7]  
        x = torch.cat([x1, x2], dim=1) # [1024, 7, 7] --------- [CAT-2/2]

        
        x = self.avgpool2(x)   # [1024, 1, 1] --------- [AVGPOOL-2/2]

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