
# Thiết kế thêm ít nhất 
# 10 conv3x30, 
# 10 conv5x50, 02 maxpooling(.), 
# 02 avgpooling(.), 
# 04 torch.cat(.), 
# 04 phép cộng 2 tensors, 
# và 02 fully connected layer
# (FC(.)). Số lượng parameter từ khoảng 10M to 20M.:
import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, nclass=10):
        super(Net, self).__init__()
        # Initial convolution
        self.initial_conv = nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
        
        # 10 conv3x3 layers (32 filters each)
        self.conv3_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_1 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)

        self.conv3_2 = nn.Conv2d(160, 64, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(64, 128, kernel_size=5, stride=1, padding=2)
        
        # Split path 1

        self.conv3_3 = nn.Conv2d(288, 32, kernel_size=3, stride=1, padding=1)
        self.conv3_4 = nn.Conv2d(288, 32, kernel_size=3, stride=1, padding=1)

        self.conv5_3 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.conv5_4 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)

        # Split path 2

        self.conv3_5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_6 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.conv5_5 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv5_6 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)


        self.conv3_7 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv3_8 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)

        self.conv5_7 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv5_8 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)

        self.conv5_9 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2)
        self.conv5_10 = nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2)



        self.conv3_9 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)




        # Pooling layers
        self.maxpool1 = nn.MaxPool2d(2, 2)
        self.maxpool2 = nn.MaxPool2d(2, 2)
        self.avgpool1 = nn.AvgPool2d(2, 2)
        self.avgpool2 = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(1024, 2048)
        self.fc2 = nn.Linear(2048, nclass)
        
    def forward(self, x):
        x = self.initial_conv(x)  # 32 channels
        
        x1 = F.relu(self.conv3_1(x))    # [64, 224, 224]
        x1 = F.relu(self.conv5_1(x1))   # [128, 224, 224]

        x = torch.cat([x1, x], dim=1) # [160, 224, 224] --------- [CAT-1/4]
        x = self.maxpool1(x)   # [160, 112, 112]  --------- [MAX_POOL-1/2]
        

        x2 = F.relu(self.conv3_2(x)) # [64, 112, 112]
        x2 = F.relu(self.conv5_2(x2)) # [128, 112, 112]

        x = torch.cat([x2, x], dim=1) # [288, 112, 112] --------- [CAT-2/4]
        x = self.maxpool2(x)   # [288, 56, 56]  --------- [MAX_POOL-2/2]

        path_1_s1 = F.relu(self.conv3_3(x)) # [32, 56, 56]
        path_2_s1 = F.relu(self.conv3_4(x)) # [32, 56, 56]

        path_1_s1 = F.relu(self.conv5_3(path_1_s1)) # [64, 56, 56]
        path_2_s1 = F.relu(self.conv5_4(path_2_s1)) # [64, 56, 56]

        x = path_1_s1 + path_2_s1 # [64, 56, 56] --------- [PLUS-1/4]

        x = self.avgpool1(x)  # 64, 28, 28  --------- [AVG_POOL-1/2]

        path_1_s2 = F.relu(self.conv3_5(x)) # [256, 28, 28]
        path_2_s2 = F.relu(self.conv3_6(x)) # [256, 28, 28]

        path_1_s2 = F.relu(self.conv5_5(path_1_s2)) # [256, 28, 28]
        path_2_s2 = F.relu(self.conv5_6(path_2_s2)) # [256, 28, 28]

        x = path_1_s2 + path_2_s2 # [256, 28, 28] --------- [PLUS-2/4]


        path_1_s3 = F.relu(self.conv3_7(x)) # [128, 28, 28]
        path_2_s3 = F.relu(self.conv3_8(x)) # [128, 28, 28]

        path_1_s3 = F.relu(self.conv5_7(path_1_s3)) # [256, 28, 28]
        path_2_s3 = F.relu(self.conv5_8(path_2_s3)) # [256, 28, 28]

        path_1_s3 = F.relu(self.conv5_9(path_1_s3)) # [256, 28, 28]
        path_2_s3 = F.relu(self.conv5_10(path_2_s3)) # [256, 28, 28]

        x1 = path_1_s3 + path_2_s3 # [256, 28, 28] --------- [PLUS-3/4]

        x = torch.cat([x1, x], dim=1) # [512, 28, 28] --------- [CAT-3/4]

        x1 =  F.relu(self.conv3_9(x1)) # [512, 28, 28]
        x = x + x1 ;  # [512, 28, 28] --------- [PLUS-4/4]

        x = torch.cat([x, x], dim=1) # [1024, 28, 28] --------- [CAT-4/4]

        print(x.shape)
        
        x = self.avgpool2(x) #[AVG_POOL-2/2]
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# net = Net(3)
# net.to(device)
# from torchsummary import summary
# summary(net, (3,224,224))