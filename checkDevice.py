# -*- coding: utf-8 -*-
import torch
print(torch.__version__)

#print(torch.cuda.is_available())  # Kiểm tra xem CUDA có khả dụng không
#print(torch.cuda.current_device())  # In ra ID của thiết bị CUDA hiện tại
#print(torch.cuda.get_device_name(0))  # In ra tên của GPU