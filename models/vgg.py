# Reference for loading vgg_face.t7: https://www.programcreek.com/python/?code=prlz77%2Fvgg-face.pytorch%2Fvgg-face.pytorch-master%2Fmodels%2Fvgg_face.py

import torch
import torch.nn as nn
import torchfile
from torchvision.models import vgg19

import cv2

class VGG_19(nn.Module):
    def __init__(self, feature_idx=[1, 6, 11, 20, 29]):
        super(VGG_19, self).__init__()
        self.feature_idx = feature_idx
        self.vgg = vgg19(pretrained = True)
        features = list(self.vgg.features)
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        result = []
        for idx, model in enumerate(self.features):
            x = model(x)
            if idx in self.feature_idx:
                result.append(x)
        
        return result

        

class VGG_FACE(nn.Module):

    def __init__(self):
        super(VGG_FACE, self).__init__()
    

        self.block_size = [2, 2, 3, 3, 3]
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.ReLU()
        self.max1 = nn.MaxPool2d(2,2)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.ReLU()
        self.max2 = nn.MaxPool2d(2,2)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU()
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.ReLU()
        self.max3 = nn.MaxPool2d(2,2)
        
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.ReLU()
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_3 = nn.ReLU()
        self.max4 = nn.MaxPool2d(2,2)

        self.conv5_1 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu5_2 = nn.ReLU()
        self.conv5_3 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu5_3 = nn.ReLU()
        self.max5 = nn.MaxPool2d(2,2)

        self.fc6 = nn.Linear(512*7*7, 4096)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)
        self.fc8 = nn.Linear(4096, 2622)

        
    def load_weights(self, pretrained_path = "./pretrained/vgg_face_torch/VGG_FACE.t7"):
        
        model = torchfile.load(pretrained_path)
        block = 1
        sub_block = 1

        for i, layer in enumerate(model.modules):
            if layer.weight is not None:
                if block <= 5:
                    # Conv layer
                    self_layer = getattr(self, "conv{}_{}".format(block, sub_block))
                    sub_block +=1
                    if sub_block > self.block_size[block -1]:
                        block += 1
                        sub_block = 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
                else:
                    self_layer = getattr(self, "fc{}".format(block))
                    block += 1
                    self_layer.weight.data[...] = torch.tensor(layer.weight).view_as(self_layer.weight)[...]
                    self_layer.bias.data[...] = torch.tensor(layer.bias).view_as(self_layer.bias)[...]
        self.eval()

    def forward(self, x):
        # 1, 6, 11, 18, 25
        result = []
        x = self.conv1_1(x)
        x = self.relu1_1(x)
        result.append(x)
        x = self.conv1_2(x)
        x = self.relu1_2(x)
        x = self.max1(x)

        x = self.conv2_1(x)
        x = self.relu2_1(x)
        result.append(x)
        x = self.conv2_2(x)
        x = self.relu2_2(x)
        x = self.max2(x)


        x = self.conv3_1(x)
        x = self.relu3_1(x)
        result.append(x)
        x = self.conv3_2(x)
        x = self.relu3_2(x)
        x = self.conv3_3(x)
        x = self.relu3_3(x)
        x = self.max3(x)

        x = self.conv4_1(x)
        x = self.relu4_1(x)
        result.append(x)
        x = self.conv4_2(x)
        x = self.relu4_2(x)
        x = self.conv4_3(x)
        x = self.relu4_2(x) 
        x = self.max4(x)

        x = self.conv5_1(x)
        x = self.relu5_1(x)
        result.append(x)
        x = self.conv5_2(x)
        x = self.relu5_2(x)
        x = self.conv5_3(x)
        x = self.relu5_3(x)
        x = self.max5(x)

        x = x.view(x.size(0), -1)
        x = self.relu6(self.fc6(x))
        x = self.dropout1(x)
        x = self.relu7(self.fc7(x))
        x = self.dropout2(x)
        x = self.fc8(x)
    
        return result

        



