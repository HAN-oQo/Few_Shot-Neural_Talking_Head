import torch
import torch.nn as nn
from models.blocks import *

class Discriminator(nn.Module):
    
    def __init__(self, batch_size= 16, curr_size = 224, ideal_size=256, fine_tuning = False, e_finetuning = None, pool_mode = "sum"):
        super(Discriminator, self).__init__()
        self.batch_size = batch_size

        self.fine_tuning = fine_tuning
        self.e_finetuning = None
        self.w_prime = nn.Parameter(torch.randn(512, 1))
        
        self.W_i = nn.Parameter(torch.randn(512, self.batch_size))
        self.w_0 = nn.Parameter(torch.randn(512, 1))
        self.b = nn.Parameter(torch.randn(1))
        
        self.init_padding= Padding(curr_size=curr_size, ideal=ideal_size)
        self.downsample0 = Residual_Downsample(6, 64)                     #output: B, 64, 128, 128
        self.downsample1 = Residual_Downsample(64, 128)                   #output: B, 128, 64, 64
        self.downsample2 = Residual_Downsample(128, 256)                  #output: B, 256, 32, 32
        self.down_attn = Self_Attention(256)                              #output: B, 256, 32, 32
        self.downsample3 = Residual_Downsample(256, 512)                  #output: B, 512, 16, 16
        self.downsample4 = Residual_Downsample(512, 512)                  #output: B, 512, 8, 8
        self.downsample5 = Residual_Downsample(512, 512)                  #output: B, 512, 4, 4
        self.additional_residual = Residual(512)                          #output: B, 512, 4, 4
        if pool_mode == "sum":
            self.sum_pooling = nn.LPPool2d(norm_type=1, kernel_size = 4)
        else:
            self.sum_pooling = nn.AdaptiveMaxPool2d((1,1))
        self.ReLU = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU()

    def load_W_i(self, W_i):
        self.W_i.data = self.LeakyReLU(W_i)
    
    def init_finetuning(self):
        if self.fine_tuning:
            self.w_prime = nn.Parameter(self.w_0 + torch.mean(self.e_finetuning, dim=0))
    
    def forward(self, img, landmark):
        B = img.size(0)
        new_input = torch.cat((img, landmark), dim=1)
        new_input = self.init_padding(new_input)
        
        
        out0 = self.downsample0(new_input)
        out1 = self.downsample1(out0)
        out2 = self.downsample2(out1)
        out3 = self.down_attn(out2)
        out4 = self.downsample3(out3)
        out5 = self.downsample4(out4)
        out6 = self.downsample5(out5)
        out7 = self.additional_residual(out6)
        out = self.sum_pooling(out7)
        out = self.ReLU(out)                #output: B, 512, 1,1
        out = out.view(-1, 512, 1)          #output: B, 512, 1

        if self.fine_tuning:
            score = torch.bmm(out.permute(0, 2, 1), (self.w_prime.unsqueeze(0).expand(B, 512, 1))) + self.b
        else:
            score = torch.bmm(out.permute(0, 2, 1), (self.W_i.unsqueeze(-1).permute(1, 0, 2) + self.w_0)) + self.b

        return score, [out0, out1, out2, out4, out5, out6, out7]