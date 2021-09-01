import torch
import torch.nn as nn
from models.blocks import *
from itertools import accumulate
import math

class Generator(nn.Module):

    def __init__(self, curr_size=224, ideal_size=256, bottleneck_num = 5,  fine_tuning= False ):
        super(Generator, self).__init__()
        
        slice_idx = [0]
        for i in range(bottleneck_num):
            slice_idx.append(512*4)
        slice_idx = slice_idx + [512*2 + 256*2,  #resUp1
                                256*2 + 128*2,  #resUp2
                                128*2 + 64*2,   #resUp3
                                64*2 + 32*2,    #resUp4
                                32*2]           #last adain
        self.slice_idx = list(accumulate(slice_idx))
        self.P_len = self.slice_idx[-1]
        self.P = nn.Parameter(torch.randn(self.P_len, 512).normal_(0.0, 0.02))
        self.psi = nn.Parameter(torch.randn(self.P_len, 512)) #used in fine_tuning stage

        self.fine_tuning = fine_tuning
        # self.e_finetuning = e_finetuning


        self.init_padding = Padding(curr_size=curr_size, ideal = ideal_size)

        # Downsample
        # input B, 3, 256, 256
        self.downsample = nn.Sequential(

            Residual_Downsample(3, 64, kernel_size = 9, stride = 1, padding = 4),
            nn.InstanceNorm2d(64, affine = True),
            # output B, 64, 128, 128

            Residual_Downsample(64, 128),
            nn.InstanceNorm2d(128, affine = True),
            # output B, 128, 64, 64

            Residual_Downsample(128, 256),
            nn.InstanceNorm2d(256, affine = True),
            # output B, 256, 32, 32

            Self_Attention(256),
            # output B, 256, 32, 32

            Residual_Downsample(256,512),
            nn.InstanceNorm2d(512, affine = True),
            # output B, 512, 16, 16
        )

        # Bottleneck
        self.bottleneck_num = bottleneck_num
        if bottleneck_num == 4:
            self.bottleneck = Bottleneck4(512)
        elif bottleneck_num == 5:
            self.bottleneck = Bottleneck5(512)


        # Upsample
        # input B, 512, 16, 16       
        self.upsample0 = Adaptive_Residual_Upsample(512,256) # output B, 256, 32, 32
        self.upsample1 = Adaptive_Residual_Upsample(256,128) # output B, 128, 64, 64
        self.up_attn =  Self_Attention(128)  # output B, 128, 64, 64         
        self.upsample2 = Adaptive_Residual_Upsample(128, 64) # output B, 64, 128, 128
        self.upsample3 = Adaptive_Residual_Upsample(64, 32, out_size = curr_size) # output B, 32, 224, 224


        # Last layer of Generator
        self.last_adain = AdaIN()
        self.last_layer = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, stride=1, padding =1),
            # output B, 3, 224, 224
            nn.Tanh()
        )



    def init_finetuning(self, e_finetuning):
        if self.fine_tuning:
            self.psi = nn.Parameter(torch.mm(self.P, torch.mean(e_finetuning, dim=0)))   # (p_len, 512) * (512, 1) = p_len* 1
    
    def forward(self, landmark, e):
        B = e.size(0)
        if math.isnan(self.P[0,0]):
            raise(RuntimeError("Wrong Parameter Matrix P"))
        
        if self.fine_tuning:
            style = self.psi.unsqueeze(0)
            style = style.expand(B, self.P_len, 1) # B, p_len, 1
        else:
            P = self.P.unsqueeze(0) # 1, p_len, 512
            P = P.expand(B, self.P_len, 512) # B, p_len, 512
            style = torch.bmm(P, e) # (B, p_len, 512) * (B, 512, 1) = B, p_len, 1

        style_bottleneck = style[:,self.slice_idx[0]:self.slice_idx[self.bottleneck_num],:]
        style_up0 = style[:,self.slice_idx[self.bottleneck_num]:self.slice_idx[self.bottleneck_num+1],:]
        style_up1 = style[:,self.slice_idx[self.bottleneck_num+1]:self.slice_idx[self.bottleneck_num+2],:]
        style_up2 = style[:,self.slice_idx[self.bottleneck_num+2]:self.slice_idx[self.bottleneck_num+3],:]
        style_up3 = style[:,self.slice_idx[self.bottleneck_num+3]:self.slice_idx[self.bottleneck_num+4],:]
        style_last_mu = style[:,self.slice_idx[self.bottleneck_num+4]:(self.slice_idx[self.bottleneck_num+4]+self.slice_idx[self.bottleneck_num+5])//2,:]
        style_last_std = style[:,(self.slice_idx[self.bottleneck_num+4]+self.slice_idx[self.bottleneck_num+5])//2:self.slice_idx[self.bottleneck_num+5],:]

        new_input = self.init_padding(landmark) 

        #Downsample
        out = self.downsample(new_input)

        #Bottleneck
        out = self.bottleneck(out, style_bottleneck)

        #Upsample
        out = self.upsample0(out, style_up0)
        out = self.upsample1(out, style_up1)
        out = self.up_attn(out)
        out = self.upsample2(out, style_up2)
        out = self.upsample3(out, style_up3)

        #last layer
        out = self.last_adain(out, style_last_mu, style_last_std)
        out = self.last_layer(out)


        return out
        

    





        
        
        

        