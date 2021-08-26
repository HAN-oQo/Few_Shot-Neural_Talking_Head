import torch
import torch.nn
from models.blocks import *

class Embedder(nn.Module):

    def __init__(self,curr_size=224, ideal_size=256, pool_mode = "sum"):
        super(Embedder, self).__init__()

        self.init_padding= Padding(curr_size=curr_size, ideal=ideal_size)

        # downsample
        # input: B, 6, 256, 256
        if pool_mode != "sum":
            self.emb = nn.Sequential(
                Residual_Downsample(6, 64),                     #output: B, 64, 128, 128
                Residual_Downsample(64, 128),                   #output: B, 128, 64, 64
                Residual_Downsample(128, 256),                  #output: B, 256, 32, 32
                Self_Attention(256),                            #output: B, 256, 32, 32
                Residual_Downsample(256, 512),                  #output: B, 512, 16, 16
                Residual_Downsample(512, 512),                  #output: B, 512, 8, 8
                Residual_Downsample(512, 512),                  #output: B, 512, 4, 4
                nn.AdaptiveMaxPool2d((1,1)),
                nn.ReLU()
            )
        else:
            self.emb = nn.Sequential(
                Residual_Downsample(6, 64),                     #output: B, 64, 128, 128
                Residual_Downsample(64, 128),                   #output: B, 128, 64, 64
                Residual_Downsample(128, 256),                  #output: B, 256, 32, 32
                Self_Attention(256),                            #output: B, 256, 32, 32
                Residual_Downsample(256, 512),                  #output: B, 512, 16, 16
                Residual_Downsample(512, 512),                  #output: B, 512, 8, 8
                Residual_Downsample(512, 512),                  #output: B, 512, 4, 4
                nn.LPPool2d(norm_type=1, kernel_size = 4),
                nn.ReLU()
            )
            


    def forward(self, img, landmark):
        new_input = torch.cat((img, landmark), dim = 1)
        new_input = self.init_padding(new_input)

        out = self.emb(new_input)
        out = out.view(out.size(0), 512, 1)
        return out