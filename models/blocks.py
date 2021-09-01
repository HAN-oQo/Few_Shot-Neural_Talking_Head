from cv2 import pyrDown
from matplotlib.pyplot import sca
import torch
import torch.nn as nn
from torch.nn.modules.pooling import AvgPool2d

class AdaIN(nn.Module):
    def __init__(self, eps = 1e-5):
        super(AdaIN, self).__init__()
        self.eps = eps

    def forward(self, input, style_mu, style_std):
        B, C, H, W = input.size()
        feat = input.view(B, C, -1)
        feat_std = (torch.std(feat, dim=-1) + self.eps).view(B, C, 1)
        feat_mu = torch.mean(feat, dim=-1).view(B, C, 1)

        result = ((feat-feat_mu)/feat_std) * style_std + style_mu
        result = result.view(B, C, H, W)
        return result

class Padding(nn.Module):
    def __init__(self, curr_size=224, ideal=256):
        super(Padding, self).__init__()
        self.curr_size = curr_size
        self.ideal = ideal
        pad_size = self.calculate_size()
        self.padding = nn.ReflectionPad2d(pad_size)
    def calculate_size(self):
        if self.curr_size < self.ideal:
            pad_size= (self.ideal - self.curr_size) // 2
        else:
            pad_size = 0
            
        return pad_size

    def forward(self, input):
        return self.padding(input)
    

class Self_Attention(nn.Module):
    def __init__(self, in_channel):
        super(Self_Attention, self).__init__()

        self.query_conv = nn.Conv2d(in_channels= in_channel, out_channels= in_channel//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels= in_channel, out_channels= in_channel//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels= in_channel, out_channels= in_channel, kernel_size=1)
        self.gamma = nn.Parameter(torch.rand(1).normal_(0.0, 0.02))
        self.softmax = nn.Softmax(dim = -1)
        
    def forward(self, input):
        
        B, C, H, W = input.size()
        query = self.query_conv(input).view(B, -1, H*W)   #B, C//8, H*W
        key = self.key_conv(input).view(B,-1, H*W)        #B, C//8, H*W
        value = self.value_conv(input).view(B, -1, H*W)   #B, C, H*W

        energy = torch.bmm(query.permute(0, 2, 1), key)   #B, H*W, H*W
        attention = self.softmax(energy) #B, H*W, softmax(H*W)

        out = torch.bmm(value, attention.permute(0,2,1)) #B, C, H*W
        out = out.view(B, C, H, W)

        out = self.gamma * out + input
        return out

class Conv_Layer(nn.Module):
# convolution 2d layer with spectral noramlization
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding= 0):
        super(Conv_Layer, self).__init__()
        # self.conv = nn.Sequential(
        #     nn.ReflectionPad2d(padding),
        #     nn.utils.parametrizations.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride))
        # )
        self.conv = nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding = padding))
        
    def forward(self, input):
        # print(input.size())
        out = self.conv(input)
        # print(out.size())
        return out

# Large Scale GAN Training for High Fidelity Natural Image Synthesis
class Residual_Downsample(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size =3, stride=1, padding=1):
        super(Residual_Downsample, self).__init__()

        #right
        self.right = nn.Sequential(
            nn.ReLU(),
            Conv_Layer(in_channels= in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding= padding),
            nn.ReLU(),
            Conv_Layer(in_channels= out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding= padding),
            nn.AvgPool2d(2)
        )

        #left
        if out_channels != in_channels:
            self.left = nn.Sequential(
                Conv_Layer(in_channels=in_channels, out_channels=out_channels, kernel_size = 1, stride = 1, padding = 0),
                nn.AvgPool2d(2)
            )
        else:
            self.left = nn.Sequential(
                nn.Identity(),
                nn.AvgPool2d(2)
            )
            
    def forward(self, input):
        residual = input
        out = self.right(input)
        residual = self.left(residual)
        out = out + residual
        return out


class Adaptive_Residual_Bottleneck(nn.Module):
    
    def __init__(self, in_channels):
        super(Adaptive_Residual_Bottleneck, self).__init__()
        
        #right
        self.adain0 = AdaIN()
        self.right0 = nn.Sequential(
            nn.ReLU(),
            Conv_Layer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride = 1, padding = 1),
        )
        self.adain1 = AdaIN()
        self.right1 = nn.Sequential(
            nn.ReLU(),
            Conv_Layer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride = 1, padding = 1),

        )
        #left
        self.left = nn.Sequential(
            nn.Identity()
        )
    def forward(self, input, psi):
        C = psi.size(1)
        mu0 = psi[:, 0:C//4, :]
        std0 = psi[:, C//4:C//2, :]
        mu1 = psi[:, C//2:3*C//4, :]
        std1 = psi[:, 3*C//4:C, :]
        residual = input

        #right
        out = self.adain0(input, style_mu = mu0, style_std = std0)
        out = self.right0(out)
        out = self.adain1(input, style_mu= mu1, style_std =std1)
        out = self.right1(out)

        #left
        residual = self.left(residual)

        out = out + residual
        return out

class Adaptive_Residual_Upsample(nn.Module):

    def __init__(self, in_channels, out_channels, out_size = None, kernel_size=3, stride=1, padding=1, mode="bilinear"):
        super(Adaptive_Residual_Upsample, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        #right
        self.adain0 = AdaIN()
        if out_size == None:
            self.right0 = nn.Sequential(
                nn.ReLU(),
                nn.Upsample(scale_factor=2, mode = mode, align_corners=False),
                Conv_Layer(in_channels = in_channels, out_channels=out_channels, kernel_size= kernel_size, stride=stride, padding= padding)
            )
        else:
            self.right0 = nn.Sequential(
                nn.ReLU(),
                nn.Upsample(size = out_size, mode = mode, align_corners=False),
                Conv_Layer(in_channels = in_channels, out_channels=out_channels, kernel_size= kernel_size, stride=stride, padding= padding)
            )

        self.adain1 = AdaIN()
        self.right1 = nn.Sequential(
            nn.ReLU(),
            Conv_Layer(in_channels= out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding = padding)
        )

        #left
        if out_size == None:
            self.left = nn.Sequential(
                nn.Upsample(scale_factor= 2, mode = mode, align_corners=False),
                Conv_Layer(in_channels= in_channels, out_channels= out_channels, kernel_size= 1, stride=1, padding = 0)
            )
        else:
            self.left = nn.Sequential(
                nn.Upsample(size = out_size, mode = mode, align_corners=False),
                Conv_Layer(in_channels= in_channels, out_channels= out_channels, kernel_size= 1, stride=1, padding = 0)
            )


    def forward(self, input, psi):
        C = psi.size(1)
        ind0 = self.in_channels
        ind1 = ind0 + self.in_channels
        ind2 = ind1 + self.out_channels
        ind3 = ind2 + self.out_channels
        style_mu0 = psi[:, 0:ind0, :]
        style_std0 = psi[:, ind0:ind1, :]
        style_mu1 = psi[:, ind1:ind2, :]
        style_std1 = psi[:, ind2:ind3, :]

        residual = input

        #right
        out = self.adain0(input, style_mu = style_mu0, style_std = style_std0)
        out = self.right0(out)
        out = self.adain1(out, style_mu= style_mu1, style_std = style_std1)
        out = self.right1(out)

        #left
        residual = self.left(residual)

        out = out + residual
        return out


class Residual(nn.Module):

    def __init__(self, in_channels):
        super(Residual, self).__init__()
        
        
        self.right0 = Conv_Layer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        self.right1 = nn.ReLU(inplace = False)
        self.right2 = Conv_Layer(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1)
        

        self.left = nn.Sequential(
            nn.Identity()
        )


    def forward(self, input):
        residual = input
        out0 = self.right0(input)
        out1 = self.right1(out0)
        # print(out1.size())
        out2 = self.right2(out1)
        # out = self.right(input)
        # residual = self.left(residual)
        out = out2 + residual
        return out

        
class Bottleneck4(nn.Module):

    def __init__(self, in_channels=512):
        super(Bottleneck4, self).__init__()
        self.in_channels = in_channels
        self.b0 = Adaptive_Residual_Bottleneck(512)
        self.b1 = Adaptive_Residual_Bottleneck(512)
        self.b2 = Adaptive_Residual_Bottleneck(512)
        self.b3 = Adaptive_Residual_Bottleneck(512)

    def forward(self, input, psi):
        C = psi.size(1)
        psi0 = psi[:, 0:C//4, :]
        psi1 = psi[:, C//4:C//2, :]
        psi2 = psi[:, C//2:3*C//4, :]
        psi3 = psi[:, 3*C//4:C, :]

        out = self.b0(input, psi0)
        out = self.b1(out, psi1)
        out = self.b2(out, psi2)
        out = self.b3(out, psi3)

        return out

class Bottleneck5(nn.Module):

    def __init__(self, in_channels=512):
        super(Bottleneck5, self).__init__()
        self.in_channels = in_channels
        self.b0 = Adaptive_Residual_Bottleneck(512)
        self.b1 = Adaptive_Residual_Bottleneck(512)
        self.b2 = Adaptive_Residual_Bottleneck(512)
        self.b3 = Adaptive_Residual_Bottleneck(512)
        self.b4 = Adaptive_Residual_Bottleneck(512)

    def forward(self,input, psi):
        C = psi.size(1)
        psi0 = psi[:, 0:C//5, :]
        psi1 = psi[:, C//5:2*C//5, :]
        psi2 = psi[:, 2*C//5:3*C//5, :]
        psi3 = psi[:, 3*C//5:4*C//5, :]
        psi4 = psi[:, 4*C//5:C, :]
        out = self.b0(input, psi0)
        out = self.b1(out, psi1)
        out = self.b2(out, psi2)
        out = self.b3(out, psi3)
        out = self.b4(out, psi4)
    
        return out

    



 
