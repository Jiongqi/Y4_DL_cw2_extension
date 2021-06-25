import torch
import numpy as np
from torch import nn

## architecture
class One_Block(nn.Module):
    def __init__(self, ch_in, ch_out, depth=3):
        '''
        ch_in: input channel number
        ch_out: output channel number
        depth: number of add_block
        '''
        super(One_Block, self).__init__()

        layers = []
        for _ in range(depth-1):
            layers.extend(self.sub_block(ch_out, ch_out))

        self.one_block_first = torch.nn.Sequential(*self.sub_block(ch_in, ch_out))
        self.one_block_rest = torch.nn.Sequential(*layers)

    def sub_block(self, ch_in, ch_out):
        sub_block_ = [
            torch.nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=False), 
            torch.nn.BatchNorm2d(num_features=ch_out),
            torch.nn.ReLU(inplace=True)
        ]
        return sub_block_

    def forward(self, x):
        x = self.one_block_first(x)
        out = x + self.one_block_rest(x)
        return out


class UNet(torch.nn.Module):
    def __init__(self, ch_in=1, ch_out=1, ch_nums=[32,64,128,256]):
        super(UNet, self).__init__()

        self.encoder1 = One_Block(ch_in, ch_nums[0], depth=3)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = One_Block(ch_nums[0], ch_nums[1], depth=3)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = One_Block(ch_nums[1], ch_nums[2], depth=3)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = One_Block(ch_nums[2], ch_nums[3], depth=3)


        self.upconv3 = torch.nn.ConvTranspose2d(ch_nums[3], ch_nums[2], kernel_size=2, stride=2)
        self.decoder3 = One_Block(ch_nums[2] *2, ch_nums[2], depth=3)

        self.upconv2 = torch.nn.ConvTranspose2d(ch_nums[2], ch_nums[1], kernel_size=2, stride=2)
        self.decoder2 = One_Block(ch_nums[1] *2, ch_nums[1], depth=3)

        self.upconv1 = torch.nn.ConvTranspose2d(ch_nums[1], ch_nums[0], kernel_size=2, stride=2)
        self.decoder1 = One_Block(ch_nums[0] *2, ch_nums[0], depth=3)

        self.conv = torch.nn.Conv2d(in_channels=ch_nums[0], out_channels=ch_out, kernel_size=1)


    def forward(self, x):
        enc1 = self.encoder1(x) 
        enc1_ = self.pool1(enc1)

        enc2 = self.encoder2(enc1_) 
        enc2_ = self.pool2(enc2)

        enc3 = self.encoder3(enc2_) 
        enc3_ = self.pool3(enc3)


        bottleneck = self.bottleneck(enc3_) 


        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((enc3, dec3), dim=1) ## skip layers -> concatenation
        dec3 = self.decoder3(dec3) 

        dec2 = self.upconv2(dec3) 
        dec2 = torch.cat((enc2, dec2), dim=1) ## skip layers -> concatenation
        dec2 = self.decoder2(dec2) 

        dec1 = self.upconv1(dec2) 
        dec1 = torch.cat((enc1, dec1), dim=1) ## skip layers -> concatenation
        dec1 = self.decoder1(dec1)

        dec0 = self.conv(dec1)
        return torch.sigmoid(dec0) 

# print(UNet())

######### Leave there because they can form the structure but are not able to take intermediate values for concatenation.
# class Encoder(nn.Module):
#     def __init__(self, ch_nums):
#         super(Encoder, self).__init__()

#         layers = []
#         max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
#         for i in len(ch_nums-1):
#             ch_in = ch_nums[i]
#             ch_out = ch_nums[i+1]
#             layers.extend(One_Block(ch_in, ch_out, depth=3))
#             if i is not len(ch_nums-1):
#                 layers.extend(max_pool)

#         self.encoder = torch.nn.Sequential(*layers)



# def __init__(self, ch_nums=[1,32,64,128,256]):
#         super(UNet, self).__init__()

#         ## encoder layers
#         encoder_layers = []
#         max_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
#         for i in len(ch_nums-1):
#             encoder_ch_in = ch_nums[i]
#             encoder_ch_out = ch_nums[i+1]
#             encoder_layers.extend(One_Block(encoder_ch_in, encoder_ch_out, depth=3))
#             if i is not len(ch_nums-1):
#                 encoder_layers.extend(max_pool)
        
#         self.encoder = torch.nn.Sequential(*encoder_layers)


#         ## decoder layers
#         decoder_layers = []
#         decoder_ch_nums = ch_nums[::-1]
#         for j in len(decoder_ch_nums):
#             decoder_ch_in = decoder_ch_nums[i]
#             decoder_ch_out = decoder_ch_nums[i+1]
#             decoder_layers.append(torch.nn.ConvTranspose2d(decoder_ch_in, decoder_ch_out, kernel_size=2, stride=2))
#             decoder_layers.append(One_Block(decoder_ch_in *2, decoder_ch_out, depth=3))


#         self.decoder = torch.nn.Sequential(*decoder_layers)





 


