import torch
import numpy as np
from torch import nn

## architecture
class UNet(torch.nn.Module):
    def __init__(self, ch_in=1, ch_out=1, init_n_feat=32):
        super(UNet, self).__init__()

        n_feat = init_n_feat
        self.encoder1 = UNet._block(ch_in, n_feat, depth=1)
        self.en_res1 = UNet._block(n_feat, n_feat, depth=2)
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder2 = UNet._block(n_feat, n_feat*2, depth=1)
        self.en_res2 = UNet._block(n_feat*2, n_feat*2, depth=2)
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.encoder3 = UNet._block(n_feat*2, n_feat*4, depth=1)
        self.en_res3 = UNet._block(n_feat*4, n_feat*4, depth=2)
        self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)


        self.bottleneck = UNet._block(n_feat*4, n_feat*8, depth=3)


        self.upconv3 = torch.nn.ConvTranspose2d(n_feat*8, n_feat*4, kernel_size=2, stride=2)
        self.decoder3 = UNet._block((n_feat*4)*2, n_feat*4, depth=3)

        self.upconv2 = torch.nn.ConvTranspose2d(n_feat*4, n_feat*2, kernel_size=2, stride=2)
        self.decoder2 = UNet._block((n_feat*2)*2, n_feat*2, depth=3)

        self.upconv1 = torch.nn.ConvTranspose2d(n_feat*2, n_feat, kernel_size=2, stride=2)
        self.decoder1 = UNet._block(n_feat*2, n_feat, depth=3)

        self.conv = torch.nn.Conv2d(in_channels=n_feat, out_channels=ch_out, kernel_size=1)


    def forward(self, x):
        enc1 = self.encoder1(x) 
        enc1_ = self.en_res1(enc1)
        enc1__ = self.pool1(enc1_+enc1) ## add residual block

        enc2 = self.encoder2(enc1__) 
        enc2_ = self.en_res2(enc2)
        enc2__ = self.pool2(enc2_+enc2) ## add residual block

        enc3 = self.encoder3(enc2__) 
        enc3_ = self.en_res3(enc3)
        enc3__ = self.pool3(enc3_+enc3) ## add residual block


        bottleneck = self.bottleneck(enc3__) 


        dec3 = self.upconv3(bottleneck)
        dec3 = torch.cat((dec3, enc3), dim=1) ## skip layers -> concatenation
        dec3 = self.decoder3(dec3) 

        dec2 = self.upconv2(dec3) 
        dec2 = torch.cat((dec2, enc2), dim=1) ## skip layers -> concatenation
        dec2 = self.decoder2(dec2) 

        dec1 = self.upconv1(dec2) 
        dec1 = torch.cat((dec1, enc1), dim=1) ## skip layers -> concatenation
        dec1 = self.decoder1(dec1)

        dec0 = self.conv(dec1)
        return torch.sigmoid(dec0) 

    @staticmethod
    def _block(ch_in, n_feat, depth):
        '''
        ch_in: input channel
        n_feat: channel number (except the first input channel)
        depth: number of add_block
        '''
        def add_block(input_channels, channels):
            block = [
                torch.nn.Conv2d(input_channels, channels, kernel_size=3, padding=1, bias=False), 
                torch.nn.BatchNorm2d(num_features=n_feat),
                torch.nn.ReLU(inplace=True)
            ]
            return block

        layers = []
        for _ in range(depth):
            layers.extend(add_block(ch_in, n_feat))
            ch_in = n_feat
        return torch.nn.Sequential(*layers)

print(UNet())





 


