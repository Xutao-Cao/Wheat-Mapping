import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """Conv -> BN -> ReLU -> Conv -> BN -> ReLU"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)            
        )

    def forward(self, x):
        return self.double_conv(x)

# class ResDoubleConv(nn.Module):

#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         nn.

class DownSampleBlock(nn.Module):
    """Maxpool -> double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down_sample = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down_sample(x)

class UpSampleBlock(nn.Module):
    """Transposedconv -> concatenate -> double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, down_sample_x):
        x = self.up(x)
        x = torch.cat([down_sample_x, x], dim=1)
        x = self.double_conv(x)
        return x

class U_net(nn.Module):

    def __init__(self, n_channels, n_classes, n_feature_maps = 128):
        super().__init__()
        self.channels = n_channels
        self.classes = n_classes
        self.doubleconv_0 = DoubleConv(n_channels, n_feature_maps)
        self.down_0 = DownSampleBlock(n_feature_maps, 2 * n_feature_maps)
        self.down_1 = DownSampleBlock(2 * n_feature_maps, 4 * n_feature_maps)
        self.down_2 = DownSampleBlock(4 * n_feature_maps, 8 * n_feature_maps)
        self.down_3 = DownSampleBlock(8 * n_feature_maps, 16 * n_feature_maps)
        self.up_0 = UpSampleBlock(16 * n_feature_maps, 8 * n_feature_maps)
        self.up_1 = UpSampleBlock(8 * n_feature_maps, 4 * n_feature_maps)
        self.up_2 = UpSampleBlock(4 * n_feature_maps, 2 * n_feature_maps)
        self.up_3 = UpSampleBlock(2 * n_feature_maps, n_feature_maps)
        self.out = nn.Conv2d(n_feature_maps, n_classes, kernel_size=1)
        
    def forward(self, x):
        x0 = self.doubleconv_0(x)
        x1 = self.down_0(x0)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x = self.down_3(x3)
        x = self.up_0(x, x3)
        x = self.up_1(x, x2)
        x = self.up_2(x, x1)
        x = self.up_3(x, x0)
        x = self.out(x)
        return x.squeeze()
    
    def get_feature(self, x):
        x0 = self.doubleconv_0(x)
        x1 = self.down_0(x0)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x = self.down_3(x3)
        x = self.up_0(x, x3)
        x = self.up_1(x, x2)
        x = self.up_2(x, x1)
        x = self.up_3(x, x0)
        return x
