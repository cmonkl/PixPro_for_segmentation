import torchvision.transforms as transforms
import torch.nn as nn
from resnet import MomentumBatchNorm2d
import torch

# double conv block
def double_conv_o(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class UNet(nn.Module):
    """
    UNet with padding architecture

    Parameters:
        bn_type: Momentum Batch Norm Vanilla Batch Norm
        channels: number of channels for each layer
        num_feat: num of output upsampled features
    """

    def __init__(
        self, bn_type="vanilla", channels=[64, 128, 256, 512, 1024], num_feat=128
    ):
        super().__init__()
        self.norm_layer = (
            nn.BatchNorm2d if bn_type == "vanilla" else MomentumBatchNorm2d
        )

        self.dconv_down1 = double_conv_o(3, channels[0])
        self.dconv_down2 = double_conv_o(channels[0], channels[1])
        self.dconv_down3 = double_conv_o(channels[1], channels[2])
        self.dconv_down4 = double_conv_o(channels[2], channels[3])
        self.dconv_down5 = double_conv_o(channels[3], channels[4])

        self.maxpool = nn.MaxPool2d(2)
        self.upsample4 = nn.ConvTranspose2d(
            channels[4], channels[3], kernel_size=2, stride=2
        )
        self.upsample3 = nn.ConvTranspose2d(
            channels[3], channels[2], kernel_size=2, stride=2
        )
        self.upsample2 = nn.ConvTranspose2d(
            channels[2], channels[1], kernel_size=2, stride=2
        )
        self.upsample1 = nn.ConvTranspose2d(
            channels[1], channels[0], kernel_size=2, stride=2
        )

        self.dconv_up4 = double_conv_o(channels[3] + channels[3], channels[3])
        self.dconv_up3 = double_conv_o(channels[2] + channels[2], channels[2])
        self.dconv_up2 = double_conv_o(channels[1] + channels[1], channels[1])
        self.dconv_up1 = double_conv_o(channels[0] + channels[0], channels[0])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, use_down_feat=False):
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)

        to_proj = x

        x = self.upsample4(x)
        x = torch.cat([x, conv4], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample3(x)
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample2(x)
        x = torch.cat([x, conv2], dim=1)

        x = self.dconv_up2(x)
        x = self.upsample1(x)
        x = torch.cat([x, conv1], dim=1)

        x = self.dconv_up1(x)

        if use_down_feat:
            return x, to_proj
        else:
            return x

