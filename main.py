import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

device = "cuda" if torch.cuda.is_available() else "cpu"


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.double_conv = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.double_conv(x)
        x = self.max_pool(skip)

        return x, skip


class UpBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels: int) -> None:
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(out_channels * 2, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)

        # Center Cropping
        diff_y = (skip.size()[2] - x.size()[2]) // 2
        diff_x = (skip.size()[3] - x.size()[3]) // 2
        
        skip = skip[:, :, 
                   diff_y:diff_y + x.size()[2],
                   diff_x:diff_x + x.size()[3]]

        # Concatenate skip connection with upsampled features
        x = torch.cat([skip, x], dim=1)

        x = self.double_conv(x)

        return x


class UNET(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        
        self.down_block1 = DownBlock(1, 64)
        self.down_block2 = DownBlock(64, 128)
        self.down_block3 = DownBlock(128, 256)
        self.down_block4 = DownBlock(256, 512)

        self.double_conv = DoubleConv(512, 1024)

        self.up_block4 = UpBlock(1024, 512)
        self.up_block3 = UpBlock(512, 256)
        self.up_block2 = UpBlock(256, 128)
        self.up_block1 = UpBlock(128, 64)

        self.conv_out = nn.Conv2d(64, 2, kernel_size=1)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, x1 = self.down_block1(x)
        x, x2 = self.down_block2(x)
        x, x3 = self.down_block3(x)
        x, x4 = self.down_block4(x)

        x = self.double_conv(x)

        x = self.up_block4(x, x4)
        x = self.up_block3(x, x3)
        x = self.up_block2(x, x2)
        x = self.up_block1(x, x1)

        x = self.conv_out(x)

        return x
    

unet = UNET().to(device)

summary(unet, input_size=(1, 572, 572))