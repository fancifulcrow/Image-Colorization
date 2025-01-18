import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
    

class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.double_conv = DoubleConv(in_channels, out_channels)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip = self.double_conv(x)
        x = self.max_pool(skip)

        return x, skip
    

class UpBlock(nn.Module):
    def __init__(self, in_channels:int, out_channels: int) -> None:
        super().__init__()

        self.up_conv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels, out_channels)


    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up_conv(x)

        # Concatenate skip connection with upsampled features
        x = torch.cat([skip, x], dim=1)

        x = self.double_conv(x)

        return x
    

class UNET(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        
        self.down_block1 = DownBlock(in_channels, 64)
        self.down_block2 = DownBlock(64, 128)
        self.down_block3 = DownBlock(128, 256)

        self.bottleneck = DoubleConv(256, 512)

        self.up_block3 = UpBlock(512, 256)
        self.up_block2 = UpBlock(256, 128)
        self.up_block1 = UpBlock(128, 64)

        self.conv_out = nn.Conv2d(64, out_channels, kernel_size=1, stride=1)

        self.apply(self.init_weights)

    def init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='leaky_relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GroupNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip1 = self.down_block1(x)
        x, skip2 = self.down_block2(x)
        x, skip3 = self.down_block3(x)

        x = self.bottleneck(x)

        x = self.up_block3(x, skip3)
        x = self.up_block2(x, skip2)
        x = self.up_block1(x, skip1)

        x = self.conv_out(x)
        x = F.tanh(x)
        
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
    
            DBlock(64, 128),
            DBlock(128, 256),

            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.sigmoid(self.network(x))
