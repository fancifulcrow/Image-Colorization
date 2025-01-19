import torch
import kornia.color as K
from typing import Tuple, Dict
import yaml


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def rgb_to_lab(rgb_img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:    
    # Convert RGB to L*a*b
    lab_imgs = K.rgb_to_lab(rgb_img)
    
    # Split channels and normalize
    l_channel = lab_imgs[:, 0:1, :, :] / 100.0  # L channel: [0, 100] -> [0, 1]
    ab_channels = lab_imgs[:, 1:, :, :] / 128.0 # [-128, 127] to [-1, 1]
    
    return l_channel, ab_channels


def lab_to_rgb(l_channel: torch.Tensor, ab_channels: torch.Tensor) -> torch.Tensor:
    # Denormalize L channel from [0, 1] to [0, 100]
    l_channel = l_channel * 100.0
    # Denormalize a*b channels from [-1, 1] to [-128, 127]
    ab_channels = ab_channels * 128.0
    
    # Combine channels
    lab_imgs = torch.cat([l_channel, ab_channels], dim=1)
    
    # Convert back to RGB
    rgb_img = K.lab_to_rgb(lab_imgs)

    return rgb_img


def load_config(config_path: str) -> Dict:
    with open(config_path, mode="r") as f:
        config = yaml.safe_load(f)

    return config
