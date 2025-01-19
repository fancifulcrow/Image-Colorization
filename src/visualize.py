import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

from .utils import rgb_to_lab, lab_to_rgb


def visualize(model, data_loader, device, num_images: int = 5) -> None:
    model.eval()
    plt.figure(figsize=((1 + num_images) * 1.5, 4.5))
    
    row_titles = ['B & W', 'Original', 'Colorized']
    rows = len(row_titles)
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            if i >= num_images:
                break
            
            data = data.to(device)
            
            l, _ = rgb_to_lab(data)
            l = l.to(device)
            
            ab_pred = model(l)
            
            rgb_pred = lab_to_rgb(l.cpu(), ab_pred.cpu())
            
            lightness = l[0].cpu().squeeze().numpy()
            original = data[0].cpu().permute(1, 2, 0).numpy()
            predicted = rgb_pred[0].permute(1, 2, 0).numpy()
            
            predicted = np.clip(predicted, 0, 1)
            
            # Add row titles
            if i == 0:  # Only add row titles once for the first column
                for row_idx, title in enumerate(row_titles):
                    plt.subplot(rows, num_images + 1, (row_idx + 1) * (num_images + 1) - num_images)
                    plt.text(0.5, 0.5, title, fontsize=12, ha='center', va='center', rotation=0)
                    plt.axis('off')
            
            # Add images
            plt.subplot(rows, num_images + 1, i + 2)
            plt.imshow(lightness, cmap='gray')
            plt.axis('off')
            
            plt.subplot(rows, num_images + 1, i + 2 + num_images + 1)
            plt.imshow(original)
            plt.axis('off')
            
            plt.subplot(rows, num_images + 1, i + 2 + 2 * (num_images + 1))
            plt.imshow(predicted)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()
