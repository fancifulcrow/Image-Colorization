import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.model import UNET, PatchDiscriminator
from src.utils import count_parameters, rgb_to_lab, lab_to_rgb


def train(model, discriminator, train_loader, optimizer_g, optimizer_d, device, num_epochs: int) -> None:
    # Loss Functions
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    # Constants for loss weights
    lambda_l1 = 100.0  # Weight for L1 loss
    lambda_fm = 10.0   # Weight for feature matching loss
    
    
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        running_d_loss = 0.0
        running_g_loss = 0.0
        
        for batch_idx, (images, _) in enumerate(pbar):
            images = images.to(device)

            l_channel, real_ab = rgb_to_lab(images)
            
            # Train Discriminator
            with torch.no_grad():
                test_output = discriminator(images)

            real_labels = torch.ones(test_output.shape, device=device)
            fake_labels = torch.zeros(test_output.shape, device=device)
            
            optimizer_d.zero_grad()
            
            real_rgb = images
            real_disc_output = discriminator(real_rgb)
            d_real_loss = bce_loss(real_disc_output, real_labels)
            
            fake_ab = model(l_channel)
            fake_rgb = lab_to_rgb(l_channel, fake_ab.tanh())
            fake_disc_output = discriminator(fake_rgb.detach())
            d_fake_loss = bce_loss(fake_disc_output, fake_labels)
            
            d_loss = (d_real_loss + d_fake_loss) * 0.5
            d_loss.backward()
            optimizer_d.step()
            
            # Train Generator
            optimizer_g.zero_grad()
            
            fake_disc_output = discriminator(fake_rgb)
            g_gan_loss = bce_loss(fake_disc_output, real_labels)
            
            g_l1_loss = l1_loss(fake_ab, real_ab) * lambda_l1
            
            g_loss = g_gan_loss + g_l1_loss
            g_loss.backward()
            optimizer_g.step()
            
            # Progress bar
            running_d_loss += d_loss.item()
            running_g_loss += g_loss.item()
            
            pbar.set_postfix({
                'D_loss': f'{running_d_loss / (batch_idx + 1):.4f}',
                'G_loss': f'{running_g_loss / (batch_idx + 1):.4f}'
            })
     
 
def visualize_results(model, data_loader, device, num_images: int = 5) -> None:
    model.eval()
    plt.figure(figsize=(9, 7.5))
    
    row_titles = ['B & W', 'Original', 'Colorized', 'Predicted *a', 'Predicted *b']
    rows = len(row_titles)
    
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
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
            
            plt.subplot(rows, num_images + 1, i + 2 + 3 * (num_images + 1))
            plt.imshow(ab_pred[0, 0].cpu().numpy(), cmap='RdYlGn_r')
            plt.axis('off')
            
            plt.subplot(rows, num_images + 1, i + 2 + 4 * (num_images + 1))
            plt.imshow(ab_pred[0, 1].cpu().numpy(), cmap='YlGnBu_r')
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((64, 64)),
])

train_dataset = datasets.Flowers102(root='./data', split="train", download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = datasets.Flowers102(root='./data', split="test", download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNET(1, 2).to(device)
discriminator = PatchDiscriminator(3).to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
d_optimizer = optim.Adam(discriminator.parameters(), lr=1e-3, betas=(0.5, 0.999))

torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

num_epochs = 10

train(model, discriminator, train_loader, optimizer, d_optimizer, device, num_epochs)

print("\nGenerating visualizations...")
visualize_results(model, test_loader, device)