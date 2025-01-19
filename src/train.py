import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import time

from .utils import rgb_to_lab, lab_to_rgb


def train(model, discriminator, train_loader, num_epochs: int, optimizer_g, optimizer_d, device) -> None:
    l1_loss = nn.L1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    lambda_l1 = 100.0  # Weight for L1 loss

    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_id = time.time()

    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        running_d_loss = 0.0
        running_g_loss = 0.0
        
        for batch_idx, images in enumerate(pbar):
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
            fake_rgb = lab_to_rgb(l_channel, fake_ab)
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
        
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"{models_dir}/{model_id}_{epoch + 1}_unet.pth")
            torch.save(discriminator.state_dict(), f"{models_dir}/{model_id}_{epoch + 1}_discriminator.pth")
            torch.save(optimizer_g.state_dict(), f"{models_dir}/{model_id}_{epoch + 1}_optimizer_g.pth")
            torch.save(optimizer_d.state_dict(), f"{models_dir}/{model_id}_{epoch + 1}_optimizer_d.pth")
