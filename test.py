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
    model.train()
    discriminator.train()

    # Use BCEWithLogitsLoss for adversarial loss, L1Loss for colorization
    criterion_d = nn.BCEWithLogitsLoss()
    criterion_g = nn.L1Loss()

    for epoch in range(num_epochs):
        running_loss_g = 0.0
        running_loss_d = 0.0

        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')

        for batch_idx, (data, _) in enumerate(pbar):
            # Convert RGB data to Lab format
            l_channel, ab_channels = rgb_to_lab(data)
            l_channel, ab_channels = l_channel.to(device), ab_channels.to(device)

            # Combine L and ab channels for the discriminator input
            real_lab = torch.cat((l_channel, ab_channels), dim=1)

            ## Discriminator training
            optimizer_d.zero_grad()

            # Real images (ground truth Lab)
            real_output = discriminator(real_lab)
            real_labels = torch.ones(real_output.shape).to(device)

            # Generate fake ab channels using the generator
            with torch.no_grad():
                fake_ab = model(l_channel)
            
            # Combine L channel with generated ab for discriminator
            fake_lab = torch.cat((l_channel, fake_ab), dim=1)
            fake_output = discriminator(fake_lab)
            fake_labels = torch.zeros(fake_output.shape).to(device)

            # Discriminator loss
            loss_d_real = criterion_d(real_output, real_labels)
            loss_d_fake = criterion_d(fake_output, fake_labels)
            loss_d = (loss_d_real + loss_d_fake) / 2

            loss_d.backward()
            optimizer_d.step()

            ## Generator training
            optimizer_g.zero_grad()

            # Generate fake ab channels again for the generator loss
            fake_ab = model(l_channel)
            fake_lab = torch.cat((l_channel, fake_ab), dim=1)
            fake_output = discriminator(fake_lab)

            # Generator loss
            loss_colorization = criterion_g(fake_ab, ab_channels)
            loss_adversarial = criterion_d(fake_output, real_labels)
            loss_g = loss_adversarial + 10 * loss_colorization

            loss_g.backward()
            optimizer_g.step()

            # Track losses
            running_loss_g += loss_g.item()
            running_loss_d += loss_d.item()

            # Update progress bar
            pbar.set_postfix({
                'loss_g': f'{(running_loss_g / (batch_idx + 1)):.4f}',
                'loss_d': f'{(running_loss_d / (batch_idx + 1)):.4f}'
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
optimizer = optim.Adam(model.parameters(), lr=1e-3)
d_optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 10

train(model, discriminator, train_loader, optimizer, d_optimizer, device, num_epochs)

print("\nGenerating visualizations...")
visualize_results(model, test_loader, device)