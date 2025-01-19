import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.model import UNET, PatchDiscriminator
from src.utils import load_config
from src.train import train
from src.visualize import visualize


if __name__ == "__main__":
    config_path = "config/default.yaml"
    config = load_config(config_path=config_path)

    assert config["data"]["image_size"] % 8 == 0, "The image size must be divisible by 8"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    unet = UNET(1, 2).to(device)
    discriminator = PatchDiscriminator(3).to(device)

    optimizer_g = optim.Adam(unet.parameters(), lr=config["training"]["learning_rate"], betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=config["training"]["learning_rate"], betas=(0.5, 0.999))

    transform = transforms.Compose([
        transforms.Resize((config["data"]["image_size"], config["data"]["image_size"])),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.Flowers102(root='./data', split="train", download=True, transform=transform)
    test_dataset = datasets.Flowers102(root='./data', split="test", download=True, transform=transform)

    # Strip Labels
    train_dataset = [(img) for img, _ in train_dataset]
    test_dataset = [(img) for img, _ in test_dataset]

    train_loader = DataLoader(train_dataset, batch_size=config["data"]["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    num_epochs = config["training"]["num_epochs"]

    train(unet, discriminator, train_loader, num_epochs, optimizer_g, optimizer_d, device)

    print("\nGenerating visualizations...")
    visualize(unet, test_loader, device)
