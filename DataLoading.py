import torchvision
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def load_data(batch_size = 32, download=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=0, std=1)
    ])
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, transform=transform, download=download)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, transform=transform, download=download)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=2, persistent_workers=True)

    return train_loader,test_loader

if __name__ == "__main__":
    pass