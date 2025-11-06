import time
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # (64,1,28,28)
        self.net = nn.Sequential(
            # (64, 32, 28, 28)
            nn.Conv2d(1, 32, 3),
            # (64, 64, 28, 28)
            nn.Conv2d(32, 64, 3),
            # (64, 128, 28, 28)
            nn.Conv2d(64, 128, 3),
            # (64, 256, 28, 28)
            nn.Conv2d(128, 256, 3),
            # (64, 512, 28, 28)
            nn.Conv2d(256, 512, 3),
            # ()
            nn.Linear(512 * 7 * 7, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = MyNet()

    train_dataset = torchvision.datasets.MNIST(
        "../data/train/", train=True, download=False,
        transform=torchvision.transforms.ToTensor())
    test_dataset = torchvision.datasets.MNIST(
        '../data/test/', train=False, download=False,
        transform=torchvision.transforms.ToTensor())
    print(train_dataset)
    print(test_dataset)

    # (img[batch_size, 1, 28, 28], label[batch_size, 1])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # 测试gpu能否使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    print(model)
