import time
import torch.nn as nn


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.Conv2d(32, 64, 3),
            nn.Conv2d(64, 128, 3),
            nn.Conv2d(128, 256, 3),
            nn.Conv2d(256, 512, 3),
            nn.Linear(512 * 7 * 7, 512),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.net(x)


if __name__ == '__main__':
    model = MyNet()
