import time
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader


class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        # (64,1,28,28)
        # conv （输入尺寸-卷积核尺寸+2*填充尺寸）/步长 +1
        self.net = nn.Sequential(
            # (64, 32, 28, 28)
            nn.Conv2d(1, 32, 3, 1, 1),
            # (64, 32, 14, 14)
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            # (64, 64, 12, 12)
            nn.Conv2d(32, 64, 3),
            # (64, 64, 6, 6)
            nn.MaxPool2d(2, 2),
            nn.ReLU(),
            # (64, 64*6*6)
            nn.Flatten(),

            nn.Linear(64 * 6 * 6, 512),
            nn.Linear(512, 10),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        return self.net(x)


def load_dataset(root_path, is_train=True):
    _dataset = torchvision.datasets.MNIST(
        root=root_path, train=is_train, download=False,
        transform=torchvision.transforms.ToTensor()
    )
    return _dataset


def test(model, loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for img, label in loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            test_loss += loss_fn(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()
    test_loss /= len(loader.dataset)
    acc = correct / len(loader.dataset)
    print(f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({acc:.2f}%)\n")
    return acc


def train(model, loader, optimizer, _epochs):
    model.train()

    for epoch in range(_epochs):

        total_loss = 0
        for batch_idx, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_idx % 5000 == 0:
                print('Train Epoch: {}, Loss: {:.6f}'.format(_epochs, total_loss / len(loader)))
        avg_loss = total_loss / len(loader)
        print('Train Epoch: {}, Avg loss: {:.6f}'.format(_epochs, avg_loss))
        
    return model


if __name__ == '__main__':
    batch_size = 64
    learning_rate = 0.001
    epochs = 5

    train_dataset = load_dataset("../data/train", True)
    test_dataset = load_dataset('../data/test/', False)
    print(train_dataset)
    print(test_dataset)

    # (img[batch_size, 1, 28, 28], label[batch_size, 1])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 测试gpu能否使用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MyNet()
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model = train(model, train_loader, optimizer, epochs)
    acc = test(model, test_loader)
    print(f"test data predict acc :{acc}")
    torch.save(model.state_dict(), 'model.pth')
