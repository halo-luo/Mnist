import torch
import torchvision
from torch.utils.data import DataLoader
import sys

sys.path.append("../src")
from main import MyNet

dataset = torchvision.datasets.MNIST(
    root='../data', train=True,
    transform=torchvision.transforms.ToTensor(), download=False)
print(len(dataset))

loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)
print(loader)
# for img, label in loader:
#     # batchsize, 1, 28, 28
#     print(img.shape)  # torch.Size([64, 1, 28, 28])
#     print(label)
#     print(label.shape)
#     break

first_batch_imgs, first_batch_targets = next(iter(loader))
print(first_batch_imgs.shape)
print(first_batch_targets.shape)
# 生成一个随机3*3kernel
kernel = torch.rand(3, 3)
print(kernel)
kernel = kernel.reshape((1, 1, 3, 3))
# 将图片和kernel进行卷积操作
img = first_batch_imgs[0]  # (1, 28, 28)
print(img.shape)
result = torch.conv2d(first_batch_imgs, kernel, None, 2, 0)
print(result.shape)
model = MyNet()
result = model.forward(first_batch_imgs)
print(result.shape)
print(result[0])

# model.summary()
