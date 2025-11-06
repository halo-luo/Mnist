import torch
import torchvision

# 下载mnist数据集

mnist_train = torchvision.datasets.MNIST("../data/train", train=True, download=True)
print(mnist_train)

mnist_test = torchvision.datasets.MNIST("../data/test", train=False, download=True)
print(mnist_test)
