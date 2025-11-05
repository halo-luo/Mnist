import torch
import torchvision

# 下载mnist数据集

mnist = torchvision.datasets.MNIST("../data", download=True)
print(mnist)
