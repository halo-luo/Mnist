import numpy
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import transforms
from torch.utils.data import DataLoader
import os


def read_img(img_path):
    img = Image.open(img_path)
    print(img.size)
    img = numpy.array(img)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    img = img.unsqueeze(0)
    return img


if __name__ == '__main__':
    dataset = torchvision.datasets.MNIST("../data/test", train=True,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=False)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    piltrans = torchvision.transforms.ToPILImage()

    img = read_img("MNIST/num_0/1.jpg")
    img1 = next(iter(loader))[0]
    print(img1.shape)
    print(img.shape)

    print(torch.equal(img, img1))

    for img, label in loader:
        if label == 0:
            img1 = img
            break

    print(torch.equal(img, img1))
