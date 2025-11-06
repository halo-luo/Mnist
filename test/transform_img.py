import numpy as np
import torch
import torchvision
from PIL import Image
from torch.utils.data import DataLoader
import os

dataset = torchvision.datasets.MNIST("../data/train", train=True,
                                     transform=torchvision.transforms.ToTensor(),
                                     download=False)
loader = DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

piltrans = torchvision.transforms.ToPILImage()


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


# 建立文件夹
def mkdir(dir_path='./'):
    num = 10
    for i in range(num):
        dir = f"{dir_path}/MNIST/num_{i}"
        if not os.path.exists(dir):
            os.makedirs(dir)


mkdir()

count_list = [1 for i in range(10)]

for img, label in loader:
    # print(img)
    num = int(label[0])
    print(num)
    print(type(img), img.shape)
    print(type(label), label.shape)
    img = torch.reshape(img, (1, 28, 28))
    image = piltrans(img)
    # image.show()

    dir = "./MNIST/num_{}".format(num)
    if not os.path.exists(dir):
        os.mkdir(dir)

    Image.Image.save(image, dir + r"/" + str(count_list[num]) + r".jpg")
    count_list[num] += 1
