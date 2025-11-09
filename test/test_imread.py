import cv2
import numpy
from matplotlib import pyplot as plt
import torch

if __name__ == '__main__':
    img_path = "./MNIST/num_0/1.jpg"
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = torch.from_numpy(img).float()
    print(img.shape)
    img = img.unsqueeze(0)
    print(img.shape)

    img = plt.imread(img_path)
    img = numpy.array(img)
    img = torch.from_numpy(img).float()
    print(img.shape)
    img = img.unsqueeze(0)
    print(img.shape)
