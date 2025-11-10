import numpy
import torch
import torchvision
from torch.utils.data import DataLoader
from main import MyNet
from main import test
import cv2
from matplotlib import pyplot as plt


def predict_one(model, img):
    model.eval()
    with torch.no_grad():
        output = model(img)
        pred = output.argmax(dim=1, keepdim=True)

    return pred.item()


def predict_many(model, imgs):
    model.eval()

    with torch.no_grad():
        output = model(imgs)
        pred = output.argmax(dim=1, keepdim=True)
    return pred


# 使用opencv读取图片
def read_img(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    # 转为tensor,(28, 28)
    img = torch.from_numpy(img).float()
    # (1, 28, 28)
    img = img.unsqueeze(0)
    return img


# 使用plt读取图片
def read_img2(img_path):
    img = plt.imread(img_path)
    img = numpy.array(img)
    img = torch.from_numpy(img).float()
    img = img.unsqueeze(0)
    return img


if __name__ == '__main__':
    print("predict mnist number")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")

    model_path = "./model.pth"
    model = MyNet().to(device)
    model.load_state_dict(torch.load(model_path))
    # print(f"load model successfully {model}")

    test_dataset = torchvision.datasets.MNIST(
        root='../data/test/', train=False, download=False,
        transform=torchvision.transforms.ToTensor()
    )
    print(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    print(len(test_loader))

    data, target = next(iter(test_loader))
    data = data.to(device)
    target = target.to(device)
    print(data.shape)
    print(target.shape)

    predict_result = predict_many(model, data)
    # print(predict_result)
    acc_count = predict_result.eq(target.view_as(predict_result)).sum().item()
    print(acc_count)
