import torch
import torchvision
from main import MyNet
from main import test

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

if __name__ == '__main__':
    print("predict mnist number")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device : {device}")

    model_path = "./model.pth"
    model = torch.load(model_path, device)
    # print(f"load model successfully {model}")

    test_dataset = torchvision.datasets.MNIST(
        root='../data/test/', train=False, download=False,
        transform=torchvision.transforms.ToTensor()
    )
    print(test_dataset)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    print(len(test_loader))
    data, target = next(iter(test_loader))
    print(data.shape)
    print(target.shape)

