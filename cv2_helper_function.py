import torch
import matplotlib.pyplot as plt
import cv2
from torchvision import transforms
from PIL import Image

def read_img(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img_bgr

def vis_img(img, color_order="BGR"):
    assert color_order in ["RGB", "BGR"]
    
    if color_order == "RGB":
        plt.imshow(img), plt.axis("off")
        plt.show()
    
    elif color_order == "BGR":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img), plt.axis("off")
        plt.show()


def transform(img): 
    tsfn = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return tsfn(img).view(1, 3, 224, 224)


def recognize(model, img):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img.to(device)
    labels = model(img)
    return labels


def cv2pil(image_cv):
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert('RGB')
    return image_pil
