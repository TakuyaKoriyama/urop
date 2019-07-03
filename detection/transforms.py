import random
import torch
from PIL import Image
from torchvision.transforms import functional as F
import cv2
import numpy as np 
def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def cv2pil(image_cv):
    image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    image_pil = Image.fromarray(image_cv)
    image_pil = image_pil.convert('RGB')
    return image_pil

def mask_transform(mask):
    mask = mask.detach().numpy()
    mask = mask.astype(np.uint8)
    mask_ed = mask.transpose(1, 2, 0)
    return mask_ed

def read_img(img_path):
    img_bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    return img_bgr

def IOU_cheack(p, q , threshold=0.5):
    px_0 = p[0]
    py_0 = p[1]
    px_1 = p[2]
    py_1 = p[3]

    qx_0 = q[0]
    qy_0 = q[1]
    qx_1 = q[2]
    qy_1 = q[3]

    s_p = (px_0 - px_1) * (py_0 - py_1)
    s_q = (qx_0 - qx_1) * (qy_0 - qy_1)

    if px_1 < qx_0 and px_0 < qx_1 and py_1 < qy_0 and py_0 < qy_1:
        x_list = [px_0, px_1, qx_1, qx_1]
        y_list = [py_0, py_1, qy_1, qy_1]
        x_max = max(x_list)
        x_min = min(x_list)
        y_max = max(y_list)
        y_min = min(y_list)
        x_list.remove(x_max)
        x_list.remove(x_min)
        y_list.remove(y_max)
        y_list.remove(y_min)
        x = x_list[0] - x_list[1]
        y = y_list[0] - y_list[1]
        iou = abs(x*y) /s_p + s_q - abs(x*y)
    else:
        iou = 0
    
    if iou > threshold:
        return False
    else:
        return True