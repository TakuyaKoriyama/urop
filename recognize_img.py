from finetune_helper_function import train_model, set_parameter_requires_grad, initialize_model
from cv2_helper_function import read_img, vis_img, transform, recognize, cv2pil
import torch
import argparse
import pickle

parser = argparse.ArgumentParser(description="recognizer")

parser.add_argument("--model_path", required=True)
parser.add_argument("--img_path", required=True)
parser.add_argument("--label_to_name_path", required=True)
args = parser.parse_args()


model_path = args.model_path
label_to_name_path = args.label_to_name_path
img_path = args.img_path



model = torch.load(model_path)
f = open(label_to_name_path, 'rb')
label_to_name = pickle.load(f)
img_cv = read_img(img_path=img_path)
vis_img(img=img_cv)
img_pil = cv2pil(img_cv)
img = transform(img_pil)
outputs = recognize(model=model, img=img)

_, pred = torch.max(outputs, 1)

print(label_to_name[pred])



