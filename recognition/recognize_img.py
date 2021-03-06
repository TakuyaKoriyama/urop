from cv2_helper_function import read_img, vis_img, transform, recognize, cv2pil
import torch
import argparse
import pickle
from finetune_helper_function import load_model
parser = argparse.ArgumentParser(description="recognizer")

parser.add_argument("data_name", type=str, help="example: 'ants_bees'")
parser.add_argument('img_name', type=str, help="example: 'ants.jpg'")
args = parser.parse_args()

data_name = args.data_name ##'ants_bees'
data_path = "./data/" + data_name  
label_to_name_path = data_path + "/label_to_name"

img_name = args.img_name 
test_img_path = data_path + "/test_img/" + img_name


f = open(label_to_name_path, 'rb')
label_to_name = pickle.load(f)
model = load_model(root = data_path, num_classes=len(label_to_name))

img_cv = read_img(img_path=test_img_path)
vis_img(img=img_cv)
img_pil = cv2pil(img_cv)
img = transform(img_pil)
outputs = recognize(model=model, img=img)
_, pred = torch.max(outputs, 1)

print(label_to_name[pred])


