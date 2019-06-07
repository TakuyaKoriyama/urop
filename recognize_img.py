from finetune_helper_function import train_model, set_parameter_requires_grad, initialize_model
from cv2_helper_function import read_img, vis_img, transform, recognize, cv2pil
import torch


model = torch.load("./model/ant_bee_classifier.pkl")

label_to_name = ["ants", "bees"]

img_path = "./img/test_img/ants.jpg"


img_cv = read_img(img_path=img_path)
vis_img(img=img_cv)

img_pil = cv2pil(img_cv)
img = transform(img_pil)
outputs = recognize(model=model, img=img)

_, pred = torch.max(outputs, 1)


print(label_to_name[pred])



