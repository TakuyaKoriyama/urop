from cv2_helper_function import read_img, vis_img, transform, recognize, cv2pil
import torch
import argparse
import pickle
import cv2
from finetune_helper_function import load_model

parser = argparse.ArgumentParser(description="recognizer")
parser.add_argument("data_name", type=str, help="example: 'ants_bees'")
args = parser.parse_args()

data_name = args.data_name
data_path = "./data/" + data_name  
label_to_name_path = data_path + "/label_to_name"

f = open(label_to_name_path, 'rb')
label_to_name = pickle.load(f)
model = load_model(root=data_path, num_classes=len(label_to_name))

cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()

    img_pil = cv2pil(frame)
    img = transform(img_pil)
    outputs = recognize(model=model, img=img)
    _, pred = torch.max(outputs, 1)

    
    
    edframe = frame

    cv2.putText(edframe, label_to_name[pred], (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)


    cv2.imshow('Edited Frame', edframe)

    k = cv2.waitKey(1)
    if k == 27:
        break


cap.release()
cv2.destroyAllWindows()