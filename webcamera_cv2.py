from cv2_helper_function import read_img, vis_img, transform, recognize, cv2pil
import torch
import argparse
import pickle
import cv2

parser = argparse.ArgumentParser(description="recognizer")
parser.add_argument("data_name", type=str, help="example: 'ants_bees'")
args = parser.parse_args()

data_name = args.data_name
data_path = "./data/" + data_name  
label_to_name_path = data_path + "/label_to_name"
model_path = data_path + "/model"

model = torch.load(model_path)
f = open(label_to_name_path, 'rb')
label_to_name = pickle.load(f)


cap = cv2.VideoCapture(0)

while True:
    # VideoCaptureから1フレーム読み込む
    ret, frame = cap.read()

    img_pil = cv2pil(frame)
    img = transform(img_pil)
    outputs = recognize(model=model, img=img)
    _, pred = torch.max(outputs, 1)

    
    
    edframe = frame

    cv2.putText(edframe, label_to_name[pred], (0,50), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255,0), 3, cv2.LINE_AA)

    # 加工済の画像を表示する
    cv2.imshow('Edited Frame', edframe)

    # キー入力を1ms待って、k が27（ESC）だったらBreakする
    k = cv2.waitKey(1)
    if k == 27:
        break

# キャプチャをリリースして、ウィンドウをすべて閉じる
cap.release()
cv2.destroyAllWindows()