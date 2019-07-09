import cv2
import numpy as np
import os

root = 'finetune_data'
imgs_list = list(sorted(os.listdir(os.path.join(root, "Images"))))
print(imgs_list)
box_label_path = os.path.join(root, 'box_label.txt')
file = open(box_label_path, 'a')
for i in range(len(imgs_list) - 1):
    img_path = os.path.join(root, "Images", imgs_list[i+1])
    window_name = 'Image:{}'.format(i)
    im = cv2.imread(img_path)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.imshow(window_name, im)
    box_id = 0
    while True:
        print(box_id, ': enter waitkey')
        k = cv2.waitKey(0)
        if k == 48:
            cv2.destroyAllWindows()
            break
        else:
            print('class label:{}'.format(k))
            r = cv2.selectROI(window_name, im)
            class_label = k
            x_0 = str(r[1])
            x_1 = str(r[1]+r[3])
            y_0 = str(r[0])
            y_1 = str(r[0]+r[2])
            file.write(str(i) + ',' + str(box_id) + ',' +  str(class_label) + ',' + x_0 + ',' + y_0 + ',' + x_1 + ',' + y_1 + '\n')
            box_id += 1

file.close()