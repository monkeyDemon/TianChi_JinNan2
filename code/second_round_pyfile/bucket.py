import cv2
import json
import os
import numpy as np

json_path = "./jinnan2_round2_train_20190401/train_restriction.json"
image_dir = "./restricted/"

with open(json_path,'r') as load_f:
    load_dict = json.load(load_f)
annotations = load_dict["annotations"]

for annotation in annotations:
    image_id = annotation["image_id"]
    image_name = str(image_id) + ".jpg"
    image_path = os.path.join(image_dir, image_name)
    img = cv2.imread(image_path)

    segmentation = annotation["segmentation"]
    pts = np.array([segmentation], np.int32)
    pts = pts.reshape((-1,1,2))

    img = cv2.polylines(img, [pts], True, (0, 0, 0))
    cv2.imwrite(image_path, img)