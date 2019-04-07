"""
Created on Sun Apr  7 11:18:22 2019

extract restrcted objects

save the restricted objects' images and relative infomations

@author: ljy, zyb_as
"""
from __future__ import division
import os
import cv2
import json
import numpy as np
import argparse, textwrap

# set options
parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent('''\
        command example
        python %(prog)s --src_dir='src_directory' --dst_dir='dst_director' '''),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--json_path', type = str, default = None,
        help = 'the path of the modified json file.')
parser.add_argument('--restricted_img_dir', type = str, default = None,
        help = 'the source directory of the restricted images.')
parser.add_argument('--restricted_object_info_save_dir', type = str, default = None,
        help = 'the destination directory to save the restricted objects.')


# set directory parameter
args = parser.parse_args()
json_path = args.json_path
restricted_img_dir = args.restricted_img_dir
restricted_object_info_save_dir = args.restricted_object_info_save_dir

#json_path = "json/train_restriction.json"
#restricted_img_dir = "restricted/"
#restricted_object_info_save_dir = "extract_restricted_objects"
restricted_object_img_save_dir = os.path.join(restricted_object_info_save_dir, 'imgs')


# if the directory to save extracted restrictd objects don't exist, create it
if not os.path.exists(restricted_object_info_save_dir):
    os.mkdir(restricted_object_info_save_dir)
if not os.path.exists(restricted_object_img_save_dir):
    os.mkdir(restricted_object_img_save_dir)

def get_restricted_info(json_path):
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)

    bbox_num = len(load_dict["annotations"]) 
    image_num = len(load_dict["images"])  
    information = []
    for box in range(bbox_num):
        box_dict = {}
        box_dict["restricted_id"] = load_dict["annotations"][box]["id"]
        box_dict["category_id"] = load_dict["annotations"][box]["category_id"]
        image_id = load_dict["annotations"][box]["image_id"]
        box_dict["bbox"] = load_dict["annotations"][box]["bbox"]
        box_dict["area"] = load_dict["annotations"][box]["area"]
        box_dict["segmentation"] = load_dict["annotations"][box]["segmentation"]
        for im in range(image_num):
            if image_id == load_dict["images"][im]["id"]:
                box_dict["original_img_name"] = load_dict["images"][im]["file_name"]
        information.append(box_dict)
    return information


# get all restricted objects info
restricted_info_list = get_restricted_info(json_path)
restricted_total_num = len(restricted_info_list)

# save extract restricted objects and modify relation infomations(such as segmentation)
for idx in range(restricted_total_num):
    print(idx)

    restricted_obj = restricted_info_list[idx]
    img_path = os.path.join(restricted_img_dir, restricted_obj['original_img_name'])

    bbox = restricted_obj['bbox']
    xmin = int(bbox[0])
    ymin = int(bbox[1])
    xmax = int(bbox[0] + bbox[2] - 1)
    ymax = int(bbox[1] + bbox[3] - 1)

    # load image
    img_total = cv2.imread(img_path, 1)
    # extract restricted
    restricted_obj_img = img_total[ymin : ymax + 1, xmin : xmax + 1,:]

    # save extract restricted objects
    save_name = str(restricted_obj['category_id']) + '_' + str(restricted_obj['restricted_id']) + '.jpg'
    save_path = os.path.join(restricted_object_img_save_dir, save_name)
    cv2.imwrite(save_path, restricted_obj_img)
    
    # modify relation infomations
    # modify bbox and segmentation, keep area and category_id unchanged
    # add restrcted object image name
    restricted_obj['restricted_img_path'] = save_path
    # modify bbox
    bbox[0] = 0
    bbox[1] = 0
    # modify segmentation
    segmentation = restricted_obj['segmentation'][0]
    for idx in range(0, len(segmentation), 2):
        segmentation[idx] -= xmin
    for idx in range(1, len(segmentation), 2):
        segmentation[idx] -= ymin

# Save restricted objects infomation
restricted_info_save_path = os.path.join(restricted_object_info_save_dir, 'extract_restricted_info_dict.npy')
np.save(restricted_info_save_path, restricted_info_list) 
    
