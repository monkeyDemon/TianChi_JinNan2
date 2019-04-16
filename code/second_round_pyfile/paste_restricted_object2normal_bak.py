# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:31:02 2019

data augmentation

@author: zyb_as
"""
from __future__ import division
import os
import math
import copy
import cv2
import random
import numpy as np
import argparse, textwrap
import json
import collections

from poi_within_polygon import isPointWithinPoly



def convert_segmentation2polygen(segmentation):
    '''
    convert segmentation to polygen that suitable for function isPointWithinPoly
    '''
    # 将segmentation 构造成如下形式
    #poly= [ [[x1,y1],[x2,y2],……,[xn,yn],[x1,y1]], ... ]
    poly = copy.deepcopy(segmentation)
    poly.append(poly[0])
    poly.append(poly[1])
    poly = np.asarray(poly)
    poly = poly.reshape((1, -1, 2))
    return poly


def visual_augmentation_result(img, category_id, segmentation):
    color_list = [[0,0,0], [0,0,0], [255,0,0], 
                  [128,128,128], [0,255,0], [0,0,255]]
    draw_width = 3
    
    # draw bbox
    '''
    bbox = bbox_info['bbox']
    pt1 = (bbox[0], bbox[1])
    pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
    img = cv2.rectangle(img.copy(), pt1, pt2, color, draw_width)
    '''
    # draw segmentation
    color = color_list[category_id]
    pts = np.array([segmentation], np.int32)
    pts = pts.reshape((-1,2))
    img = cv2.polylines(img.copy(), [pts], True, color, draw_width)
    return img


# -----------------------------------------------------------------------------
        
# set options
parser = argparse.ArgumentParser(
    description='verify the format of the specify image dataset',
    usage=textwrap.dedent(
        '''\command example python %(prog)s --src_dir='src_directory' --dst_dir='dst_director' '''),
    formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument('--normal_dir', type=str, default=None,
                    help='the directory of the normal images.')
#parser.add_argument('--restricted_dir', type=str, default=None,
                    #help='the source directory of the extracted restrict images.')
parser.add_argument('--restricted_info_path', type=str, default=None,
                    help='the source path of the npy file to save the restrict objects informations.')
parser.add_argument('--img_save_dir', type=str,
                    default="../../data/ProcessData/train_val_jpg",
                    help='the destination directory to save the restricted objects.')
parser.add_argument('--input_json_path', type = str, default = None,
                    help = 'input_json_path')
parser.add_argument('--output_json_path', type=str, default=None,
                    help='output_json_path')
parser.add_argument('--generate_num', type=int, default=1000,
                    help='generate_num')
parser.add_argument('--max_restricted_num', type=int, default=3,
                    help='max_restricted_num')
parser.add_argument('--min_restricted_num', type=int, default=1,
                    help='min_restricted_num')

# TODO: set parameters
args = parser.parse_args()
normal_img_dir = args.normal_dir
#restricted_object_dir = args.restricted_dir
restricted_info_path = args.restricted_info_path
save_result_dir = args.img_save_dir
input_json_path = args.input_json_path
output_json_path = args.output_json_path
generate_num = args.generate_num
min_restricted_num = args.min_restricted_num
max_restricted_num = args.max_restricted_num 


#normal_img_dir = "./normal/"
#restricted_object_dir = "./restricted_objects/"
#restricted_info_path = './extract_restricted_objects/extract_restricted_info_dict.npy'
#save_result_dir = "./paste_normal_images/"
#input_json_path = "json/train_restriction.json"
#output_json_path = "json/train_restriction_paste_normal.json"
#generate_num = 50


if not os.path.exists(save_result_dir):
    os.makedirs(save_result_dir)

# get init json informations
with open(input_json_path, 'r') as load_f:
    load_dict = json.load(load_f)
annotations = load_dict["annotations"]
images = load_dict["images"]

# get all normal images path list
normal_imgs_list = []
for normal_img in os.listdir(normal_img_dir):
    normal_imgs_list.append(os.path.join(normal_img_dir, normal_img))
normal_total_num = len(normal_imgs_list)

# get all restricted objects info
restricted_info_list = np.load(restricted_info_path)
restricted_total_num = len(restricted_info_list)

# start random paste restricted objects on normal images
cnt = 0
while cnt < generate_num:
    print(cnt)
    # random a dst normal image
    normal_rand_idx = random.randint(0, normal_total_num - 1)
    img_normal_path = normal_imgs_list[normal_rand_idx]
    img_normal = cv2.imread(img_normal_path, 1)

    rows, cols, channel = img_normal.shape

    # random the restricted objects number to paste
    paste_num = random.randint(min_restricted_num, max_restricted_num)

    file_name = ''
    category_list = []
    bbox_list = []
    segmentation_list = []
    area_list= []
    for _ in range(paste_num):
        # paste a restricted object in image img_normal

        # random a restricted object
        restricted_rand_idx = random.randint(0, restricted_total_num - 1)
        restricted_info = restricted_info_list[restricted_rand_idx]
        restricted_info = copy.deepcopy(restricted_info)
        
        restricted_img_path = restricted_info['restricted_img_path']
        restricted_obj_category = restricted_info['category_id']
        bbox = restricted_info['bbox']
        segmentation = restricted_info['segmentation'][0]
        area = restricted_info['area']
        
        # load restricted object image
        restricted_obj = cv2.imread(restricted_img_path, 1)
        res_rows, res_cols, res_channel = restricted_obj.shape
        
        # random the paste position
        safety_boundary = 10
        x_start = random.randint(safety_boundary, cols - res_cols - safety_boundary - 1)
        y_start = random.randint(safety_boundary, rows - res_rows - safety_boundary - 1)
        
        for x_offset in range(res_cols):
            for y_offset in range(res_rows):
                # if current point in object(use segmentation to judge)
                poly = convert_segmentation2polygen(segmentation)
                in_object = isPointWithinPoly([x_offset,y_offset], poly)
                if in_object:
                    img_normal[y_start+y_offset, x_start+x_offset, :] = restricted_obj[y_offset, x_offset, :]
        
        # modify informations
        # modify bbox
        bbox[0] += x_start
        bbox[1] += y_start
        # modify segmentation
        for idx in range(0, len(segmentation), 2):
            segmentation[idx] += x_start
        for idx in range(1, len(segmentation), 2):
            segmentation[idx] += y_start

        bbox_list.append(bbox)
        segmentation_list.append(restricted_info['segmentation'])
        category_list.append(restricted_obj_category)
        area_list.append(area)

    # file_name = str(cnt+10000) + '_' + restricted_object_dir.split('/')[-1] + '_auto_generate.jpg'
    image_id = cnt + 10000
    file_name = str(image_id) + '_paste_normal.jpg'

    # visualize to test segmention
    '''
    for idx, segmentation in enumerate(segmentation_list):
        img_normal = visual_augmentation_result(img_normal, category_list[idx], segmentation)
    '''
     
    # save paste image
    save_path = os.path.join(save_result_dir, file_name)
    cv2.imwrite(save_path, img_normal)

    image_dic = collections.OrderedDict()
    image_dic = {
        "coco_url": "",
        "data_captured": "",
        "file_name": file_name,
        "flickr_url": "",
        "id": image_id,
        "height": rows,
        "width": cols,
        "license": 1
    }
    images.append(image_dic)

    box_len = len(bbox_list)
    for index in range(box_len):
        category_id = category_list[index]
        bbox = bbox_list[index]
        seg = segmentation_list[index]
        area = area_list[index]

        annotation_dic = collections.OrderedDict()
        annotation_dic = {
            "id": len(annotations) + 1,
            "image_id": image_id,
            "category_id": category_id,
            "iscrowd": 0,
            "segmentation": seg,
            "area": area,
            "bbox": bbox,
            "minAreaRect": []
        }
        annotations.append(annotation_dic)
    cnt += 1


# save json result
result = {
    "info": load_dict["info"],
    "licenses": load_dict["licenses"],
    "categories": load_dict["categories"],
    "images": images,
    "annotations": annotations
}

with open(output_json_path, "w") as dump_f:
    json.dump(result, dump_f)
