# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:31:02 2019

data augmentation

@author: zyb_as
"""
from __future__ import division
import os
import math
import cv2
import random
import numpy as np
import argparse, textwrap
import json
import collections


# set options
parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent('''\command example python %(prog)s --src_dir='src_directory' --dst_dir='dst_directory(此行待修改)' '''),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--normal_dir', type = str, default = None,
        help = 'the directory of the normal images.')      
parser.add_argument('--restricted_dir', type = str, default = None,
        help = 'the source directory of the restricted images.')
parser.add_argument('--img_save_dir', type = str, default = "../../data/ProcessData/train_val_jpg",
        help = 'the destination directory to save the restricted objects.')
parser.add_argument('--input_json_path', type = str, default = None,
        help = 'input_json_path')
parser.add_argument('--output_json_path', type = str, default = None,
        help = 'output_json_path')
parser.add_argument('--generate_num', type = int, default = 1000,
        help = 'generate_num')
parser.add_argument('--max_restricted_num', type = int, default = 6,
        help = 'max_restricted_num')
parser.add_argument('--min_restricted_num', type = int, default = 1,
        help = 'min_restricted_num')
parser.add_argument('--is_diffcult', type = bool, default = False,
        help = 'diffcult')



# TODO: set parameters
args = parser.parse_args()
normal_img_dir = args.normal_dir
restricted_object_dir = args.restricted_dir
save_result_dir = args.img_save_dir
if not os.path.exists(save_result_dir):
    os.makedirs(save_result_dir)

json_save_path = args.output_json_path
init_json_path = args.input_json_path
if args.is_diffcult:
    generate_num = args.generate_num + 10000
else:
    generate_num = args.generate_num  # TODO: modify
min_restricted_num = args.min_restricted_num
max_restricted_num = args.max_restricted_num  # 生成的图中违禁品的最大个数
category_dict = {1:'Iron-shell lighter',2:'Nail lighter',3:'Knives',4:'Batteries',5:'Scissors'}


def rotate_min_rect(min_rect, centor_point, angle):
    '''旋转最小矩形框
    
    Arguments:
        min_rect: 待旋转的最小矩形框
        centor_point: 旋转中心点
        theta: 旋转角度
    
    Returns:
        min_rect_rotate: 旋转后的最小矩形框
    '''
    angle = -1 * angle
    angle = angle * math.pi / 180
    min_rect_rotate = []
    for p in min_rect:
        x = p[0]
        y = p[1]
        new_p = [0, 0]
        new_p[0] = (x - centor_point[0]) * math.cos(angle) - (y - centor_point[1]) * math.sin(angle) + centor_point[0]
        new_p[1] = (x - centor_point[0]) * math.sin(angle) + (y - centor_point[1]) * math.cos(angle) + centor_point[1]
        new_p[0] = int(new_p[0])
        new_p[1] = int(new_p[1])
        min_rect_rotate.append(new_p)
    return min_rect_rotate


def get_coordinate_after_rotate(relative_min_rect_rotate, x_relative, y_relative, rect_len):
    centor = int(rect_len / 2)
    x_relative += centor
    y_relative += centor
    min_rect_rotate = []
    for p_relative in relative_min_rect_rotate:
        p = [0, 0]
        p[0] = p_relative[0] + x_relative
        p[1] = p_relative[1] + y_relative
        min_rect_rotate.append(p)
    return min_rect_rotate


def get_bounding_box(min_rect_rotate):
    '''截取违禁品目标
    
    Arguments:
        total_img:
        min_rect_rotate: 旋转后违禁品所在区域
    Returns:
        restricted_obj_img:
    '''
    p1 = min_rect_rotate[0]
    p2 = min_rect_rotate[1]
    p3 = min_rect_rotate[2]
    p4 = min_rect_rotate[3]
    x_list = [p1[0], p2[0], p3[0], p4[0]]
    y_list = [p1[1], p2[1], p3[1], p4[1]]
    x_min = min(x_list)
    x_max = max(x_list)
    y_min = min(y_list)
    y_max = max(y_list)
    boundding_box = [x_min, y_min, x_max, y_max]
    return boundding_box



    

# get all normal images path list
normal_total_num = len(os.listdir(normal_img_dir))
normal_imgs_list = []
for normal_img in os.listdir(normal_img_dir):
    normal_imgs_list.append(os.path.join(normal_img_dir, normal_img))

# get all restricted objects info
restricted_total_num = len(os.listdir(restricted_object_dir))
restricted_object_list = []
for restricted_object in os.listdir(restricted_object_dir):
    restricted_object_list.append(os.path.join(restricted_object_dir, restricted_object))
if args.is_diffcult:
    cnt = 10000
else:
    cnt = 0
with open(init_json_path,'r') as load_f:
    load_dict = json.load(load_f)
images = load_dict["images"]
annotations = load_dict["annotations"]

while cnt < generate_num:
    
    try:
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
        minAreaRect_list = []
        for _ in range(paste_num):
            # paste a restricted object in image img_normal
            
            # random a restricted object
            restricted_rand_idx = random.randint(0, restricted_total_num - 1)
            restricted_img_path = restricted_object_list[restricted_rand_idx]
            restricted_obj_category = int(restricted_img_path[-5])
    
            # load restricted object image
            restricted_obj = cv2.imread(restricted_img_path, 1)
            
            # random resize
            # TODO:
            
            obj_rows, obj_cols, obj_channel = restricted_obj.shape
            rect_len = 0
            if obj_rows < obj_cols:
                rect_len = int(obj_cols * np.sqrt(2))
            else:
                rect_len = int(obj_rows * np.sqrt(2))
            centor = int(rect_len / 2)
            half_x = int(obj_cols / 2)
            half_y = int(obj_rows / 2)
            relative_min_rect = [[-half_x, -half_y], [half_x, -half_y],
                                 [half_x, half_y], [-half_x, half_y]]
            
            # padding
            # 240,240,223 # TODO: 解决黑边
            padd_color = (223, 240, 240) # channel order(cv order): B G R
            padding_bak = np.ones((rect_len,rect_len,3), dtype=np.uint8)
            padding_bak[:,:,0] = padd_color[0]
            padding_bak[:,:,1] = padd_color[1]
            padding_bak[:,:,2] = padd_color[2]
            padding_bak[centor-half_y:centor-half_y+obj_rows, centor-half_x:centor-half_x+obj_cols, :] = restricted_obj
            restricted_obj = padding_bak
            
            # random rotate restricted object
            angle = random.randint(0, 360)
            affine_shrink_rotation = cv2.getRotationMatrix2D((centor, centor), angle, 1)
            restricted_obj_rotate = cv2.warpAffine(restricted_obj, affine_shrink_rotation, 
                                        (rect_len, rect_len), borderValue=padd_color)
            relative_min_rect_rotate = rotate_min_rect(relative_min_rect, [0, 0], angle)
            
            # random a coordinate to paste
            rand_x = random.randint(0, cols - rect_len - 1)
            rand_y = random.randint(0, rows - rect_len - 1)
            for y_offset in range(rect_len):
                for x_offset in range(rect_len):
                    pix = restricted_obj_rotate[y_offset, x_offset, :]
                    if pix[0] == padd_color[0] and pix[1] == padd_color[1] and pix[2] == padd_color[2]:
                        continue
                    img_normal[rand_y+y_offset, rand_x+x_offset, :] = pix * 0.95 + img_normal[rand_y+y_offset, rand_x+x_offset, :] * 0.1
            
            
            min_rect_rotate = get_coordinate_after_rotate(relative_min_rect_rotate, rand_x, rand_y, rect_len)
            minAreaRect_list.append(min_rect_rotate)
            
            bbox = get_bounding_box(min_rect_rotate)
            bbox_list.append(bbox)
    
            category_list.append(restricted_obj_category)
    except:
        continue
        
        
    #file_name = str(cnt+10000) + '_' + restricted_object_dir.split('/')[-1] + '_auto_generate.jpg'
    file_name = str(cnt+10000) + '_generate.jpg'
    #print(file_name)
    # rows height
    # cols width
    # bbox_list [[xmin ymin xmax ymax], [], ...]
    
    
    save_path = os.path.join(save_result_dir, file_name)
    #print(save_path)
    cv2.imwrite(save_path, img_normal)
        

    image_len = len(images)
    image_dic= collections.OrderedDict()
    image_dic = {
      "coco_url": "",
      "data_captured": "",
      "file_name": str(cnt+10000) + '_generate.jpg',
      "flickr_url": "",
      "id": image_len + 10000,
      "height": rows,
      "width": cols,
      "license": 1
      }
    images.append(image_dic)
    
    box_len = len(bbox_list)
    for index in range(box_len):
        xmin = bbox_list[index][0]
        ymin = bbox_list[index][1]
        xmax = bbox_list[index][2]
        ymax = bbox_list[index][3]
        
        annotation_dic= collections.OrderedDict()
        annotation_dic = {
          "id": len(annotations) + 10000,
          "image_id": image_len + 10000,
          "category_id": category_list[index],
          "iscrowd": 0,
          "segmentation": [],
          "area": [],
          "bbox": [ xmin, ymin, xmax-xmin, ymax-ymin ],
          "minAreaRect": []
        }
        annotations.append(annotation_dic)
    if cnt % 10 == 0:
        if args.is_diffcult:
            print("has creat {} images".format(cnt-10000))
        else:
            print("has creat {} images".format(cnt))
    cnt += 1
    
result = {
        "info": load_dict["info"],
        "licenses": load_dict["licenses"],
        "categories": load_dict["categories"],
        "images": images,
        "annotations": annotations
        }

with open(json_save_path,"w") as dump_f:
    json.dump(result ,dump_f)
