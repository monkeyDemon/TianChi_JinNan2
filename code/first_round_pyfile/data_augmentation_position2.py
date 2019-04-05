# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 10:51:59 2019

对数据集随机进行位置相关的数据增强

这些数据增强方法包括：
Zoom
Rotate
等

@author: zyb_as
"""

import os
import cv2
import json
import copy
import random
import collections
import numpy as np
import argparse, textwrap


# set options
parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent('''\
        command example
        python %(prog)s --src_dir='src_directory' --dst_dir='dst_directory(此行待修改)' '''),
        formatter_class = argparse.RawTextHelpFormatter)  
parser.add_argument('--dense_restricted_dir', type = str, default = None,
        help = 'the source directory of the dense restricted images.')
parser.add_argument('--img_save_dir', type = str, default = None,
        help = 'the destination directory to save the restricted objects.')
parser.add_argument('--input_json_path', type = str, default = None,
        help = 'input_json_path')
parser.add_argument('--output_json_path', type = str, default = None,
        help = 'output_json_path')
parser.add_argument('--generate_num', type = int, default = 1000,
        help = 'generate_num')
parser.add_argument('--test_mode', type = bool, default = False,
        help = 'is test mode or not')

# TODO: set parameters
args = parser.parse_args()
dense_restricted_dir = args.dense_restricted_dir
save_result_dir = args.img_save_dir
input_json_path = args.input_json_path
output_json_path = args.output_json_path
generate_num = args.generate_num
test_mode = args.test_mode

#-----------------------------------------------------------------------------

def get_restricted_info(json_path):
    with open(json_path,'r') as load_f:
        load_dict = json.load(load_f)
    
    bbox_num = len(load_dict["annotations"])
    image_num = len(load_dict["images"])
    imgs_name_list = []
    restricted_info_list = []
    for img_idx in range(image_num):
        img_id = load_dict["images"][img_idx]['id']
        img_name = load_dict["images"][img_idx]['file_name']
        imgs_name_list.append(img_name)
        
        restricted_info = []
        for box in range(bbox_num):
            if load_dict["annotations"][box]['image_id'] != img_id:
                continue
            box_dict = {}
            box_dict["category_id"] = load_dict["annotations"][box]["category_id"]
            box_dict["bbox"] = load_dict["annotations"][box]["bbox"]
            box_dict["minAreaRect"] = load_dict["annotations"][box]["minAreaRect"]
            restricted_info.append(box_dict)
        restricted_info_list.append(restricted_info)
    return imgs_name_list, restricted_info_list



def RandomHorizontalFlip(img, bbox_info_list, p):
    rows, cols, channel = img.shape
    has_change = False 
    if random.random() < p:
        # horizontal flip image
        has_change = True
        img = img[:, ::-1, :]
        # horizontal flip bbox
        for bbox_info in bbox_info_list:
            bbox = bbox_info['bbox']
            bbox[0] = cols - bbox[0] - bbox[2]
    return img, bbox_info_list, has_change


def RandomVerticalFlip(img, bbox_info_list, p):
    rows, cols, channel = img.shape
    has_change = False 
    if random.random() < p:
        # horizontal flip image
        has_change = True
        img = img[::-1, :, :]
        # horizontal flip bbox
        for bbox_info in bbox_info_list:
            bbox = bbox_info['bbox']
            bbox[1] = rows - bbox[1] - bbox[3]
    return img, bbox_info_list, has_change


def RandomResize(img, bbox_info_list, p):
    has_change = False
    if random.random() < p:
        has_change = True 
        rows, cols, channel = img.shape
        if random.random() < 0.5:   
            # zoom in
            ratio_min = 0.85 
            resize_ratio = random.uniform(ratio_min, 1)
            new_width = int(cols * resize_ratio)
            new_height = int(rows * resize_ratio)
            # zoom in  image
            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
            # random paste the total image at a new position
            x_offset = random.randint(0, cols - new_width)
            y_offset = random.randint(0, rows - new_height)
            back_img = np.ones((rows,cols,3), dtype=np.uint8) * 255
            back_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = img
            img = back_img
            # horizontal flip bbox
            for bbox_info in bbox_info_list:
                bbox = bbox_info['bbox']
                bbox[0] = int(bbox[0] * resize_ratio) + x_offset
                bbox[1] = int(bbox[1] * resize_ratio) + y_offset
                bbox[2] = int(bbox[2] * resize_ratio)
                bbox[3] = int(bbox[3] * resize_ratio)
        else:
            bbox_info_list_cp = copy.deepcopy(bbox_info_list)
            ratio_min = 0.85
            max_try_num = 10
            try_num = 1
            has_find_crop = False
            while try_num <= max_try_num:
                crop_ratio = random.uniform(ratio_min, 1)
                new_width = int(cols * crop_ratio)
                new_height = int(rows * crop_ratio)
                x_offset = random.randint(0, cols - new_width)
                y_offset = random.randint(0, rows - new_height)
                crop_box = [x_offset, y_offset, x_offset+new_width, y_offset+new_height]
                # check truncation ratio
                crop_suitable = True
                for bbox_info in bbox_info_list_cp:
                    bbox = bbox_info['bbox']
                    truncation_ratio = _compute_truncation_ratio(crop_box, bbox)
                    if truncation_ratio > 0 and truncation_ratio < 0.4:
                        # don't meet the requirement
                        crop_suitable = False
                        break
                if crop_suitable:
                    has_find_crop = True
                    break
                else:
                    try_num += 1
            
            if has_find_crop:
                #print("has_find")
                # crop image
                img = img[crop_box[1]:crop_box[3],crop_box[0]:crop_box[2],:]
                # remapping boungding boxes' coordinates
                bbox_info_list = []
                for bbox_info in bbox_info_list_cp:
                    bbox = bbox_info['bbox']
                    truncation_ratio = _compute_truncation_ratio_and_cut_bbox(crop_box, bbox)
                    if truncation_ratio > 0:
                        bbox_info_list.append(bbox_info)
    return img, bbox_info_list, has_change


def RandomMerge(img1, bbox_info_list1, img2, bbox_info_list2, probability):
    # find split position of image1
    rows1, cols1, channel = img1.shape
    max_try_num = 10
    try_num = 1
    has_find_split = False
    split1_x = 0
    while try_num <= max_try_num:
        crop_ratio = random.uniform(0.3, 0.7)
        split1_x = int(cols1 * crop_ratio)
        split_suitable = True
        for bbox_info in bbox_info_list1:
            bbox = bbox_info['bbox']
            is_trun = is_truncate(split1_x, bbox)
            if is_trun:
                split_suitable = False
                break
        if split_suitable:
            has_find_split = True
            break
        else:
            try_num += 1
    if has_find_split == False:
        return img1, bbox_info_list1
    
    # find split position of image2
    rows2, cols2, channel = img2.shape
    try_num = 1
    has_find_split = False
    split2_x = 0
    while try_num <= max_try_num:
        crop_ratio = random.uniform(0.3, 0.7)
        split2_x = int(cols2 * crop_ratio)
        split_suitable = True
        for bbox_info in bbox_info_list2:
            bbox = bbox_info['bbox']
            is_trun = is_truncate(split2_x, bbox)
            if is_trun:
                split_suitable = False
                break
        if split_suitable:
            has_find_split = True
            break
        else:
            try_num += 1
    if has_find_split == False:
        return img1, bbox_info_list1
    
    # merge image
    left_width = split1_x + 1
    right_width = cols2 - split2_x
    new_height = max(rows1, rows2)
    new_width = left_width + right_width
    new_img = np.ones((new_height, new_width, 3), dtype=np.uint8)
    new_img *= 255
    new_img[0:rows1, 0:left_width, :] = img1[:, 0:left_width, :]
    new_img[0:rows2, left_width:left_width+right_width, :] = img2[:, split2_x:cols2, :]
    
    # merge new bbox list
    new_bbox_list = []
    for bbox_info in bbox_info_list1:
        bbox = bbox_info['bbox']
        if bbox[0] + bbox[2] - 1 <= split1_x:
            new_bbox_list.append(bbox_info)
    for bbox_info in bbox_info_list2:
        bbox = bbox_info['bbox']
        if bbox[0] >= split2_x:
            bbox[0] = bbox[0] - split2_x + left_width
            new_bbox_list.append(bbox_info)
    return new_img, new_bbox_list


def is_truncate(split, bbox):
    if bbox[0] <= split and bbox[0] + bbox[2] - 1 >= split:
        return True
    else:
        return False


def _compute_truncation_ratio(crop_box, bbox):
    '''
    计算 boundingbox 被 cropbox 截断的比例
    bbox和cropbox的结构为：[xmin, ymin, witdth, height]
    '''
    crop_box = copy.deepcopy(crop_box)
    bbox = copy.deepcopy(bbox)
    bbox_area = bbox[2] * bbox[3]
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    crop_box[2] += crop_box[0]
    crop_box[3] += crop_box[1]
    xxmin = max(bbox[0], crop_box[0])
    yymin = max(bbox[1], crop_box[1])
    xxmax = min(bbox[2], crop_box[2])
    yymax = min(bbox[3], crop_box[3])

    # 计算重叠区域的长宽
    w = np.maximum(0, xxmax - xxmin + 1)
    h = np.maximum(0, yymax - yymin + 1)
    
    # 计算重叠区域占bounding box的面积比（bbox被crop_box截断比率）
    truncation_ratio = (w * h) / bbox_area
    return truncation_ratio

def _compute_truncation_ratio_and_cut_bbox(crop_box, bbox):
    crop_box = copy.deepcopy(crop_box)
    bbox_area = bbox[2] * bbox[3]
    bbox[2] += bbox[0]
    bbox[3] += bbox[1]
    crop_box[2] += crop_box[0]
    crop_box[3] += crop_box[1]
    xxmin = max(bbox[0], crop_box[0])
    yymin = max(bbox[1], crop_box[1])
    xxmax = min(bbox[2], crop_box[2])
    yymax = min(bbox[3], crop_box[3])

    # 计算重叠区域的长宽
    w = np.maximum(0, xxmax - xxmin + 1)
    h = np.maximum(0, yymax - yymin + 1)
    
    # 计算重叠区域占bounding box的面积比（bbox被crop_box截断比率）
    truncation_ratio = (w * h) / bbox_area
    
    # 裁剪bbox坐标及宽高
    if truncation_ratio > 0:
        if bbox[0] < crop_box[0]:
            bbox[0] = 0
            bbox[2] = w
        else:
            bbox[0] -= crop_box[0]
            bbox[2] = w
            
        if bbox[1] < crop_box[1]:
            bbox[1] = 0
            bbox[3] = h
        else:
            bbox[1] -= crop_box[1]
            bbox[3] = h
    return truncation_ratio


def visual_augmentation_result(img, bbox_info_list):
    color = [255,0,0]
    draw_width = int(max(img.shape[:2])/300)
    for bbox_info in bbox_info_list:
        bbox = bbox_info['bbox']
        pt1 = (bbox[0], bbox[1])
        pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        img = cv2.rectangle(img.copy(), pt1, pt2, color, draw_width)
    return img

#-----------------------------------------------------------------------------

# get all restricted objects info
imgs_name_list, restricted_info_list = get_restricted_info(input_json_path)
restricted_total_num = len(imgs_name_list)


with open(input_json_path, 'r') as load_f:
    load_dict = json.load(load_f)
images = load_dict["images"]
annotations = load_dict["annotations"]

cnt = 0
start_id = 20000
while cnt < generate_num:
    # random a restricted image to do data augmentation
    rand_idx = random.randint(0, restricted_total_num - 1)
    img_file_name = imgs_name_list[rand_idx]
    # get the image and bounding boxes info
    image_path = os.path.join(dense_restricted_dir, img_file_name)
    restricted_info = restricted_info_list[rand_idx]
    restricted_info = copy.deepcopy(restricted_info) #对象拷贝，深拷贝
    
    # load image
    img = cv2.imread(image_path, 1)
    probability = 0.5
    
    # random horizontal flip
    img, restricted_info, rhf_change = RandomHorizontalFlip(img, restricted_info, probability)

    # random vertical flip
    img, restricted_info, rvf_change = RandomVerticalFlip(img, restricted_info, probability)

    # random resize
    img, restricted_info, rr_change = RandomResize(img, restricted_info, probability)

    if rhf_change or rvf_change or rr_change == False:
        # hasn't do any change, try again
        continue

    # merge image
    if random.random() < probability:
        # random a restricted image to do data augmentation
        rand_idx2 = random.randint(0, restricted_total_num - 1)
        # get another image 
        image_path2 = os.path.join(dense_restricted_dir, imgs_name_list[rand_idx2])
        img2 = cv2.imread(image_path2, 1)
        restricted_info2 = restricted_info_list[rand_idx2]
        restricted_info2 = copy.deepcopy(restricted_info2) #对象拷贝，深拷贝
        img, restricted_info = RandomMerge(img, restricted_info,
                                            img2, restricted_info2, probability)
        if len(restricted_info) == 0:
            continue

    # visual data augmentation result(if choose test mode)
    if test_mode == True:
        img = visual_augmentation_result(img, restricted_info)
        
    # save result image
    img_id = cnt + start_id
    save_name = str(img_id) + '_data_augmentation_position1.jpg'
    #save_name = 'position1_' + imgs_name_list[rand_idx]
    img_save_path = os.path.join(save_result_dir, save_name)
    cv2.imwrite(img_save_path, img)
    
    image_len = len(images)
    image_dic= collections.OrderedDict()
    image_dic = {
        "coco_url": "",
        "data_captured": "",
        "file_name": save_name,
        "flickr_url": "",
        "id": img_id,
        "height": img.shape[0],
        "width": img.shape[1],
        "license": 1
    }
    images.append(image_dic)
    
    for bbox_info in restricted_info:
        bbox = bbox_info['bbox']
        category_id = bbox_info['category_id']
        xmin = bbox[0]
        ymin = bbox[1]
        width = bbox[2]
        height = bbox[3]

        annotation_dic= collections.OrderedDict()
        annotation_dic = {
          "id": len(annotations) + start_id,
          "image_id": img_id,
          "category_id": category_id,
          "iscrowd": 0,
          "segmentation": [],
          "area": [],
          "bbox": [ xmin, ymin, width, height],
          "minAreaRect": []
        }
        annotations.append(annotation_dic)
    
    if cnt % 10 == 0:
        print("has creat {} images".format(cnt))
    cnt += 1

# save json result
result = {
        "info": load_dict["info"],
        "licenses": load_dict["licenses"],
        "categories": load_dict["categories"],
        "images": images,
        "annotations": annotations
        }
with open(output_json_path,"w") as dump_f:
    json.dump(result ,dump_f)
