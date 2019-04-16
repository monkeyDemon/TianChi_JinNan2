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
from __future__ import division
import os
import cv2
import json
import copy
import math
import random
import collections
import numpy as np
import argparse, textwrap


# set options
parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent('r'),
        formatter_class = argparse.RawTextHelpFormatter)  
parser.add_argument('--restricted_dir', type = str, default = None,
        help = 'the source directory of the restricted images.')
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
restricted_dir = args.restricted_dir
save_result_dir = args.img_save_dir
input_json_path = args.input_json_path
output_json_path = args.output_json_path
generate_num = args.generate_num
test_mode = args.test_mode

#restricted_dir = 'restricted'
#save_result_dir = 'result'
#input_json_path = 'json/train_restriction.json'
#output_json_path = 'json/result.json'
#generate_num = 1000
#test_mode = False


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
            box_dict["area"] = load_dict["annotations"][box]["area"]
            box_dict["segmentation"] = load_dict["annotations"][box]["segmentation"]
            restricted_info.append(box_dict)
        restricted_info_list.append(restricted_info)
    return imgs_name_list, restricted_info_list



def RandomHorizontalFlip(img, bbox_info_list, p):
    rows, cols, channel = img.shape
    
    if random.random() < p:
        # horizontal flip image
        img = img[:, ::-1, :]
        # horizontal flip bbox and segmentation
        for bbox_info in bbox_info_list:
            # flip bbox
            bbox = bbox_info['bbox']
            bbox[0] = cols - bbox[0] - bbox[2]
            
            # flip segmentation
            segmentation = bbox_info['segmentation'][0]
            for idx in range(0, len(segmentation), 2):
                segmentation[idx] = cols - segmentation[idx] - 1
         
    return img, bbox_info_list


def RandomVerticalFlip(img, bbox_info_list, p):
    rows, cols, channel = img.shape
    
    if random.random() < p:
        # horizontal flip image
        img = img[::-1, :, :]
        # horizontal flip bbox and segmentation
        for bbox_info in bbox_info_list:
            # flip bbox
            bbox = bbox_info['bbox']
            bbox[1] = rows - bbox[1] - bbox[3]
            
            # flip segmentation
            segmentation = bbox_info['segmentation'][0]
            for idx in range(1, len(segmentation), 2):
                segmentation[idx] = rows - segmentation[idx] - 1
    return img, bbox_info_list


def RandomZoomOut(img, bbox_info_list, p):
    '''random zoom out
    Here we won't change the image size, but do operation similar as pulling 
    the angle of view away. That is because the object detection network will 
    scale the input image to a fixed scale, just resize the image to a small 
    size actually do nothing.
    '''
    if random.random() < p:
        rows, cols, channel = img.shape
        # zoom out
        ratio_min = 0.8       
        resize_ratio = random.uniform(ratio_min, 1)
        new_width = int(cols * resize_ratio)
        new_height = int(rows * resize_ratio)
        # zoom out image
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
        # random paste the total image at a new position
        x_offset = random.randint(0, cols - new_width)
        y_offset = random.randint(0, rows - new_height)
        back_img = np.ones((rows,cols,3), dtype=np.uint8) * 255
        back_img[y_offset:y_offset+new_height, x_offset:x_offset+new_width, :] = img
        img = back_img
        # zoom out related information 
        for bbox_info in bbox_info_list:
            # zoom out bbox
            bbox = bbox_info['bbox']
            bbox[0] = int(bbox[0] * resize_ratio) + x_offset
            bbox[1] = int(bbox[1] * resize_ratio) + y_offset
            bbox[2] = int(bbox[2] * resize_ratio)
            bbox[3] = int(bbox[3] * resize_ratio)
            # zoom out segmentation
            segmentation = bbox_info['segmentation'][0]
            for idx in range(0, len(segmentation), 2):
                segmentation[idx] = int(segmentation[idx] * resize_ratio) + x_offset
            for idx in range(1, len(segmentation), 2):
                segmentation[idx] = int(segmentation[idx] * resize_ratio) + y_offset
            # zoom out area
            bbox_info['area'] = int(bbox_info['area'] * resize_ratio)
    return img, bbox_info_list


def RandomCrop(img, bbox_info_list, p, ratio_min = 0.8, max_try_num = 10):
    '''random crop
    In general, because the object detection network will scale the input image
    to a fixed scale, random crop part region in the image can play a role like
    zoom in the objects size.
    Arguments:
        img:
        bbox_info_list: infomations include bbox, segmentation, area and so on
        p: the probability to do data augmentation
        ratio_min: the min size of the crop box relative to the image size
        max_try_num: if the random crop box intersect with any bbox, retry.
    '''
    if random.random() < p:
        rows, cols, channel = img.shape
        bbox_info_list_cp = copy.deepcopy(bbox_info_list)
        try_num = 1
        has_find_crop = False
        while try_num <= max_try_num:
            crop_ratio = random.uniform(ratio_min, 1)
            new_width = int(cols * crop_ratio)
            new_height = int(rows * crop_ratio)
            x_offset = random.randint(0, cols - new_width)
            y_offset = random.randint(0, rows - new_height)
            crop_box = [x_offset, y_offset, new_width, new_height]
            # check truncation ratio
            crop_suitable = True
            for bbox_info in bbox_info_list_cp:
                bbox = bbox_info['bbox']
                intersect = _is_bbox_intersect_cropbox(crop_box, bbox)
                if intersect == True:
                    # don't meet the requirement
                    crop_suitable = False
                    break
            if crop_suitable:
                has_find_crop = True
                break
            else:
                try_num += 1
        
        if has_find_crop:
            # crop image
            img_crop = img[crop_box[1]:crop_box[1]+crop_box[3]-1, crop_box[0]:crop_box[0]+crop_box[2]-1, :]
            # remapping boungding boxes' coordinates
            _bbox_info_list = []
            for bbox_info in bbox_info_list_cp:
                bbox = bbox_info['bbox']
                segmentation = bbox_info['segmentation'][0]
                in_crop_bbox = _crop_bbox_segmentation(crop_box, bbox, segmentation)
                if in_crop_bbox:
                    _bbox_info_list.append(bbox_info)
            if len(_bbox_info_list) != 0:
                bbox_info_list = _bbox_info_list
                img = img_crop
    return img, bbox_info_list


def RandomRotate(img, bbox_info_list, p):
    if random.random() < p:
        # get shape
        rows, cols, channel = img.shape
        # random rotate angle
        angle = random.randint(-2,2)
        # rotate image
        padd_color = (255, 255, 255)
        centor_x = int(cols / 2)
        centor_y = int(rows / 2)
        affine_shrink_rotation = cv2.getRotationMatrix2D((centor_x, centor_y), angle, 1)
        img_rotate = cv2.warpAffine(img, affine_shrink_rotation, 
                                        (cols, rows), borderValue=padd_color)
        # rotate bbox coordinate
        for bbox_info in bbox_info_list:
            min_rect = bbox_info['minAreaRect']
            min_rect_rotate = _rotate_min_rect(min_rect, [centor_x, centor_y], angle)
            bbox = get_bbox_by_min_rect(min_rect_rotate)
            bbox_info['bbox'] = bbox
        
        return img_rotate, bbox_info_list
    else:   
        return img, bbox_info_list
    
    
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

def _is_bbox_intersect_cropbox(crop_box, bbox):
    '''judge if the bbox is intersect with cropbox
    first compute the truncation_ratio between boundingbox and cropbox
    if truncation_ratio = 1, bbox is in crop box
    if truncation_ratio = 0, bbox is out of crop box
    if 0 < truncation_ratio < 1, bbox is intersect with crop box
    Arguments:
        bbox: [xmin, ymin, witdth, height]
        cropbox：[xmin, ymin, witdth, height]
    Returns:
        intersect: if the bbox is intersect with cropbox
    '''
    crop_box = copy.deepcopy(crop_box)
    bbox = copy.deepcopy(bbox)
    bbox_area = bbox[2] * bbox[3]
    bbox[2] += bbox[0] - 1
    bbox[3] += bbox[1] - 1
    crop_box[2] += crop_box[0] - 1
    crop_box[3] += crop_box[1] - 1
    xxmin = max(bbox[0], crop_box[0])
    yymin = max(bbox[1], crop_box[1])
    xxmax = min(bbox[2], crop_box[2])
    yymax = min(bbox[3], crop_box[3])

    # 计算重叠区域的长宽
    w = np.maximum(0, xxmax - xxmin + 1)
    h = np.maximum(0, yymax - yymin + 1)
    
    # 计算重叠区域占bounding box的面积比（bbox被crop_box截断比率）
    truncation_ratio = (w * h) / bbox_area
    
    intersect = True
    if truncation_ratio == 0 or truncation_ratio == 1:
        intersect = False
    return intersect


def _crop_bbox_segmentation(crop_box, bbox, segmentation):
    '''
    modify bbox and segmentation's infomation after crop
    Arguments:
        crop_box: x, y, width, height of the crop box
        bbox: x, y, width, height of the bounding box
        segmentation: [x1, y1, x2, y2, ...]
    returns:
        in_crop_box: identify whether bbox is in the crop box
                      if true, modify bbox and segmentation's infomation
    '''
    crop_box = copy.deepcopy(crop_box)
    crop_box[2] += crop_box[0] - 1
    crop_box[3] += crop_box[1] - 1
    
    in_crop_box = False
    if bbox[0] >= crop_box[0] and bbox[0] < crop_box[2] and \
        bbox[1] >= crop_box[1] and bbox[1] < crop_box[3]:
        in_crop_box = True
    
    if in_crop_box:
        x_offset = crop_box[0]
        y_offset = crop_box[1]
        # modify bbox
        bbox[0] -= x_offset
        bbox[1] -= y_offset
        # modify segmentation
        for idx in range(0, len(segmentation), 2):
            segmentation[idx] -= x_offset
        for idx in range(1, len(segmentation), 2):
            segmentation[idx] -= y_offset
    return in_crop_box



def _rotate_min_rect(min_rect, centor_point, angle):
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


def get_bbox_by_min_rect(min_rect):
    min_rect = np.asarray(min_rect)
    x_min = min(min_rect[:,0])
    x_max = max(min_rect[:,0])
    y_min = min(min_rect[:,1])
    y_max = max(min_rect[:,1])
    bbox = [x_min, y_min, x_max - x_min + 1, y_max - y_min + 1]
    return bbox


def visual_augmentation_result(img, bbox_info_list):
    color_list = [[0,0,0], [0,0,0], [255,0,0], 
                  [128,128,128], [0,255,0], [0,0,255]]
    draw_width = 3
    for bbox_info in bbox_info_list:
        # draw bbox
        '''
        bbox = bbox_info['bbox']
        pt1 = (bbox[0], bbox[1])
        pt2 = (bbox[0] + bbox[2], bbox[1] + bbox[3])
        img = cv2.rectangle(img.copy(), pt1, pt2, color, draw_width)
        '''
        # draw segmentation
        category_id = bbox_info['category_id']
        color = color_list[category_id]
        segmentation = bbox_info['segmentation'][0]
        pts = np.array([segmentation], np.int32)
        pts = pts.reshape((-1,2))
        img = cv2.polylines(img.copy(), [pts], True, color, draw_width)
    return img

#-----------------------------------------------------------------------------

# get all restricted objects info
imgs_name_list, restricted_info_list = get_restricted_info(input_json_path)
imgs_total_num = len(imgs_name_list)

# load the current images and annotations
# the new images generate by data augmentation will append to it
with open(input_json_path, 'r') as load_f:
    load_dict = json.load(load_f)
images = load_dict["images"]
annotations = load_dict["annotations"]

cnt = 0
start_id = 20000
while cnt < generate_num:
    print(cnt)
    # random a restricted image to do data augmentation
    rand_idx = random.randint(0, imgs_total_num - 1)
    
    # get the image and bounding boxes info
    image_path = os.path.join(restricted_dir, imgs_name_list[rand_idx])
    restricted_info = restricted_info_list[rand_idx]
    # deep copy restricted info, avoid to destroy original infomation
    restricted_info = copy.deepcopy(restricted_info) 

    # load image
    img = cv2.imread(image_path, 1)
    probability = 0.5
    
    # random horizontal flip
    img, restricted_info = RandomHorizontalFlip(img, restricted_info, probability)

    # random vertical flip
    img, restricted_info = RandomVerticalFlip(img, restricted_info, probability)

    # random resize
    if random.random() < 0.5:
        # zoom out
        img, restricted_info = RandomZoomOut(img, restricted_info, probability)
    else:
        # crop, it's equivalent to zoom in.
        img, restricted_info = RandomCrop(img, restricted_info, probability)
    
    # random rotate
    #img, restricted_info = RandomRotate(img, restricted_info, probability)
    
    # random merge image
    '''
    if random.random() < probability:
        # random a restricted image to do data augmentation
        rand_idx2 = random.randint(0, imgs_total_num - 1)
        # get another image 
        image_path2 = os.path.join(restricted_dir, imgs_name_list[rand_idx2])
        img2 = cv2.imread(image_path2, 1)
        restricted_info2 = restricted_info_list[rand_idx2]
        restricted_info2 = copy.deepcopy(restricted_info2) #对象拷贝，深拷贝
        img, restricted_info = RandomMerge(img, restricted_info,
                                            img2, restricted_info2, probability)
        if len(restricted_info) == 0:
            continue
    '''

    # visual data augmentation result(if choose test mode)
    if test_mode == True:
        img = visual_augmentation_result(img, restricted_info)
        
    # save result image
    img_id = cnt + start_id
    save_name = str(img_id) + '_data_augmentation_position1.jpg'
    #save_name = imgs_name_list[rand_idx][:-4] +'_' + str(img_id) + '.jpg'
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
        segmentation = bbox_info['segmentation']
        area = bbox_info['area']
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
          "segmentation": segmentation,
          "area": area,
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
with open(output_json_path, "w") as dump_f:
    json.dump(result ,dump_f)
