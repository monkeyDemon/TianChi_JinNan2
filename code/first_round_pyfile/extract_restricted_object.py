# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 21:39:20 2019

extract restricted object

@author: zyb_as
"""
from __future__ import division
import os
import cv2
import math
import json
import argparse, textwrap

# set options
parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent('''\
        command example
        python %(prog)s --src_dir='src_directory' --dst_dir='dst_directory(此行待修改)' '''),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--json_path', type = str, default = None,
        help = 'the path of the modified json file.')      
parser.add_argument('--restricted_img_dir', type = str, default = None,
        help = 'the source directory of the restricted images.')
parser.add_argument('--restricted_object_save_dir', type = str, default = None,
        help = 'the destination directory to save the restricted objects.')


# set directory parameter
args = parser.parse_args()
json_path = args.json_path
restricted_img_dir = args.restricted_img_dir
restricted_object_save_dir = args.restricted_object_save_dir

if not os.path.exists(restricted_object_save_dir):
    os.mkdir(restricted_object_save_dir)

def get_restricted_info(json_path):
    with open(json_path, 'r') as load_f:
        load_dict = json.load(load_f)
    
    bbox_num = len(load_dict["annotations"])
    image_num = len(load_dict["images"])
    information = []
    for box in range(bbox_num):
        box_dict = {}
        box_dict["category_id"] = load_dict["annotations"][box]["category_id"]
        image_id = load_dict["annotations"][box]["image_id"]
        box_dict["bbox"] = load_dict["annotations"][box]["bbox"]
        box_dict["minAreaRect"] = load_dict["annotations"][box]["minAreaRect"]
        for im in range(image_num):
            if image_id == load_dict["images"][im]["id"]:
                box_dict["filename"] = load_dict["images"][im]["file_name"]
        information.append(box_dict)
    return information

# 计算给定两点的连线与x轴正半轴夹角
def _compute_theta(p1, p2):
    tan_theta = (p2[1] - p1[1]) / (p2[0] - p1[0] + 0.00001)
    theta = math.atan(tan_theta)
    theta = int(theta / math.pi * 180)
    return theta

# 计算违禁品minAreaRect四条边中斜率最小的那条边并返回
def compute_min_theta(minAreaRect):
    p1 = minAreaRect[0]
    p2 = minAreaRect[1]
    p3 = minAreaRect[2]
    edge_12_theta = _compute_theta(p1, p2)
    edge_23_theta = _compute_theta(p2, p3)
    if abs(edge_12_theta) < abs(edge_23_theta):
        return edge_12_theta
    else:
        return edge_23_theta

def compute_center_point(minAreaRect):
    p1 = minAreaRect[0]
    p2 = minAreaRect[1]
    p3 = minAreaRect[2]
    p4 = minAreaRect[3]
    x_sum = p1[0] + p2[0] + p3[0] + p4[0]
    y_sum = p1[1] + p2[1] + p3[1] + p4[1]
    centor_p = (x_sum/4, y_sum/4)
    return centor_p


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


def crop_restricted_object(total_img, min_rect_rotate):
    '''截取违禁品目标
    
    Arguments:
        total_img:
        min_rect_rotate: 旋转后违禁品所在区域
    Returns:
        restricted_obj_img:
    '''
    rows, cols, channel = total_img.shape
    min_x = cols
    min_y = rows
    max_x = 0
    max_y = 0
    for p in min_rect_rotate:
        if p[0] < min_x:
            min_x = p[0]
        if p[0] > max_x:
            max_x = p[0]
        if p[1] < min_y:
            min_y = p[1]
        if p[1] > max_y:
            max_y = p[1]
    # TODO: 这一步不一定需要，甚至有可能导致误差
    # 如果违禁品是小物体，适当扩展外边框
    area = (max_x - min_x) * (max_y - min_y)
    if area < 2000:
        min_x -= 1
        max_x += 1
        min_y -= 1
        max_y += 1
    # crop
    restricted_obj_img = total_img[min_y:max_y, min_x:max_x, :]
    return restricted_obj_img

# ----------------------------------------------------------------------------


# get all restricted objects info
restricted_info_list = get_restricted_info(json_path)
restricted_total_num = len(restricted_info_list)


for idx in range(restricted_total_num):
    print(idx)
    
    restricted_obj = restricted_info_list[idx]
    img_path = os.path.join(restricted_img_dir, restricted_obj['filename'])
    min_rect = restricted_obj['minAreaRect']

    # load image
    img_total = cv2.imread(img_path, 1)
    rows, cols, channel = img_total.shape
    # compute rotate angle
    theta = compute_min_theta(min_rect)
    rotate_angle = 1 * theta
    # compute rotate center point
    centor_point = compute_center_point(min_rect)
    # rotate
    # rotate restricted object min area rectangle
    min_rect_rotate = rotate_min_rect(min_rect, centor_point, rotate_angle)
    # rotate the total image
    #affine_shrink_rotation = cv2.getRotationMatrix2D((cols/2, rows/2), 30, 1)
    affine_shrink_rotation = cv2.getRotationMatrix2D(centor_point, rotate_angle, 1)
    img_total_rotate = cv2.warpAffine(img_total, affine_shrink_rotation, 
                                (cols, rows), borderValue=(0,0,0))
    # crop the restricted object after rotate
    restricted_obj_img = crop_restricted_object(img_total_rotate, min_rect_rotate)
    
    save_path = str(idx) + '_' + str(restricted_obj['category_id']) + '.jpg'
    save_path = os.path.join(restricted_object_save_dir, save_path)
    cv2.imwrite(save_path, restricted_obj_img)
    
