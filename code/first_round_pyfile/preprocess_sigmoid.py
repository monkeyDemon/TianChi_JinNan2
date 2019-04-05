# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 10:58:54 2019

对数据集进行预处理

通过观察违禁品图像直方图可以发现
图像像素主要集中于255附近，因此我们定义了一种类似于sigmoid函数的变换
将像素值更均匀的映射到0-255之间
该变换在二分类CNN模型的实验中已经证明有明显效果
对于目标检测模型的效果有待验证

@author: zyb_as
"""

import os
from PIL import Image
import numpy as np
import cv2
import argparse, textwrap

parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent(""), 
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--src_dir', type = str, default = None,
        help = 'the directory of the normal images.')    
parser.add_argument('--dst_dir', type = str, default = None,
        help = 'the source directory of the restricted images.')

args = parser.parse_args()
src_dir = args.src_dir
dst_dir = args.dst_dir

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

def modify_sigmoid(img):
    img = img.astype(np.float)
    img = 510 /(1 + np.exp(-(img*5/255 - 5)))
    img = img.astype(np.uint8)
    return img


for img_name in os.listdir(src_dir):
    img_path = os.path.join(src_dir, img_name)
    save_path = os.path.join(dst_dir, img_name)
    img = cv2.imread(img_path, 1)
    
    (b, g, r) = cv2.split(img)
    
    r = modify_sigmoid(r)
    g = modify_sigmoid(g)
    b = modify_sigmoid(b)
    
    
    result = cv2.merge((b, g, r))
    cv2.imwrite(save_path, result)
