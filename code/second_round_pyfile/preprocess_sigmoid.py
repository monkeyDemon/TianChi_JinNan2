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
import copy
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


# 尝试各种预处理变换方法

# 类似sigmoid的变换
def modify_sigmoid(img):
    img = copy.deepcopy(img)
    img = img.astype(np.float)
    img = 510 /(1 + np.exp(-(img*5/255 - 5)))
    img = img.astype(np.uint8)
    return img

# 类似sigmoid的变换,先进行高斯滤波
def modify_sigmoid_gauss(img):
    img = copy.deepcopy(img)
    img = gaussian_filter(img)
    img = img.astype(np.float)
    img = 510 /(1 + np.exp(-(img*5/255 - 5)))
    img = img.astype(np.uint8)
    return img

# 多项式变换
def polynomial_transfor(img, order):
    # y = scale * x^order
    scale = 255.0 / (255.0 ** order)
    img = copy.deepcopy(img)
    img = img.astype(np.float)
    img = scale * (img ** order)
    # 截断因数值运算误差产生的越界元素
    img[img > 255] = 255
    img[img < 0] = 0
    img = np.round(img)
    img = img.astype(np.uint8)
    return img


# 高斯滤波
def gaussian_filter(image):
    dst = cv2.GaussianBlur(image, (3, 3), 0)  
    return dst


# 中值滤波
def median_filter(image):
    dst = cv2.medianBlur(image, 3)
    return dst


# 对比度增强1
def contrast_brightness_image(img):
    img = copy.deepcopy(img)
    height, width, _ = img.shape#获取shape的数值，height和width、通道    
    img = img.astype(np.float32)
    (b, g, r) = cv2.split(img)
    
    # 将图像均值变换为median
    median_b = 128
    median_g = 128
    median_r = 128
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    b_shift = b_mean - median_b
    g_shift = g_mean - median_g
    r_shift = r_mean - median_r
    b -= b_shift
    g -= g_shift
    r -= r_shift
    
    # 增强对比度 
    contrast_b = (255-median_b)/(255-b_mean)
    contrast_g = (255-median_g)/(255-g_mean)
    contrast_r = (255-median_r)/(255-r_mean)
    for y in range(height):
        for x in range(width):
            if b[y][x] > median_b:
                b[y][x] += (b[y][x] - median_b) * contrast_b
    for y in range(height):
        for x in range(width):
            if g[y][x] > median_g:
                g[y][x] += (g[y][x] - median_g) * contrast_g
    for y in range(height):
        for x in range(width):
            if r[y][x] > median_r:
                r[y][x] += (r[y][x] - median_r) * contrast_r
    '''         
    b = b + (b - median_b) * contrast_b
    g = g + (g - median_g) * contrast_g
    r = r + (r - median_r) * contrast_r
    '''
    result = cv2.merge((b, g, r))
    
    # 截断越界元素，取整，转换类型
    result[result > 255] = 255
    result[result < 0] = 0
    result = np.round(result)
    result = result.astype(np.uint8)
    return result


# 对比度增强2
def contrast_brightness_image2(img):
    img = copy.deepcopy(img)
    img = gaussian_filter(img)
    height, width, _ = img.shape#获取shape的数值，height和width、通道    
    img = img.astype(np.float32)
    (b, g, r) = cv2.split(img)
    
    # 将图像均值变换为median
    median_b = 128
    median_g = 128
    median_r = 128
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    b_shift = b_mean - median_b
    g_shift = g_mean - median_g
    r_shift = r_mean - median_r
    b -= b_shift
    g -= g_shift
    r -= r_shift
    
    # 增强对比度 
    contrast_b = 2
    contrast_g = 2
    contrast_r = 2  
    b = b * contrast_b
    g = g * contrast_g
    r = r * contrast_r
    
    result = cv2.merge((b, g, r))
    
    # 截断越界元素，取整，转换类型
    result[result > 255] = 255
    result[result < 0] = 0
    result = np.round(result)
    result = result.astype(np.uint8)
    return result


# 线性拉伸图像
def linear_stretch_image(img):
    img = copy.deepcopy(img)
    h, w, _ = img.shape#获取shape的数值，height和width、通道    
    img = img.astype(np.float32)
    (b, g, r) = cv2.split(img)
    
    b_min = np.min(b)
    g_min = np.min(g)
    r_min = np.min(r)
    b_max = np.max(b)
    g_max = np.max(g)
    r_max = np.max(r)
    b -= b_min
    g -= g_min
    r -= r_min
    b *= 255/(b_max-b_min)
    g *= 255/(g_max-g_min)
    r *= 255/(r_max-r_min)
    
    result = cv2.merge((b, g, r))
    
    # 截断越界元素，取整，转换类型
    result[result > 255] = 255
    result[result < 0] = 0
    result = np.round(result)
    result = result.astype(np.uint8)
    return result



#------------------------------------------------------------------------------

for img_name in os.listdir(src_dir):
    img_path = os.path.join(src_dir, img_name)
    save_path = os.path.join(dst_dir, img_name)
    img = cv2.imread(img_path, 1)
    
    # TODO: choose one of these methods to do preprocess
    img_transfor = modify_sigmoid(img)
    #img_transfor = modify_sigmoid_gauss(img)
    #img_transfor = contrast_brightness_image(img)
    #img_transfor = contrast_brightness_image2(img)    
    #img_transfor = polynomial_transfor(img, 2)

    cv2.imwrite(save_path, img_transfor)
