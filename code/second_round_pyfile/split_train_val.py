# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 17:47:01 2019

@author: ZYH
"""

import json
import os
import random
import argparse, textwrap
import shutil
from PIL import Image 

# set options
parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent('''\command example python %(prog)s --src_dir='src_directory' --dst_dir='dst_directory(此行待修改)' '''),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--normal_jpg_dir', type = str, default = '',
        help = 'the directory of the normal images.')      
parser.add_argument('--restricted_jpg_dir', type = str, default = None,
        help = 'the source directory of the restricted images.')
parser.add_argument('--train_jpg_save_dir', type = str, default = "../Detectron/detectron/datasets/data/jinnan2/jinnan2_train",
        help = 'train_jpg_save_dir.')
parser.add_argument('--val_jpg_save_dir', type = str, default = "../Detectron/detectron/datasets/data/jinnan2/jinnan2_val",
        help = 'val_jpg_save_dir')
parser.add_argument('--input_json_path', type = str, default = None,
        help = 'input_json_path')
parser.add_argument('--output_json_dir', type = str, default = "../Detectron/detectron/datasets/data/jinnan2/annotations",
        help = 'output_json_dir')
parser.add_argument('--is_add_normal', type = bool, default = False,
        help = 'is_add_normal')
parser.add_argument('--val_rate', type = float, default = 0.1,
        help = 'is')
# TODO: set parameters
args = parser.parse_args()

normal_jpg_dir = args.normal_jpg_dir
train_jpg_save_dir = args.train_jpg_save_dir
val_jpg_save_dir = args.val_jpg_save_dir
restricted_jpg_dir = args.restricted_jpg_dir
json_dir = args.output_json_dir
input_json_path = args.input_json_path
is_add_normal = args.is_add_normal

if not os.path.exists(train_jpg_save_dir):
    os.makedirs(train_jpg_save_dir)
    
if not os.path.exists(val_jpg_save_dir):
    os.makedirs(val_jpg_save_dir)

if not os.path.exists(json_dir):
    os.makedirs(json_dir)

with open(input_json_path,'r') as file:
    Jinnan2_data = json.load(file)

    
#print(Jinnan2_data.keys())
#将普通图片写进images
image_id = []
for im in Jinnan2_data["images"]:
    image_id.append(im["id"])
#print(max(image_id))
if is_add_normal: 
    rest_max_id = max(image_id)
    #sample_rest = Jinnan2_data["images"][0]
    #print(sample_rest)
    normal_add_id = 1
    normal_images_list = os.listdir(normal_jpg_dir)
    Jinnan2_images = Jinnan2_data["images"]
    for nor_image in normal_images_list:
        samplr_nor = {}
        image = Image.open(normal_jpg_dir + '/' + nor_image)
        im_width, im_height = image.size
        #print(nor_image)
        samplr_nor['coco_url'] = ''
        samplr_nor['data_captured'] = ''
        samplr_nor['flickr_url'] = ''
        samplr_nor['license'] = 1
        samplr_nor['file_name'] = nor_image
        samplr_nor['id'] = rest_max_id + normal_add_id
        samplr_nor['height'] = im_height
        samplr_nor['width'] = im_width
        Jinnan2_images.append(samplr_nor)
        image_id.append(samplr_nor['id'])
        normal_image_path = normal_jpg_dir + '/' + nor_image
        restricted_image_path = restricted_jpg_dir + '/' + nor_image
        shutil.copy(normal_image_path, restricted_jpg_dir)
        normal_add_id = normal_add_id + 1
    Jinnan2_data["images"] = Jinnan2_images
'''
for ima in Jinnan2_data["images"]:
    if ima['file_name'] == '190102_161213_00148929.jpg':
        print(ima['id'])
'''

#rain_rate = 0.95
val_rate = args.val_rate
#print(val_rate)
train_rate = 1 - val_rate
#print(train_rate)
train_num = int(len(image_id) * train_rate)
#print(train_num)
val_num = int(len(image_id) - train_num)
#print(val_num)

train_id = random.sample(image_id,train_num)
#print(train_id)

train_josn = {}
val_json = {}
#print(Jinnan2_data['categories'])
train_josn['categories'] = Jinnan2_data['categories']
val_json['categories'] = Jinnan2_data['categories']

train_josn['info'] = Jinnan2_data['info']
val_json['info'] = Jinnan2_data['info']

train_josn['licenses'] = Jinnan2_data['licenses']
val_json['licenses'] = Jinnan2_data['licenses']
print(Jinnan2_data['licenses'])

train_annotations = []
val_annotations = []
ann_len = len(Jinnan2_data["annotations"])
ann_id = 1
for anno in Jinnan2_data["annotations"]:
    if anno['image_id'] in train_id:
        train_annotations.append(anno)
    else:
        val_annotations.append(anno)
    if ann_id % 100 == 0:
        print("{} annotations has done!".format(ann_id))
    ann_id = ann_id + 1
train_josn['annotations'] = train_annotations
val_json['annotations'] = val_annotations

train_images = []
val_images = []
im_id = 1
for ima in Jinnan2_data["images"]:
    if ima['id'] in train_id:
        train_images.append(ima)
        train_image_path = restricted_jpg_dir + '/' + ima["file_name"]
        shutil.copy(train_image_path, train_jpg_save_dir)
        #print("copy {} to train.".format(ima))
    else:
        val_images.append(ima)
        val_image_path = restricted_jpg_dir + '/' + ima["file_name"]
        shutil.copy(val_image_path, val_jpg_save_dir)
        #print("copy {} to val.".format(ima))
    if im_id % 100 == 0:
        print("copy {} images to jinnan2_data".format(im_id))
    im_id = im_id + 1
train_josn['images'] = train_images
val_json['images'] = val_images
        
train_json_path = json_dir + "/instances_train2014.json" 
with open(train_json_path,"w") as train_f:
    json.dump(train_josn,train_f)
val_json_path = json_dir + "/instances_val2014.json"
with open(val_json_path,"w") as val_f:
    json.dump(val_json,val_f)

