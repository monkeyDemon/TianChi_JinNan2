# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 09:14:30 2019

@author: ZYH
"""

import os
import shutil
import json
import argparse, textwrap

# set options
parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent('''\command example python %(prog)s --src_dir='src_directory' --dst_dir='dst_directory(此行待修改)' '''),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--input_normal_jpg_dir', type = str, default = None,
        help = 'the directory of the normal images.')      
parser.add_argument('--output_easy_normal_jpg_dir', type = str, default = None,
        help = 'out_normal_dir')
parser.add_argument('--output_diffcult_normal_jpg_dir', type = str, default = None,
        help = 'out_normal_dir')
parser.add_argument('--normal_result_json_path', type = str, default = None,
        help = 'input_json_path')
parser.add_argument('--score_threshold', type = float, default = None,
        help = 'score_threshold')
# TODO: set parameters
args = parser.parse_args()
normal_images_path = args.input_normal_jpg_dir
jpg_save_dir = args.output_diffcult_normal_jpg_dir
easy_jpg_save_dir = args.output_easy_normal_jpg_dir
if not os.path.exists(jpg_save_dir):
    os.makedirs(jpg_save_dir)
if not os.path.exists(easy_jpg_save_dir):
    os.makedirs(easy_jpg_save_dir)
json_path = args.normal_result_json_path
score_threshold = args.score_threshold

with open(json_path,'r') as load_f:
    normal_dict = json.load(load_f)
normal_result = normal_dict["results"]

image_num = len(normal_result)
easy_normal_num = 0
diffcult_normal_num = 0
#null_image = str()
for im in range(image_num):
    normal_image = normal_result[im]
    imagename = normal_images_path + '/' + normal_image["filename"]
    print(imagename)
    if normal_image["rects"] == []:
        #print(normal_image["filename"])
        #null_num = null_num + 1
        #imagename = normal_images_path + '/' + normal_image["filename"]
        shutil.copy(imagename, easy_jpg_save_dir) 
        easy_normal_num = easy_normal_num + 1
    else:
        rect_context = normal_image["rects"]
        rect_len = len(rect_context)
        max_scores = 0
        for re in range(rect_len):
            max_scores = max(max_scores,rect_context[re]["confidence"])
        if max_scores >= score_threshold:
            #imagename = normal_images_path + '/' + normal_image["filename"]
            shutil.copy(imagename, jpg_save_dir)
            diffcult_normal_num = diffcult_normal_num + 1
            #print("Add normal num:",add_normal_num)
        else:
            #imagename = normal_images_path + '/' + normal_image["filename"]
            shutil.copy(imagename, easy_jpg_save_dir)
            easy_normal_num = easy_normal_num + 1

print("easy normal num:",easy_normal_num)
print("diffcult normal num:",diffcult_normal_num)


