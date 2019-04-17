#!/usr/bin/env python
# coding: utf-8
# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import sys
import time
import datetime
import json
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    
    parser.add_argument(
        '--judgement_json_path', help='', default=None,type=str
    )
    parser.add_argument(
         '--restricted_test_path', help='json', default=None,type=str
    )
    parser.add_argument(
         '--normal_test_path', help='json', default=None,type=str
    )
    parser.add_argument(
         '--classfied_confidence', help='json', default=None,type=float
    )
    parser.add_argument(
         '--FP_save_path', help='json', default=None,type=str
    )
    parser.add_argument(
         '--FN_save_path', help='json', default=None,type=str
    )

    parser.add_argument(
        '--im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def build_img_name(image_simple_name, category_cnt):
    file_name = image_simple_name[:-4]    
    for i in range(1,6):
        file_name += '_' + str(category_cnt[i])
    file_name += '.jpg'
    return file_name


def main(args):
    logger = logging.getLogger(__name__)
    classfied_confidence = args.classfied_confidence
    # get test image list
    #if os.path.isdir(args.im_or_folder):
    #    im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    #else:
    #    im_list = [args.im_or_folder]
    im_or_folder = args.im_or_folder
    im_list = os.listdir(im_or_folder)
    # load cnn judgement result
    judgement_json_path = args.judgement_json_path
    judgement_dict = {}
    with open(judgement_json_path, 'r') as f:
        judgement_dict = json.load(f)

    # do object detection
    TP_count = 0
    FP_count = 0
    FN_count = 0
    TN_count = 0
    restricted_test_path  = args.restricted_test_path
    True_restricted_list = os.listdir(restricted_test_path)
    normal_test_path = args.normal_test_path
    True_normal_list = os.listdir(normal_test_path)
    FP_save_path = args.FP_save_path
    FN_save_path = args.FN_save_path
    for im_path in im_list:
        out_name = "../../data/ProcessData"
        logger.info('Processing {} -> {}'.format(im_path, out_name))
        im = cv2.imread(im_path)
        image_simple_name = im_path.split('/')[-1]
        restricted_confidence = judgement_dict[image_simple_name]

        if restricted_confidence > classfied_confidence:
            judgement = True
        else:
            judgement = False
        if judgement: # restricted
            if image_simple_name in True_restricted_list:
                TP_count += 1
            else:
                FP_count += 1
                print("FP:",image_simple_name)
                #save_file_name = build_img_name(image_simple_name, category_cnt)
                #save_path = os.path.join(FP_save_path, save_file_name)
                shutil.copy(os.path.join(im_or_folder,im_path),FP_save_path)
        else:
            if image_simple_name in True_normal_list:
                TN_count += 1
            else:
                FN_count += 1
                print("FN:",image_simple_name)
                #save_file_name = build_img_name(image_simple_name, category_cnt)
                shutil.copy(os.path.join(im_or_folder,im_path),FN_save_path)
    precision = TP_count / (TP_count + FP_count) * 100
    print('精确率（被预测为违禁品的样本中实际为违禁品的比例）：%f' % precision)
    recall = TP_count / (TP_count + FN_count) * 100
    print('召回率（实际为违禁品的样本中被识别出的比例）：%f' % recall)        
    #recall = TP_count / len(True_restricted_list) * 100
    #print('HAHA召回率（实际为违禁品的样本中被识别出的比例）：%f' % recall)
if __name__ == '__main__':
    args = parse_args()
    main(args)
