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
from caffe2.python import workspace

from detectron.core.config import assert_and_infer_cfg
from detectron.core.config import cfg
from detectron.core.config import merge_cfg_from_file
from detectron.utils.io import cache_url
from detectron.utils.logging import setup_logging
from detectron.utils.timer import Timer
import detectron.core.test_engine as infer_engine
import detectron.datasets.dummy_datasets as dummy_datasets
import detectron.utils.c2 as c2_utils
import detectron.utils.vis as vis_utils

c2_utils.import_detectron_ops()

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--cfg',
        dest='cfg',
        help='cfg model file (/path/to/model_config.yaml)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--wts',
        dest='weights',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        '--always-out',
        dest='out_when_no_box',
        help='output image even when no object is found',
        action='store_true'
    )
    parser.add_argument(
        '--output-ext',
        dest='output_ext',
        help='output image file format (default: pdf)',
        default='pdf',
        type=str
    )
    parser.add_argument(
        '--thresh',
        dest='thresh',
        help='Threshold for visualizing detections',
        default=0.5,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=1.0,
        type=float
    )
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
         '--FP_save_path', help='json', default=None,type=str
    )
    parser.add_argument(
         '--FN_save_path', help='json', default=None,type=str
    )
    parser.add_argument(
         '--classfied_confidence', help='json', default=None,type=float
    )
    parser.add_argument(
        '--im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()
max_area_list = [0,16960,6955,43263,19140,17290]
ave_area_list = [0,4162,2382,6815,2767,8862]
#classfied_confidence = args.classfied_confidence
def judgement_function(cnn_confidence, cls_boxes,classfied_confidence):
    if cnn_confidence > classfied_confidence:
        # cnn think is restricted
        area = [0] * 6
        category_cnt = [0] * 6
        for class_id in range(1,6):
            if len(cls_boxes[class_id]) != 0:
                for class_content in cls_boxes[class_id]:
                    if class_content[4] >= 0.5:
                        category_cnt[class_id] += 1
                        area[class_id] += (class_content[3] - class_content[1]) * (class_content[2] - class_content[0])
        for class_id in range(1,6):
            if category_cnt[class_id] != 0:
                area[class_id] = area[class_id]/category_cnt[class_id]

        cate_num = 0
        for cnt in category_cnt:
            if cnt > 0:
                cate_num += 1
        
        max_cate_num = max(category_cnt)
        '''
        if cate_num == 1:
            ave_area = max(area)/max_cate_num
            class_id_0 = 0
            for cnt in category_cnt:
                if cnt > 0:
                    break
                class_id_0 += 1
                    #max_area = area_list[cnt]
        '''            
        if cate_num == 0:
            return False,category_cnt,area
        #elif category_cnt[4] >= 3 and area[4] >= (3 * ave_area_list[4]):
            #return False,category_cnt,area
        else:
            return True, category_cnt,area
    else:
        return False, [0,0,0,0,0,0], 1
        """
        # cnn think is normal 
        area = [0] * 6
        category_cnt = [0] * 6
        for class_id in range(1,6):
            if len(cls_boxes[class_id]) != 0:
                for class_content in cls_boxes[class_id]:
                    if class_content[4] >= 0.5:
                        category_cnt[class_id] += 1
                        area[class_id] += (class_content[3] - class_content[1]) * (class_content[2] - class_content[0])
        cate_num = 0
        for cnt in category_cnt:
            if cnt > 0:
                cate_num += 1
        max_cate_num = max(category_cnt)
        if cate_num == 1:
            ave_area = max(area)/max_cate_num
            class_id_0 = 0
            for cnt in category_cnt:
                if cnt > 0:
                    break
                    #max_area = area_list[cnt]
                class_id_0 += 1

        if cate_num >= 2 and max_cate_num >= 5:
            return True, category_cnt,area
        #elif cate_num == 1 and max_cate_num >= 8 and ave_area <= (2.5 * ave_area_list[class_id_0]):
            #return True, category_cnt,area
        else:
    	    return False, category_cnt,area
        """

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
    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]
    merge_cfg_from_file(args.cfg)
    cfg.NUM_GPUS = 1
    args.weights = cache_url(args.weights, cfg.DOWNLOAD_CACHE)
    assert_and_infer_cfg(cache_urls=False)

    assert not cfg.MODEL.RPN_ONLY, \
        'RPN models are not supported'
    assert not cfg.TEST.PRECOMPUTED_PROPOSALS, \
        'Models that require precomputed proposals are not supported'

    model = infer_engine.initialize_model_from_cfg(args.weights)
    dummy_coco_dataset = dummy_datasets.get_coco_dataset()

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
    FP_Record = []
    FN_Record = []
    for i, im_path in enumerate(im_list):
        out_name = "../../data/ProcessData"
        #out_name = os.path.join(
            #args.output_dir, '{}'.format(os.path.basename(im_path) + '.' + args.output_ext)
            #args.output_dir, '{}'.format(os.path.basename(im_path))
        #)
        logger.info('Processing {} -> {}'.format(im_path, out_name))
        im = cv2.imread(im_path)
        timers = defaultdict(Timer)
        t = time.time()
        with c2_utils.NamedCudaScope(0):
            cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                model, im, None, timers=timers
            )
        logger.info('Inference time: {:.3f}s'.format(time.time() - t))
        for k, v in timers.items():
            logger.info(' | {}: {:.3f}s'.format(k, v.average_time))
        if i == 0:
            logger.info(
                ' \ Note: inference on the first image will be slower than the '
                'rest (caches and auto-tuning need to warm up)'
            )
        image_simple_name = im_path.split('/')[-1]
        restricted_confidence = judgement_dict[image_simple_name]
        judgement, category_cnt, _ = judgement_function(restricted_confidence, cls_boxes,classfied_confidence)
        if judgement: # restricted
            if image_simple_name in True_restricted_list:
                TP_count += 1
            else:
                FP_dict = {}
                FP_count += 1
                save_file_name = build_img_name(image_simple_name, category_cnt)
                #save_path = os.path.join(FP_save_path, save_file_name)
                # add
                vis_utils.vis_one_image(
                  im[:, :, ::-1],  # BGR -> RGB for visualization
                  save_file_name[:-4],
                  FP_save_path,
                  cls_boxes,
                  cls_segms,
                  cls_keyps,
                  dataset=dummy_coco_dataset,
                  box_alpha=1.0,
                  show_class=True,
                  thresh=args.thresh,
                  kp_thresh=args.kp_thresh,
                  ext=args.output_ext,
                  out_when_no_box=args.out_when_no_box )
                #shutil.copy(im_path,FP_save_path)
                FP_dict["filename"] = image_simple_name
                FP_dict["category_cnt"] = category_cnt
                FP_dict["confidence"] = float(restricted_confidence)
                FP_Record.append(FP_dict)
        else:
            if image_simple_name in True_normal_list:
                TN_count += 1
            else:
                FN_dict = {}
                FN_count += 1
                save_file_name = build_img_name(image_simple_name, category_cnt)
                vis_utils.vis_one_image(
                   im[:, :, ::-1],  # BGR -> RGB for visualization
                   save_file_name[:-4],
                   FN_save_path,
                   cls_boxes,
                   cls_segms,
                   cls_keyps,
                   dataset=dummy_coco_dataset,
                   box_alpha=1.0,
                   show_class=True,
                   thresh=args.thresh,
                   kp_thresh=args.kp_thresh,
                   ext=args.output_ext,
                   out_when_no_box=args.out_when_no_box)
                #shutil.copy(im_path,FN_save_path)
                FN_dict["filename"] = image_simple_name
                FN_dict["category_cnt"] = category_cnt
                #FN_dict["bbox"] = float(cls_boxes)
                FN_dict["confidence"] = float(restricted_confidence)
                FN_Record.append(FN_dict)
    precision = TP_count / (TP_count + FP_count) * 100
    print('精确率（被预测为违禁品的样本中实际为违禁品的比例）：%f' % precision)
    recall = TP_count / (TP_count + FN_count) * 100
    print('召回率（实际为违禁品的样本中被识别出的比例）：%f' % recall)        
    #recall = TP_count / len(True_restricted_list) * 100
    #print('HAHA召回率（实际为违禁品的样本中被识别出的比例）：%f' % recall)
    FP_Context = {}
    FP_Context["FP"] = FP_Record
    FN_Context = {}
    FN_Context["FN"] = FN_Record
    with open(FP_save_path + '/FP.json','w') as file:
        json.dump(FP_Context,file)
    with open(FN_save_path + '/FN.json','w') as file:
        json.dump(FN_Context,file)
if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
