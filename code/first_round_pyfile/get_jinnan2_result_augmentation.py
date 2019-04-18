#!/usr/bin/env python

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

this srcipt add test data augmentation
@modified: zyb_as
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
import json
import pycocotools.mask as mask_util
import numpy as np

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
        default=False,
        action='store_true',
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
        default=0.7,
        type=float
    )
    parser.add_argument(
        '--kp-thresh',
        dest='kp_thresh',
        help='Threshold for visualizing keypoints',
        default=2.0,
        type=float
    )
    parser.add_argument(
        '--im_or_folder', help='image or folder of images', default=None
    )
    parser.add_argument(
        '--npy_save_dir', help='npy_save_path', default=None,type=str
    )
    parser.add_argument(
        '--judgement_json_path', help='', default=None,type=str
    )
    parser.add_argument(
         '--classfied_confidence', help='json', default=None,type=float
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


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
        return False,[0,0,0,0,0,0],1

def test_data_augmentation(im, test_idx, ratio):
    height, width, _ = im.shape
    x_interval = int(width*(1-ratio))
    y_interval = int(height*(1-ratio))
    aug_height = height + y_interval
    aug_width = width + x_interval
    # roi save [x1,y1,x2,y2]
    test_aug_roi = [[int(x_interval/2),int(y_interval/2),int(x_interval/2)+width-1,int(y_interval/2)+height-1], 
                    [0,0,width-1,height-1], [0,y_interval,width-1,aug_height-1], 
                    [x_interval,0,aug_width-1,height-1], [x_interval,y_interval,aug_width-1,aug_height-1]]

    img_aug = np.ones((aug_height, aug_width, 3), dtype = np.uint8) 
    img_aug *= 255
    roi_aug = test_aug_roi[test_idx]
    img_aug[roi_aug[1]:roi_aug[3]+1,roi_aug[0]:roi_aug[2]+1,:] = im
    return img_aug, roi_aug

def main(args):
    logger = logging.getLogger(__name__)

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

    if os.path.isdir(args.im_or_folder):
        im_list = glob.iglob(args.im_or_folder + '/*.' + args.image_ext)
    else:
        im_list = [args.im_or_folder]

    # load cnn judgement result
    judgement_json_path = args.judgement_json_path
    judgement_dict = {}
    with open(judgement_json_path, 'r') as f:
        judgement_dict = json.load(f)
    classfied_confidence = args.classfied_confidence

    npy_save_dir = args.npy_save_dir
    for i, im_name in enumerate(im_list):
        logger.info('\nProcessing {}...'.format(im_name))

        # get current image
        im = cv2.imread(im_name)
        height, width, _ = im.shape
        im_name_ = im_name.split('/')[-1][:-4]

        # first, get binary classify CNN judgement result
        restricted_confidence = judgement_dict[im_name.split('/')[-1]]
        #simple apply judgement
        #judgement, category_cnt, _ = judgement_function(restricted_confidence, cls_boxes,classfied_confidence)
        if restricted_confidence > classfied_confidence:
            judgement = True
        else:
            judgement = False

        # if is restricted, inference mask rcnn
        if judgement: # restricted

            # do test data augmentation
            timers = defaultdict(Timer)
            t = time.time()

            # get mask rcnn result
            cls_segms_list = []
            roi_aug_list = []
            ratio = 0.95
            with c2_utils.NamedCudaScope(0):
                for test_idx in range(5):
                    img_aug, roi_aug = test_data_augmentation(im, test_idx, ratio)
                    cls_boxes, cls_segms, cls_keyps = infer_engine.im_detect_all(
                        model, img_aug, None, timers=timers
                    )
                    cls_segms_list.append(cls_segms)
                    roi_aug_list.append(roi_aug)

            x_interval = int(width*(1-ratio))
            y_interval = int(height*(1-ratio))
            aug_height = height + y_interval
            aug_width = width + x_interval

            # merge result
            mask_class_list = np.zeros((5, height, width), dtype = np.float32)
            for test_idx in range(5):
                cls_segms = cls_segms_list[test_idx]
                roi_aug = roi_aug_list[test_idx]
                if cls_segms is not None:
                    for class_id in range(1,6):
                        if cls_segms[class_id] == []:
                            pass
                        else:
                            cur_class_mask = np.zeros((aug_height, aug_width), dtype = np.uint8)
                            for class_content in cls_segms[class_id]:
                                mask_ = mask_util.decode(class_content)
                                cur_class_mask = cur_class_mask | mask_
                            mask_class_list[class_id-1,:,:] += cur_class_mask[roi_aug[1]:roi_aug[3]+1,roi_aug[0]:roi_aug[2]+1]
                else:
                    pass
            
            for class_id in range(1,6):
                class_mask = mask_class_list[class_id-1,:,:] * 0.2
                class_mask[class_mask > 0.5] = 1
                class_mask = class_mask.astype(np.uint8)
                class_save_path = npy_save_dir + '/' + im_name_ + '_' + str(class_id) + '.npy'
                np.save(class_save_path, class_mask)
            
            logger.info('Inference total time: {:.3f}s'.format(time.time() - t))
        else:
            save_numpy = np.zeros((height, width), dtype = np.uint8)
            for class_id in range(1,6):
                class_save_path = npy_save_dir + '/' + im_name_ + '_' + str(class_id) + '.npy'
                np.save(class_save_path,save_numpy)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    setup_logging(__name__)
    args = parse_args()
    main(args)
