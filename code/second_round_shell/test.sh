#!/bin/bash

python Detectron/tools/test_net.py --cfg config/e2e_mask_rcnn_R-50-C4_2x.yaml \
TEST.WEIGHTS ../model/mask_model/train/jinnan2_train/generalized_rcnn/model_iter14999.pkl \
NUM_GPUS 1
