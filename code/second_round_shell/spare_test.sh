#!/bin/bash

nohup python ../Detectron/tools/test_net.py --cfg ../config/e2e_mask_rcnn_R-50-C4_2x.yaml \
TEST.WEIGHTS ../model/mask_model/model_best.pkl \
NUM_GPUS 1 \
>> ./test.log 2>&1 &
