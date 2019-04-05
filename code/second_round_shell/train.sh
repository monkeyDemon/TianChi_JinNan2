#!/bin/bash

#rm -rf ../../submit/model/*

nohup python ../Detectron/tools/train_net.py --skip-test \
--cfg ../config/e2e_mask_rcnn_R-50-C4_2x.yaml \
OUTPUT_DIR ../model/mask_model \
>> ../model/mask_model/train.log 2>&1 &

echo "training start..."
