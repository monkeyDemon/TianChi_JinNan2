#!/bin/bash

#rm -rf ../../submit/model/*

nohup python ../Detectron/tools/train_net.py --skip-test \
--cfg ../config/first_stage_retinanet_R-50-FPN_2x.yaml \
OUTPUT_DIR ../model/object_detection_model \
>> ../model/object_detection_model/first_state_train.log 2>&1 &

echo "training start..."
