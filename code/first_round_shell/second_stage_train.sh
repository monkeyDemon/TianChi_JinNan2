#!/bin/bash

#mkdir ../../submit/init_model
#cp -r ../../submit/model ../../submit/init_model/
rm -rf ../model/object_detection_model/train/*

nohup python ../Detectron/tools/train_net.py --skip-test \
--cfg ../config/second_stage_retinanet_R-50-FPN_2x.yaml \
OUTPUT_DIR ../model/object_detection_model \
>> ../model/object_detection_model/train.log 2>&1 &

echo "train strating..."
