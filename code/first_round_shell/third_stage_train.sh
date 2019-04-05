#!/bin/bash

rm -f ../model/object_detection_model/train/jinnan2_train/retinanet/model_final.pkl

nohup python ../Detectron/tools/train_net.py --skip-test \
--cfg ../config/third_stage_retinanet_R-50-FPN_2x.yaml \
OUTPUT_DIR ../model/object_detection_model \
>> ../model/object_detection_model/train.log 2>&1 &

echo "train starting..."
