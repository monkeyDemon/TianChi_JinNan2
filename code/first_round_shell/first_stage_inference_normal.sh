#!/bin/bash

python ../py_file/get_jinnan2_result.py --cfg ../config/first_stage_retinanet_R-50-FPN_2x.yaml \
--output-dir ../model/normal_visualization_jpg \
--image-ext jpg \
--output-ext jpg \
--json_dir ../../submit/json \
--wts ../model/object_detection_model/train/jinnan2_train/retinanet/model_final.pkl \
../../data/ProcessData/normal_sigmoid
