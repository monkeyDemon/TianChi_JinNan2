#!/bin/bash

python ../src/predict_restricted.py \
--classify_wts ../../model/cnn_model/resnet50_classify2_sigmoid_v2_sigmoid.h5 \
--test_dir ../../../data/ProcessData/jinnan2_round2_test_a_20190401_sigmoid \
--output_path ../../../data/ProcessData/cnn_judgement_result.json
