#!/bin/bash

python ./src/predict_restricted.py \
--classify_wts ../model/cnn_model/resnet50_classify2_sigmoid_final.h5 \
--test_dir ../../data/ProcessData/jinnan2_round1_test_b_20190326_sigmoid \
--output_path ../../data/ProcessData/cnn_judgement_result.json
