#!/bin/bash

#python ../py_file/judgement_restricted.py \
#--classify_wts ../classify_keras/models/resnet50_classify2_v0_98864.h5 \
#--test_dir ../../data/First_round_data/jinnan2_round1_test_a_20190306 \
#--output_path ../../data/ProcessData/cnn_judgement_result.json
mkdir ../../data/ProcessData/RPTest
mkdir ../../data/ProcessData/RPTest/FP_save
mkdir ../../data/ProcessData/RPTest/FN_save

rm -rf ../../data/ProcessData/RPTest/FP_save/*
rm -rf ../../data/ProcessData/RPTest/FN_save/*

python ../second_round_pyfile/recall_precision.py --cfg ../config/e2e_mask_rcnn_R-50-C4_2x.yaml \
--output-dir ../model/test_visualization_jpg \
--image-ext jpg \
--output-ext jpg \
--classfied_confidence 0.1 \
--wts ../../submit/model/mask_model/model_best.pkl \
--judgement_json_path ../../data/ProcessData/cnn_judgement_result.json \
--restricted_test_path ../../data/ProcessData/restricted_test \
--normal_test_path ../../data/ProcessData/normal_test \
--FP_save_path ../../data/ProcessData/RPTest/FP_save \
--FN_save_path ../../data/ProcessData/RPTest/FN_save \
--im_or_folder ../../data/ProcessData/jinnan2_round2_test_a_20190401_sigmoid

rm -rf ../../data/ProcessData/RPTest.zip
zip -r ../../data/ProcessData/RPTest.zip ../../data/ProcessData/RPTest
