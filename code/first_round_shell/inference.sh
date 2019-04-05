#!/bin/bash

rm -rf ../model/test_visualization_jpg/*
rm -rf ../model/test_visualization_jpg.zip

python ../py_file/get_jinnan2_result_final.py --cfg ../config/third_stage_retinanet_R-50-FPN_2x.yaml \
--output-dir ../model/test_visualization_jpg \
--json_dir ../../submit/json \
--image-ext jpg \
--output-ext jpg \
--classfied_confidence 0.1 \
--wts ../model/object_detection_model/model_final.pkl \
--judgement_json_path ../../data/ProcessData/cnn_judgement_result.json \
--im_or_folder ../../data/ProcessData/jinnan2_round1_test_b_20190326_sigmoid

zip -r ../model/test_visualization_jpg.zip ../model/test_visualization_jpg/
