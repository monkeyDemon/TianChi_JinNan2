#!/bin/bash

rm -rf ../model/test_visualization_jpg/*
rm -rf ../model/npy/*
rm -rf ../../data/ProcessData/jinnan2_round2_test_a_20190401_sigmoid_0
rm -rf ../../data/ProcessData/jinnan2_round2_test_a_20190401_sigmoid_1

test_dir='../../data/ProcessData/jinnan2_round2_test_a_20190401_sigmoid'
split_num=2
echo "split test directory"
python ../second_round_pyfile/split_dir.py --src_dir=$test_dir --split_num=$split_num
echo "finish"

echo "start process 0 with gpu 0"
py_file='../second_round_pyfile/get_jinnan2_result_augmentation_mutil.py'
cfg_file='../config/e2e_mask_rcnn_R-101-FPN_2x.yaml'
model_weights='../model/mask_model/model_final.pkl'

test_img_dir='../../data/ProcessData/jinnan2_round2_test_a_20190401_sigmoid_0'
gpu_id=0
record_path='../../data/ProcessData/inference_record_0'
nohup python $py_file  --cfg $cfg_file \
--output-dir ../model/test_visualization_jpg \
--image-ext jpg \
--output-ext jpg \
--classfied_confidence 0.1 \
--wts $model_weights \
--judgement_json_path ../../data/ProcessData/cnn_judgement_result.json \
--im_or_folder $test_img_dir \
--npy_save_dir ../model/npy \
--gpu_id $gpu_id \
> $record_path 2>&1 &

echo "start process 1 with gpu 1"
test_img_dir='../../data/ProcessData/jinnan2_round2_test_a_20190401_sigmoid_1'
gpu_id=1
record_path='../../data/ProcessData/inference_record_1'
nohup python $py_file  --cfg $cfg_file \
--output-dir ../model/test_visualization_jpg \
--image-ext jpg \
--output-ext jpg \
--classfied_confidence 0.1 \
--wts $model_weights \
--judgement_json_path ../../data/ProcessData/cnn_judgement_result.json \
--im_or_folder $test_img_dir \
--npy_save_dir ../model/npy \
--gpu_id $gpu_id \
> $record_path 2>&1 &
