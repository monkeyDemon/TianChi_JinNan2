#!/bin/bash

rm -rf ../../data/ProcessData/train_val_jpg/*

python ../py_file/data_augmentation_position2.py \
--dense_restricted_dir ../../data/ProcessData/restricted_sigmoid \
--img_save_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/First_round_data/jinnan2_round1_train_20190305/train_no_poly.json \
--output_json_path ../../data/ProcessData/train_val_position.json \
--generate_num 20000

#--img_save_dir ../../data/ProcessData/train_val_jpg \
cp ../../data/ProcessData/restricted_sigmoid/* ../../data/ProcessData/train_val_jpg

echo "finish"
