#!/bin/bash

rm -rf ../../data/ProcessData/train_val_jpg/*

python ../py_file/data_augmentation_diffcult.py --normal_dir ../../data/ProcessData/easy_normal_background \
--restricted_dir ../../data/ProcessData/restricted_objects \
--img_save_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/ProcessData/train_no_poly_null.json \
--output_json_path ../../data/ProcessData/train_val.json \
--generate_num  2000 \
--max_restricted_num 7

echo "finish"
#cp ../../data/First_round_data/jinnan2_round1_train_20190305/restricted/* ../../data/ProcessData/train_val_jpg
