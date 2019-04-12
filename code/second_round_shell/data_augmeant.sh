#!/bin/bash

rm -rf ../../data/ProcessData/train_val_jpg/*

python ../second_round_pyfile/data_augmentation_position.py --restricted_dir ../../data/ProcessData/restricted_sigmoid \
--img_save_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/Second_round_data/jinnan2_round2_train_20190401/train_restriction_fix.json \
--output_json_path ../../data/ProcessData/train_position.json \
--generate_num 40000

cp ../../data/ProcessData/restricted_sigmoid/* ../../data/ProcessData/train_val_jpg

echo "finish!"
