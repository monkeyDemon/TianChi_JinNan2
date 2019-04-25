#!/bin/bash
mkdir ../../data/ProcessData/train_val_jpg
rm -rf ../../data/ProcessData/train_val_jpg/*

python ../second_round_pyfile/data_augmentation_position_bak.py \
--restricted_dir ../../data/ProcessData/restricted_sigmoid \
--img_save_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/ProcessData/train_restriction_fix.json \
--output_json_path ../../data/ProcessData/train_position.json \
--generate_num 40000

cp ../../data/ProcessData/restricted_sigmoid/* ../../data/ProcessData/train_val_jpg

echo "finish!"
