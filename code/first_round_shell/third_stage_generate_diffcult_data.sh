#!/bin/bash

python ../py_file/data_augmentation_diffcult.py --normal_dir ../../data/ProcessData/diffcult_normal_background \
--restricted_dir ../../data/ProcessData/restricted_objects \
--img_save_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/ProcessData/train_val_position.json \
--output_json_path ../../data/ProcessData/train_val_diffcult.json \
--generate_num  700 \
--max_restricted_num 1 \
--is_diffcult True

#cp ../../data/First_round_data/jinnan2_round1_train_20190305/restricted/* ../../data/ProcessData/train_val_jpg

echo "finish"
