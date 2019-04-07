#!/bin/bash

python ../second_round_pyfile/paste_restricted_object2normal.py \
--normal_dir ../../data/ProcessData/diffcult_normal_background \
--restricted_info_path ../../data/ProcessData/restricted_info/extract_restricted_info_dict.npy \
--img_save_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/ProcessData/train_position.json \
--output_json_path ../../data/ProcessData/train_diffcult.json \
--generate_num 600 \
--max_restricted_num 1

#--restricted_dir ../../data/ProcessData/restricted_info\

#cp ../../data/First_round_data/jinnan2_round1_train_20190305/restricted/* ../../data/ProcessData/train_val_jpg

echo "finish"
