#!/bin/bash

#cp ../../data/ProcessData/restricted_sigmoid/* ../../data/ProcessData/train_val_jpg
#cp ../../data/First_round_data/jinnan2_round1_train_20190305/train_no_poly.json ../../data/ProcessData/
rm -rf ../Detectron/detectron/datasets/data/jinnan2/*

python ../py_file/split_train_val.py --restricted_jpg_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/ProcessData/train_val.json
echo "finsh"
#python ../py_file/add_area_to_val.py

