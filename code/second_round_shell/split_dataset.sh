#!/bin/bash

mkdir ../Detectron/detectron/datasets/data/jinnan2

#cp ../../data/Second_round_data/jinnan2_round2_train_20190401/restricted/* ../../data/ProcessData/train_val_jpg
#cp ../../data/ProcessData/restricted_sigmoid/* ../../data/ProcessData/train_val_jpg

echo "if dataset has exist, delete it"
rm -rf ../Detectron/detectron/datasets/data/jinnan2/*

echo "start split dataset..."
python ../first_round_pyfile/split_train_val.py --restricted_jpg_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/ProcessData/train_diffcult.json
#--input_json_path ../../data/Second_round_data/jinnan2_round2_train_20190401/train_restriction.json

echo "finish"
#python ../py_file/add_area_to_val.py

