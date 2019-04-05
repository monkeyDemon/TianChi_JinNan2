#!/bin/bash

mkdir ../../data/ProcessData/train_val_jpg
mkdir ../Detectron/detectron/datasets/data/jinnan2

cp ../../data/ProcessData/restricted_sigmoid/* ../../data/ProcessData/train_val_jpg

echo "if dataset has exist, delete it"
rm -rf ../Detectron/detectron/datasets/data/jinnan2/*

echo "start split dataset..."
python ../py_file/split_train_val.py --restricted_jpg_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/First_round_data/jinnan2_round1_train_20190305/train_no_poly.json 

echo "finish"
#python ../py_file/add_area_to_val.py

