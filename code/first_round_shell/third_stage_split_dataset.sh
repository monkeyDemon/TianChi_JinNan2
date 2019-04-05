#!/bin/bash

rm -rf ../Detectron/detectron/datasets/data/jinnna2/*

python ../py_file/split_train_val.py --restricted_jpg_dir ../../data/ProcessData/train_val_jpg \
--input_json_path ../../data/ProcessData/train_val_diffcult.json 

#python ../py_file/add_area_to_val.py
echo "finish"
