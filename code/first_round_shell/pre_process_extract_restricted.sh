#!/bin/bash

python ../py_file/extract_restricted_object.py --json_path ../../data/First_round_data/jinnan2_round1_train_20190305/train_no_poly.json \
--restricted_img_dir ../../data/ProcessData/restricted_sigmoid \
--restricted_object_save_dir ../../data/ProcessData/restricted_objects

echo "finish"
