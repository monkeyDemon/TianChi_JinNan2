#!/bin/bash

python ../py_file/select_normal.py --input_normal_jpg_dir ../../data/ProcessData/normal_sigmoid \
--output_easy_normal_jpg_dir ../../data/ProcessData/easy_normal_background \
--output_diffcult_normal_jpg_dir ../../data/ProcessData/diffcult_normal_background \
--normal_result_json_path ../../submit/json/"$1" \
--score_threshold 0.6

echo "finish"

