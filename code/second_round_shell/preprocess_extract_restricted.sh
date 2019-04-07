#!/bin/bash

python ../second_round_pyfile/extract_restricted_object.py \
--json_path ../../data/Second_round_data/jinnan2_round2_train_20190401/train_restriction_fix.json \
--restricted_img_dir ../../data/ProcessData/restricted_sigmoid \
--restricted_object_info_save_dir ../../data/ProcessData/restricted_info

echo "finish"
