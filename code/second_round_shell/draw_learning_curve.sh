#!/bin/bash

python ../second_round_pyfile/draw_learning_curve.py \
--train_log ../model/mask_model/train.log \
--fig_save_path ../model/mask_model/learning_curve.jpg

#python ../py_file/draw_learning_curve.py \
#--train_log ../../submit/save_file/model_031909/train.log \
#--fig_save_path ../../submit/save_file/model_031909/learning_curve.jpg
