#!/bin/bash

python ../py_file/draw_learning_curve.py \
--train_log ../../submit/model/train.log \
--fig_save_path ../../submit/model/learning_curve.jpg

#python ../py_file/draw_learning_curve.py \
#--train_log ../../submit/save_file/model_031909/train.log \
#--fig_save_path ../../submit/save_file/model_031909/learning_curve.jpg
