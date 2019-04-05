#!/bin/bash

train_script="./src/resNet50_transfer_learning.py"
#train_script="./src/denseNet_transfer_learning.py"

weight_load_path="../model/pre_train_model/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
weight_save_path="../model/cnn_model/resnet50_classify2_sigmoid_v1.h5"
train_set_path="./train_dataset"
valid_set_path="./valid_dataset"
record_save_dir="./record/"
category_num=2
batch_size=32

echo "run ${train_script}"


CUDA_VISIBLE_DEVICES=0 nohup python ${train_script} --weight_load_path=${weight_load_path} \
 --weight_save_path=${weight_save_path} \
 --train_set_path=${train_set_path} --valid_set_path=${valid_set_path} \
 --record_save_dir=${record_save_dir} --category_num=${category_num} --batch_size=${batch_size} \
 > ./record/train_log_resnet_v1.out 2>&1 &

echo "training start..."
