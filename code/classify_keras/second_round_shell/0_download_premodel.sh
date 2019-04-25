#!/bin/bash

cd ../../model/pre_train_model/
echo "Downloading ... "
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
echo "Finish!"
cd ../../classify_keras/second_round_shell/
