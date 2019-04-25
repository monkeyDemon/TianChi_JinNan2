#!/bin/bash

run_script="../src/split_dataset_tool.py"
mkdir ../train_dataset
mkdir ../valid_dataset
rm -rf ../train_dataset/*
rm -rf ../valid_dataset/*

echo "split restricted dataset into training set and validation set..."
#src_dir="../../../data/Second_round_data/jinnan2_round2_train_20190401/restricted"
src_dir="../../../data/ProcessData/restricted_sigmoid"
train_dir="../train_dataset/restricted"
valid_dir="../valid_dataset/restricted"
train_ratio=0.9
shuffle=True
python ${run_script} --src_dir=${src_dir} --train_dir=${train_dir} --valid_dir=${valid_dir} \
 --train_ratio=${train_ratio} --shuffle=${shuffle} 


echo "split normal dataset into training set and validation set..."
#src_dir="../../../data/Second_round_data/jinnan2_round2_train_20190401/normal"
src_dir="../../../data/ProcessData/normal_sigmoid"
train_dir="../train_dataset/normal"
valid_dir="../valid_dataset/normal"
train_ratio=0.9
shuffle=True
python ${run_script} --src_dir=${src_dir} --train_dir=${train_dir} --valid_dir=${valid_dir} \
 --train_ratio=${train_ratio} --shuffle=${shuffle}



