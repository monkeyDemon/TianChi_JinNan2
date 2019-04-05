#!/bin/bash

rm ../../data/ProcessData/restricted_sigmoid/*
rm ../../data/ProcessData/normal_sigmoid/*
rm ../../data/ProcessData/jinnan2_round1_test_b_20190326_sigmoid/*

echo "preprocess jinnan2_roud1_train-restricted"
python ../py_file/preprocess_sigmoid.py --src_dir ../../data/First_round_data/jinnan2_round1_train_20190305/restricted \
--dst_dir ../../data/ProcessData/restricted_sigmoid

echo "preprocess jinnan2_roud1_train-normal"
python ../py_file/preprocess_sigmoid.py --src_dir ../../data/First_round_data/jinnan2_round1_train_20190305/normal \
--dst_dir ../../data/ProcessData/normal_sigmoid

echo "preprocess jinnan2_roud1_test_b"
python ../py_file/preprocess_sigmoid.py --src_dir ../../data/First_round_data/jinnan2_round1_test_b_20190326 \
--dst_dir ../../data/ProcessData/jinnan2_round1_test_b_20190326_sigmoid
