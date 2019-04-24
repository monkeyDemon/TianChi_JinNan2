#!/bin/bash

#python ../second_round_pyfile/preprocess_sigmoid.py --src_dir ../../data/Second_round_data/jinnan2_round2_test_a_20190401 \
#--dst_dir ../../data/ProcessData/jinnan2_round2_test_a_20190401_sigmoid


#python ../second_round_pyfile/preprocess_sigmoid.py --src_dir ../../data/Second_round_data/jinnan2_round2_train_20190401/restricted \
#--dst_dir ../../data/ProcessData/restricted_sigmoid

#python ../second_round_pyfile/preprocess_sigmoid.py --src_dir ../../data/Second_round_data/jinnan2_round2_train_20190401/normal \
#--dst_dir ../../data/ProcessData/normal_sigmoid

python ../second_round_pyfile/preprocess_sigmoid.py --src_dir ../../data/Second_round_data/jinnan2_round2_test_b_20190424 \
--dst_dir ../../data/ProcessData/jinnan2_round2_test_b_20190424_sigmoid
