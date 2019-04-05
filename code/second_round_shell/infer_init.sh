#!/bin/bash

python ../second_round_pyfile/get_jinnan2_result.py --cfg ../config/e2e_mask_rcnn_R-50-C4_2x.yaml \
--output-dir ../model/test_visualization_jpg \
--image-ext jpg \
--output-ext jpg \
--wts ../model/mask_model/train/jinnan2_train/generalized_rcnn/model_final.pkl \
--im_or_folder ../../data/Second_round_data/jinnan2_round2_test_a_20190401 \
--npy_save_dir ../model/npy
