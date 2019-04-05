#!/bin/bash

python ../first_round_pyfile/get_jinnan2_result.py --cfg ../config/e2e_mask_rcnn_R-50-C4_2x.yaml \
--output-dir ../model/normal_visualization_jpg \
--image-ext jpg \
--output-ext jpg \
--json_dir ../../submit/json \
--wts ../model/mask_model/train/jinnan2_train/generalized_rcnn/model_iter77999.pkl \
../../data/ProcessData/normal_sigmoid
