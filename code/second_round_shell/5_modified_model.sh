#!/bin/bash

python ../second_round_pyfile/modified_model.py --pre_model_path ../model/pre_train_model/model-101-FPN.pkl \
--modified_model_path ../model/pre_train_model/jinnan2_mask-101-FPN.pkl \
--remove_list cls_score_w cls_score_b bbox_pred_w bbox_pred_b mask_fcn_logits_w mask_fcn_logits_b

echo "finish"
