#!/bin/bash

python ../py_file/modified_model.py --pre_model_path ../model/pre_train_model/model_final.pkl \
--modified_model_path ../model/pre_train_model/pre_jinnan2_mask_model.pkl \
--remove_list cls_score_w cls_score_b bbox_pred_w bbox_pred_b mask_fcn_logits_w mask_fcn_logits_b

echo "finish"
#--remove_keys1 retnet_cls_pred_fpn3_w \
#--remove_keys2 retnet_cls_pred_fpn3_b \
#--remove_keys3 retnet_bbox_pred_fpn3_w \
#--remove_keys4 retnet_bbox_pred_fpn3_b
