#!/bin/bash

cd ../model/pre_train_model/
echo "Downloading ... "
wget https://dl.fbaipublicfiles.com/detectron/35861858/12_2017_baselines/e2e_mask_rcnn_R-101-FPN_2x.yaml.02_32_51.SgT4y1cO/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
echo "Finish!"
cd ../../second_round_shell/
