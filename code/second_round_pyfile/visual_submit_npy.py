# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 20:01:19 2019

可视化掩码

@author: zyb_as
"""
import numpy as np
import os
import cv2



if __name__=="__main__":
    '''
    Example code for visual npy
    '''

    prediction_dir="../model/npy"
    visual_dir="../model/npy"

    submits_dict=dict()
    file_name = os.listdir(prediction_dir)
    for npy in file_name:
            cls_visual_name="{}.jpg".format(npy[:-4])
            save_path = os.path.join(visual_dir,cls_visual_name)
            pred_p = prediction_dir + "/" + npy
            pred=np.load(pred_p)
            pred *= 255
            cv2.imwrite(save_path,pred)


