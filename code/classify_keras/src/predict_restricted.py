#!/usr/bin/env python

# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Perform inference on a single image or all images with a certain extension
(e.g., .jpg) in a folder.
"""

from __future__ import print_function

import os
import sys
import argparse
import json
import numpy as np

from keras.preprocessing import image
from keras.models import load_model


def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--classify_wts',
        help='weights model file (/path/to/model_weights.pkl)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--test_dir',
        help='weights model file (/path/to/test_img_dir)',
        default=None,
        type=str
    )
    parser.add_argument(
        '--output_path',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='/tmp/infer_simple',
        type=str
    )
    return parser.parse_args()



class CNN_Evaluator(object):
    """ 
    CNN model evaluator
    A packaged evaluator that makes it easier to call models for prediction
    """
    def __init__(self, weight_path, input_size, confidence):
        """init evaluator
        
        # Arguments:
            weight_path: the file of the CNN weight file
            input_size: the input size of the CNN you use
        """
        # load model
        self.model = load_model(weight_path)
        # set input size
        self.input_size = input_size
        # set the confidence when model discrimination
        self.confidence = confidence
        
        # TODO: confirm category index
        self.normal_idx = 0
        self.restricted_idx = 1
    

    def predictSingleImage(self, image_path):
        """ predict the specified image
        
        # Arguments:
            image_path: string, the file path of the image to predict
        """
        # preprocess
        image = self._preprocess_image(image_path)
        # predict
        y_predict = self.model.predict(image)
        restricted_confidence = y_predict[0][self.restricted_idx]
        return restricted_confidence


    def _preprocess_image(self, image_path):
        """ preprocess the specified image
        
        The preprocess operation before training a CNN model may be different.
        So, you should modify this function by your actual demand
        
        # Arguments:
            image_path: the path of the image to do preprocess
        # Returns:
            the image after preprocess(type is ndarray)
        """
        img = image.load_img(image_path, target_size=(224,224))
        # TODO: perform the same pre-processing operations when training CNN
        x = image.img_to_array(img)/255.
        x = np.expand_dims(x, axis = 0)
        return x


def main(args):


    # use classify CNN 
    weight_file = args.classify_wts
    test_dir = args.test_dir
    output_path = args.output_path
    input_size = (224, 224, 3)
    confidence = 0.1      # the confidence when model discrimination

    # init CNN evaluator
    evaluator = CNN_Evaluator(weight_file, input_size, confidence)

    CNN_judgement_dict = {}
    for im_name in os.listdir(test_dir):
        im_name_path = os.path.join(test_dir, im_name)
        restricted_confidence = evaluator.predictSingleImage(im_name_path)
        CNN_judgement_dict[im_name] = float(restricted_confidence)

    with open(output_path, 'w') as f:
        json.dump(CNN_judgement_dict, f)


if __name__ == '__main__':
    args = parse_args()
    main(args)
