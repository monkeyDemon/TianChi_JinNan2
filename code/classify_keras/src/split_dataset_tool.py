# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 11:05:26 2018

split dataset tool

split a dataset directory into training set and validation set

@author: zyb_as
"""

import os
import shutil
import random
import argparse, textwrap

# set options
parser = argparse.ArgumentParser(description = 'split dataset into trainset and validset',
        usage = textwrap.dedent('''\
        command example
        python %(prog)s --src_dir='src_directory' --train_dir='dst_directory' --valid_dir='valid_directory' '''),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--src_dir', type = str, default = None,
        help = 'the source directory of the total dataset need to be splited.')
parser.add_argument('--train_dir', type = str, default = None,
        help = 'the destination directory to save the training dataset.')
parser.add_argument('--valid_dir', type = str, default = None,
        help = 'the destination directory to save the validation dataset.')
parser.add_argument('--train_ratio', type = float, default = 0.9,
        help = 'the proportion of the training set in the total data set .')
parser.add_argument('--shuffle', type = bool, default = True,
        help = 'shuffle dataset or not.')

args = parser.parse_args()
src_dir = args.src_dir
train_dir = args.train_dir
valid_dir = args.valid_dir
train_ratio = args.train_ratio
shuffle = args.shuffle


# ------------------------------------
# check if the save directory exists
if os.path.exists(train_dir) == False:
    os.mkdir(train_dir)
if os.path.exists(valid_dir) == False:
    os.mkdir(valid_dir)

# statistical basic infomation
total_count = 0
file_list = []
for root, dirs, files in os.walk(src_dir):
    for file_name in files:
        total_count += 1
        file_path = root + '/' + file_name
        file_list.append(file_path)

# shuffle the image file list
if shuffle:
    random.shuffle(file_list)

# split the training set
split_idx = int(total_count * train_ratio)
for src_file in file_list[:split_idx]:
    file_name = src_file.split('/')[-1]
    dst_file = os.path.join(train_dir, file_name)
    shutil.copy(src_file, dst_file)

# split the validation set
for src_file in file_list[split_idx:]:
    file_name = src_file.split('/')[-1]
    dst_file = os.path.join(valid_dir, file_name)
    shutil.copy(src_file, dst_file)

print("finish")      
