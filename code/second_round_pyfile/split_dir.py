"""
Created on Wed Dec 26 10:13:17 2018

split the aim directory into specified number subdirectories

@author: zyb_as
"""

import os
import shutil
import argparse, textwrap

# set options
parser = argparse.ArgumentParser(description = 'manual to this script',
                usage = textwrap.dedent('''\
                command example: 'todo:' '''),
                formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--src_dir', type = str, default = None,
                help = 'the source directory')
parser.add_argument('--split_num', type = int, default = 2,
                help = 'the split sub directory number')


args = parser.parse_args()
src_dir = args.src_dir
split_num = args.split_num


total_num = 0
for file_name in os.listdir(src_dir):
    total_num += 1

one_share_num = int(total_num/split_num)

share_idx = 0
share_dir = src_dir + "_" + str(share_idx)
if os.path.exists(share_dir) == False:
    os.mkdir(share_dir)
else:
    os.remove(share_dir)
    os.mkdir(share_dir)

next_share_num = one_share_num
cnt = 0
for file_name in os.listdir(src_dir):
    if cnt > next_share_num:
        next_share_num += one_share_num 
        share_idx += 1
        share_dir = src_dir + "_" + str(share_idx)
        if os.path.exists(share_dir) == False:
            os.mkdir(share_dir)
        else:
            os.remove(share_dir)
            os.mkdir(share_dir)
    src_file = os.path.join(src_dir, file_name)
    dst_file = os.path.join(share_dir, file_name)
    shutil.copy(src_file, dst_file)
    cnt += 1
