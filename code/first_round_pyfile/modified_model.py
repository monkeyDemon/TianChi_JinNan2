import cPickle as pkl 
import argparse, textwrap

parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
                                 usage = textwrap.dedent('a'),formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--pre_model_path', type = str, default = 'pre_train_model/model_final.pkl',
        help = 'pre_model_path')
parser.add_argument('--modified_model_path', type = str, default = 'pre_train_model/pre_jinnan2_model.pkl',
        help = 'modified_model_path')
parser.add_argument('--remove_list', type=str, nargs='+', action='append', help='remove list')

args = parser.parse_args()
pre_model_path = args.pre_model_path
modified_model_path = args.modified_model_path

remove_list = args.remove_list[0]
#print(remove_list)
with open(pre_model_path, 'rb') as f:
    wts = pkl.load(f)
#print(wts['blobs']['res4_3_branch2c_b'.encode('utf-8')])

for blob in wts['blobs'].keys():
    #print(blob)
    for remove_ob in remove_list:
        if blob.startswith(remove_ob):
            print(blob)
            #print(wts['blobs'][blob])
            del wts['blobs'][blob.encode('utf-8')]

with open(modified_model_path, 'wb') as f:
    pkl.dump(wts, f)

