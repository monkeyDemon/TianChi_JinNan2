import json
import argparse, textwrap

# set options
parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent('--'),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--input_json_path', type = str, default = None,
        help = 'input_json_path')
parser.add_argument('--output_json_path', type = str, default = None,
        help = 'output_json_path')

args = parser.parse_args()
json_save_path = args.output_json_path
init_json_path = args.input_json_path

with open(init_json_path,'r') as file:
    Jinnan2_data = json.load(file)
    
Jinnan2_data["annotations"] = []
Jinnan2_data["images"] = []

with open(json_save_path,'w') as file:
    json.dump(Jinnan2_data,file)
