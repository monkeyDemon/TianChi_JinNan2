import argparse, textwrap
import json


# set options
parser = argparse.ArgumentParser(description = 'verify the format of the specify image dataset',
        usage = textwrap.dedent('r'),
        formatter_class = argparse.RawTextHelpFormatter)
parser.add_argument('--input_json_path', type = str, default = None,
        help = 'input_json_path')
parser.add_argument('--output_json_path', type = str, default = None,
        help = 'output_json_path')
#original_json_path = "./jinnan2_round2_train_20190401/train_restriction.json"
#json_save_path = "train_restriction_fix.json"


args = parser.parse_args()
original_json_path = args.input_json_path
json_save_path = args.output_json_path

error_items = [
    {
      "id": 7706,
      "category_id": 2
    },
    {
      "id": 7707,
      "category_id": 2
    },
    {
      "id": 12237,
      "category_id": 5
    },
    {
      "id": 12238,
      "category_id": 5
    },
    {
      "id": 7154,
      "category_id": 3
    },
    {
      "id": 7155,
      "category_id": 3
    },
    {
      "id": 7156,
      "category_id": 5
    },
    {
      "id": 7157,
      "category_id": 5
    },
    {
      "id": 1786,
      "category_id": 2
    },
    {
      "id": 1787,
      "category_id": 2
    },
    {
      "id": 1788,
      "category_id": 2
    },
    {
      "id": 9218,
      "category_id": 4
    },
    {
      "id": 9942,
      "category_id": 4
    },
    {
      "id": 9943,
      "category_id": 4
    },
    {
      "id": 9944,
      "category_id": 4
    },
    {
      "id": 11900,
      "category_id": 2
    },
    {
      "id": 11901,
      "category_id": 2
    },
    {
      "id": 11902,
      "category_id": 2
    }
]

with open(original_json_path,'r') as original_json:
    load_dict = json.load(original_json)
annotations = load_dict["annotations"]

for i in range(len(error_items)):
    id = error_items[i]["id"]
    annotations[id-1]["category_id"] = error_items[i]["category_id"]


load_dict["annotations"] = annotations
result = load_dict

with open(json_save_path,"w") as dump_f:
    json.dump(result ,dump_f)
