#Plug-TPL/utility/transcoding.py

import numpy as np
from typing import Dict, Union, List
from utility.parse_args import arg_parse
import json
import torch
args = arg_parse()
app_functions = torch.load(args.function_feature_path)


def read_json_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    datas = []
    for line in lines:
        data = json.loads(line.strip())
        datas.append(data)
    f.close()
    return datas


def encode_data(data: Dict) -> Dict[str, Union[List, np.ndarray, int]]:
    tpl_context = data['tpl_list']
    app_function = app_functions[data['app_id']]
    target_tpl = data['target_tpl']
    data = {
        'tpl_context': tpl_context,
        'target_tpl': target_tpl,
        'function_feature': app_function,
        'context_len': len(data['tpl_list']),
        'function_len': app_function.size(0)
    }
    return data

