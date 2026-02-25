#Plug-TPL/utility/dataset.py

import json
import torch
import utility.transcoding
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import re
from utility.parse_args import arg_parse
from torch.nn.utils.rnn import pad_sequence

args = arg_parse()


def collate_fn(batch):
    data_list, label_list = zip(*batch)

    max_context_len = max([len(data['tpl_context']) for data in data_list])
    padded_contexts = [data['tpl_context'] + [0] * (max_context_len - len(data['tpl_context'])) for data
                       in data_list]

    padded_contexts_tensor = torch.tensor(padded_contexts).to(args.device)


    function_list = [data['function_feature'] for data in data_list]
    # 填充：padding_value=0 表示不足的部分用 0 补全
    padded_function_tensor = pad_sequence(function_list, batch_first=True, padding_value=0).to(args.device)

    target_tpl_tensor = torch.tensor(
        [data['target_tpl'] for data in data_list]).to(args.device)
    context_lens_tensor = torch.tensor([data['context_len'] for data in data_list]).to(args.device)
    labels_tensor = torch.tensor(label_list).to(args.device)

    inputs = {
        'tpl_context': padded_contexts_tensor,
        'function': padded_function_tensor,
        'target_tpl': target_tpl_tensor,
        'context_len': context_lens_tensor
    }
    return inputs, labels_tensor


class TPLDataSet(Dataset):
    def __init__(self):
        super(TPLDataSet, self).__init__()
        file_path: str = args.training_data_path + args.train_dataset
        number: int = 0
        with open(file=file_path, mode='r') as fp:
            for _ in tqdm(fp, desc='load dataset', leave=False):
                number += 1
        fp = open(file=file_path, mode='r')
        lines = fp.readlines()
        self.size: int = number
        self.file = lines

    def __len__(self):
        return self.size

    def __getitem__(self, item_idx):
        line = self.file[item_idx]

        data = json.loads(line.strip('\n'))
        label = data['label']
        data = utility.transcoding.encode_data(data)

        return data, label


def get_dataloader(train: bool = True) -> DataLoader:
    dataset = TPLDataSet()
    batch_size = args.train_batch_size if train else args.test_batch_size
    loader = DataLoader(dataset=dataset, shuffle=True, batch_size=batch_size, num_workers=0, collate_fn=collate_fn)

    return loader
