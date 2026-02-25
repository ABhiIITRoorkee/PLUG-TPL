#APlug-TPL/train_model.py
import os
import re
import torch
import model_Plug_TPL.py
import utility.dataset
import numpy as np
from tqdm import tqdm
from utility.parse_args import arg_parse
args = arg_parse()

model = model_Plug_TPL.py.AttenTPL()
model = model.to(args.device)

criterion = torch.nn.BCELoss()
params = model.named_parameters()
adagrad_params = []
for name, param in params:
    adagrad_params.append(param)

optimizer = torch.optim.Adagrad(adagrad_params, lr=args.lr, weight_decay=args.weight_decay)
# optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)

fold: str = re.findall('[0-9]', args.train_dataset)[0]
path: str = 'model_Plug_TPL.py'


def ensure_dir(ensure_path: str) -> None:
    if not os.path.exists(ensure_path):
        os.makedirs(ensure_path)


def train() -> None:

    model.train()
    for i in range(args.epoch):
        loss_list = []
        data_loader = utility.dataset.get_dataloader()
        print('load data completed.')
        bar = tqdm(data_loader, total=len(data_loader), ascii=True, desc='train')
        for idx, (datas, labels) in enumerate(bar):
            outputs = model(datas)
            # outputs = model(datas)
            labels = labels.to(args.device).float()
            outputs = outputs.view(-1).float()
            loss_deep = criterion(outputs, labels)
            optimizer.zero_grad()
            loss_deep.backward()
            optimizer.step()
            loss_list.append(loss_deep.item())
            bar.set_description("epoch:{} idx:{} loss:{:.3f}".format(i, idx, np.mean(loss_list)))
            # if not (idx % 3000):
            #     torch.save(model.state_dict(), './' + path + '/model_' + fold + '.pth')
            #     torch.save(optimizer.state_dict(), './' + path + '/deep_optimizer_' + fold + '.pth')
        torch.save(optimizer.state_dict(), './' + path + '/deep_optimizer_' + fold +'.pth')
        torch.save(model.state_dict(), './' + path + '/model_' + fold + '.pth')


if __name__ == '__main__':
    if args.continue_training:
        torch.save(optimizer.state_dict(), './' + path + '/deep_optimizer_' + fold + str(i) + '.pth')
        torch.save(model.state_dict(), './' + path + '/model_' + fold + str(i) + '.pth')
        print('load model')
    ensure_dir('./' + path)
    train()

