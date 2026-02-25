#Plug-TPL/utility/parse_args.py



import argparse
import torch


def arg_parse():
    args = argparse.ArgumentParser()

    args.add_argument('--tpl_range', nargs='?', default=763, type=int)

    # 训练集路径
    args.add_argument('--training_data_path', nargs='?', default='./training data/', type=str)

    # 测试集路径
    args.add_argument('--testing_data_path', nargs='?', default='./testing data/', type=str)

    # app描述文本feature的路径
    args.add_argument('--function_feature_path', nargs='?', default='./data/function_feature_dict.pt', type=str)

    # 输出结果路径
    args.add_argument('--output_path', nargs='?', default='./output/', type=str)

    # 训练数据集
    args.add_argument('--train_dataset', nargs='?', default='training_0.json', type=str)

    # 测试数据集
    args.add_argument('--test_dataset', nargs='?', default='testing_0_1.json', type=str)

    # 模型和训练的参数
    args.add_argument('--n_heads', nargs='?', default=1, type=int)

    args.add_argument('--d_k', nargs='?', default=763, type=int)

    args.add_argument('--d_v', nargs='?', default=763, type=int)

    args.add_argument('--d_q', nargs='?', default=763, type=int)

    args.add_argument('--continue_training', nargs='?', default=0, type=int)

    args.add_argument('--train_batch_size', nargs='?', default=2048, type=int)

    args.add_argument('--epoch', nargs='?', default=8, type=int)

    args.add_argument('--lr', nargs='?', default=0.01, type=float)

    args.add_argument('--weight_decay', nargs='?', default=0.0001, type=float)

    args.add_argument('--fold', nargs='?', default=0, type=int,
                      help='Number of removed TPL')

    args.add_argument('--rm', nargs='?', default=3, type=int,
                      help='Number of removed TPL')

    args.add_argument('--device', nargs='?', default='cuda' if torch.cuda.is_available() else 'cpu', type=str,
                      help='Device to use (cuda or cpu)')

    args.add_argument(
                "--stage",
                type=str,
                default="C",
                choices=["A", "B", "C"],
                help="Stage ablation: A=context only, B=context+raw description, C=context+verified functions"
            )



    return args.parse_args()
