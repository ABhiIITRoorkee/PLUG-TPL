import torch
import torch.nn as nn
from utility.parse_args import arg_parse
args = arg_parse()


def print_list_shape(x):
    shape = []
    while isinstance(x, list):
        shape.append(len(x))
        x = x[0] if len(x) > 0 else []
    print("List shape:", shape)


class AttentionModule(nn.Module):
    def __init__(self):
        super(AttentionModule, self).__init__()
        self.W_K = nn.Sequential(
            nn.Linear(args.tpl_range, args.n_heads * args.d_k),
            nn.Sigmoid()
        )
        self.W_Q = nn.Sequential(
            nn.Linear(args.tpl_range, args.n_heads * args.d_q),
            nn.Sigmoid()
        )
        self.softmax = nn.Softmax(dim=-1)


    def forward(self, input_k, input_q, context_len):
        batch_size, max_len, _ = input_k.size()

        # K: (batch, n_heads, len, d_k), Q: (batch, d_q) -> (batch, 1, 1, d_q)
        K = self.W_K(input_k).reshape(batch_size, -1, args.n_heads, args.d_k).transpose(1,
                                                                                        2)  # (batch, n_heads, len, d_k)
        Q = self.W_Q(input_q).reshape(batch_size, -1, args.d_q)  # (batch, 1, d_q)
        Q = Q.unsqueeze(2)  # (batch, 1, 1, d_q)

        # QK^T: (batch, 1, 1, d_q) x (batch, n_heads, d_k, len) -> (batch, n_heads, 1, len)
        QK_t = torch.matmul(Q, K.transpose(-1, -2))  # (batch, n_heads, 1, len)
        QK_t_new = QK_t.squeeze(2).squeeze(1)  # (batch, len)

        mask = torch.arange(max_len, device=input_k.device).expand(batch_size, max_len)
        mask = mask < context_len.unsqueeze(1)  # (batch, len), bool

        QK_t_new_masked = QK_t_new.masked_fill(~mask, float('-inf'))
        attention_score = self.softmax(QK_t_new_masked)  # (batch, len)

        attention_score = attention_score.unsqueeze(-1)  # (batch, len, 1)
        context = torch.sum(attention_score * input_k, dim=1)  # (batch, input_k_dim)

        return context, attention_score


def custom_one_hot(indices, num_classes):
    shape = indices.shape
    flat_indices = indices.view(-1)
    one_hot = torch.zeros(len(flat_indices), num_classes, device=indices.device)
    out_of_range_indices = (flat_indices < 0) | (flat_indices >= num_classes)
    in_range_indices = (flat_indices >= 0) & (flat_indices < num_classes)
    one_hot[in_range_indices, flat_indices[in_range_indices]] = 1
    return one_hot.view(*shape, num_classes)


class AttenTPL(nn.Module):
    def __init__(self):
        super(AttenTPL, self).__init__()

        self.attn_layer = AttentionModule()

        self.deep = nn.Sequential(
            nn.Linear(args.n_heads * args.tpl_range + 512 + args.tpl_range, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.sigmoid_act_fun = nn.Sigmoid()


    def forward(self, input):
        tpl_context = input['tpl_context'] - 1
        encoded_tpl_context = custom_one_hot(tpl_context, args.tpl_range)
        candidate_tpl = input['target_tpl'] - 1
        encoded_target_tpl = custom_one_hot(candidate_tpl, args.tpl_range)

        input_k, input_q = encoded_tpl_context, encoded_target_tpl
        context_len = input['context_len']
        input_k = input_k.to(args.device).float()
        input_q = input_q.to(args.device).float()

        tpl_context, atten_score = self.attn_layer(input_k, input_q, context_len)
        tpl_context = tpl_context.to(args.device).float()

        function = input['function'].to(args.device)
        function_pooled = function

        target_tpl = encoded_target_tpl.to(args.device).float()

        deep_inputs = [function_pooled, tpl_context, target_tpl]
        deep_input = torch.cat(deep_inputs, dim=1)

        deep_output = self.deep(deep_input)

        output = self.sigmoid_act_fun(deep_output)

        return output
