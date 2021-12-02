# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.utils
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

from model.nn_utils import masked_log_softmax


class PointerNet(nn.Module):
    def __init__(self, query_vec_size, src_encoding_size, attention_type='affine'):
        super(PointerNet, self).__init__()

        self.src_encoding_linear = None
        #if not query_vec_size == src_encoding_size:
        self.src_encoding_linear = nn.Linear(src_encoding_size, query_vec_size, bias=False)

        self.attention_type = attention_type

    def forward(self, src_encodings, src_token_mask, query_vec):
        """
        :param src_encodings: Variable(batch_size, src_sent_len, hidden_size * 2)
        :param src_token_mask: Variable(batch_size, src_sent_len)
        :param query_vec: Variable(tgt_action_num, batch_size, query_vec_size)
        :return: Variable(tgt_action_num, batch_size, src_sent_len)
        """

        # (batch_size, 1, src_sent_len, query_vec_size)
        #if not self.src_encoding_linear == None:
        #print(query_vec.size())
        #print(src_encodings.size())
        src_encodings = self.src_encoding_linear(src_encodings)
        src_encodings = src_encodings.unsqueeze(1)

        # (batch_size, tgt_action_num, query_vec_size, 1)
        q = query_vec.permute(1, 0, 2).unsqueeze(3)

        # (batch_size, tgt_action_num, src_sent_len)
        weights = torch.matmul(src_encodings, q).squeeze(3)

        # (tgt_action_num, batch_size, src_sent_len)
        weights = weights.permute(1, 0, 2)

        if src_token_mask is not None:
            # (tgt_action_num, batch_size, src_sent_len)
            src_token_mask = src_token_mask.unsqueeze(0).expand_as(weights)
            log_ptr_weights = masked_log_softmax(weights, ~src_token_mask, dim=-1)
            weights.data.masked_fill_(src_token_mask, -float('inf'))
        else:
            log_ptr_weights = F.log_softmax(weights, dim=-1)

        ptr_weights = F.softmax(weights, dim=-1)

        return log_ptr_weights, ptr_weights
