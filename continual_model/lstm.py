# coding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils
from torch.autograd import Variable
from torch.nn import Parameter, init
from torch.nn.modules.rnn import RNNCellBase
    
    
class ParentFeedingLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ParentFeedingLSTMCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_i = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_i = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_i_p = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = Parameter(torch.Tensor(hidden_size))

        self.W_f = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_f = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_f_p = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = Parameter(torch.Tensor(hidden_size))
        self.b_f_p = Parameter(torch.Tensor(hidden_size))

        self.W_c = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_c = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_c_p = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_c = Parameter(torch.Tensor(hidden_size))

        self.W_o = Parameter(torch.Tensor(hidden_size, input_size))
        self.U_o = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.U_o_p = Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):

        init.orthogonal_(self.W_i)
        init.orthogonal_(self.U_i)
        init.orthogonal_(self.U_i_p)

        init.orthogonal_(self.W_f)
        init.orthogonal_(self.U_f)
        init.orthogonal_(self.U_f_p)

        init.orthogonal_(self.W_c)
        init.orthogonal_(self.U_c)
        init.orthogonal_(self.U_c_p)

        init.orthogonal_(self.W_o)
        init.orthogonal_(self.U_o)
        init.orthogonal_(self.U_o_p)

        self.b_i.data.fill_(0.)
        self.b_c.data.fill_(0.)
        self.b_o.data.fill_(0.)
        # forget bias set to 1.
        self.b_f.data.fill_(1.)
        self.b_f_p.data.fill_(1.)

    def forward(self, input, hidden_states):
        h_tm1, c_tm1, h_tm1_p, c_tm1_p = hidden_states
        i_t = F.sigmoid(F.linear(input, self.W_i) + F.linear(h_tm1, self.U_i) + F.linear(h_tm1_p, self.U_i_p) + self.b_i)

        xf_t = F.linear(input, self.W_f)
        f_t = F.sigmoid(xf_t + F.linear(h_tm1, self.U_f) + self.b_f)
        f_t_p = F.sigmoid(xf_t + F.linear(h_tm1_p, self.U_f_p) + self.b_f_p)

        xc_t = F.linear(input, self.W_c) + F.linear(h_tm1, self.U_c) + F.linear(h_tm1_p, self.U_c_p) + self.b_c
        c_t = f_t * c_tm1 + f_t_p * c_tm1_p + i_t * F.tanh(xc_t)

        o_t = F.sigmoid(F.linear(input, self.W_o) + F.linear(h_tm1, self.U_o) + F.linear(h_tm1_p, self.U_o_p) + self.b_o)
        h_t = o_t * F.tanh(c_t)

        return h_t, c_t