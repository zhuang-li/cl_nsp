# coding=utf-8
from itertools import chain

import torch
import torch.nn as nn
import torch.nn.utils
import numpy as np
from transformers import AutoTokenizer, BertTokenizer, AutoModel, BertModel

from components.dataset import Batch
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from common.registerable import Registrable
from model import nn_utils
import torch.nn.functional as F
from grammar.vertex import *
from grammar.rule import GenTAction, GenNTAction, Rule
from grammar.hypothesis import Hypothesis
from common.utils import update_args, init_arg_parser
from model.locked_dropout import LockedDropout
from model.attention_util import AttentionUtil, Attention
from model.lstm import ParentFeedingLSTMCell
import os

from model.utils import merge_vocab_entry


def copy_embedding_to_current_layer(pre_trained_vocab, current_vocab, pre_trained_layer, current_layer):
    if len(pre_trained_vocab) == len(current_vocab):
        print("vocab not need to update")
    else:
        print("update vocab pre-trained length {} and current length {}".format(len(pre_trained_vocab),
                                                                                len(current_vocab)))

    uninit_vocab = {}
    #print (is_exist_vocab)
    #print (pre_trained_vocab.token2id)
    for current_idx, current_token in current_vocab.id2token.items():
        if current_token in pre_trained_vocab:
            #print (type(current_token))

            pre_trained_id = pre_trained_vocab[current_token]
            #print (pre_trained_id)
            #print (pre_trained_layer.weight.data[pre_trained_id].size())
            #print (pre_trained_layer.weight.data.size())
            #print (current_layer.weight.size())
            if isinstance(current_layer, nn.Parameter):

                current_layer.data[current_idx]= pre_trained_layer.data[pre_trained_id]
            else:
                current_layer.weight.data[current_idx].copy_(pre_trained_layer.weight.data[pre_trained_id])

        else:
            uninit_vocab[current_token] = current_idx

    return uninit_vocab


def update_vocab_related_parameters(parser, loaded_vocab):
    parser.vocab = loaded_vocab


    print("update src embedding")
    pre_trained_src_vocab = parser.src_vocab
    parser.src_vocab = loaded_vocab.source

    pre_trained_src_embedding = parser.src_embed
    parser.src_embed = nn.Embedding(len(parser.src_vocab), parser.embed_size)
    nn.init.xavier_normal_(parser.src_embed.weight.data)


    unit_src_vocab = copy_embedding_to_current_layer(pre_trained_src_vocab, parser.src_vocab, pre_trained_src_embedding,
                                                     parser.src_embed)

    print("update nt action embedding")
    pre_trained_nt_action_vocab = parser.nt_action_vocab
    parser.nt_action_vocab = loaded_vocab.nt_action

    pre_trained_action_nt_embed = parser.action_nt_embed
    parser.action_nt_embed = nn.Embedding(len(parser.nt_action_vocab), parser.action_embed_size)
    nn.init.xavier_normal_(parser.action_nt_embed.weight.data)

    unit_nt_action_vocab = copy_embedding_to_current_layer(pre_trained_nt_action_vocab, parser.nt_action_vocab,
                                                        pre_trained_action_nt_embed, parser.action_nt_embed)


    print("update t action embedding")
    pre_trained_t_action_vocab = parser.t_action_vocab
    parser.t_action_vocab = loaded_vocab.t_action

    pre_trained_action_t_embed = parser.action_t_embed
    parser.action_t_embed = nn.Embedding(len(parser.t_action_vocab), parser.action_embed_size)
    nn.init.xavier_normal_(parser.action_t_embed.weight.data)

    unit_t_action_vocab = copy_embedding_to_current_layer(pre_trained_t_action_vocab, parser.t_action_vocab,
                                                        pre_trained_action_t_embed, parser.action_t_embed)


    print("update bias embedding")
    #for k in parser.named_parameters():
     #   print (k[0])
    pre_trained_t_readout_b = parser.t_readout_b
    #parser.t_readout_b = nn.Parameter(torch.FloatTensor(len(parser.t_action_vocab)).zero_())
    parser.register_parameter('t_readout_b', nn.Parameter(torch.FloatTensor(len(parser.t_action_vocab)).zero_()))

    copy_embedding_to_current_layer(pre_trained_t_action_vocab, parser.t_action_vocab,
                                                        pre_trained_t_readout_b, parser.t_readout_b)



    pre_trained_nt_readout_b = parser.nt_readout_b

    #parser.nt_readout_b = nn.Parameter(torch.FloatTensor(len(parser.nt_action_vocab)).zero_())

    parser.register_parameter('nt_readout_b', nn.Parameter(torch.FloatTensor(len(parser.nt_action_vocab)).zero_()))

    copy_embedding_to_current_layer(pre_trained_nt_action_vocab, parser.nt_action_vocab,
                                                        pre_trained_nt_readout_b, parser.nt_readout_b)
    #parser.nt_readout_b.detach()
    #parser.register_parameter('nt_readout_b', parser.nt_readout_b)
    # label smoothing
    print("update label smoothing layer")

    parser.t_label_smoothing_layer = nn_utils.LabelSmoothing(parser.label_smoothing, len(parser.t_action_vocab),
                                                           ignore_indices=[0], use_cuda=parser.use_cuda)
    parser.nt_label_smoothing_layer = nn_utils.LabelSmoothing(parser.label_smoothing, len(parser.nt_action_vocab),
                                                            ignore_indices=[0], use_cuda=parser.use_cuda)

    parser.nt_readout = lambda q: F.linear(q, parser.action_nt_embed.weight, parser.nt_readout_b)
    parser.t_readout = lambda q: F.linear(q, parser.action_t_embed.weight, parser.t_readout_b)

    return unit_src_vocab, unit_nt_action_vocab, unit_t_action_vocab

@Registrable.register('seq2seq_topdown')
class Seq2SeqModel(nn.Module):
    """
    a standard seq2seq model
    """

    def __init__(self, vocab, args):
        super(Seq2SeqModel, self).__init__()
        self.use_cuda = args.use_cuda
        self.embed_size = args.embed_size
        self.action_embed_size = args.action_embed_size
        self.hidden_size = args.hidden_size
        self.vocab = vocab
        self.args = args
        self.src_vocab = vocab.source
        self.nt_action_vocab = vocab.nt_action
        self.t_action_vocab = vocab.t_action

        self.utterance_embed = None

        self.src_embed = nn.Embedding(len(self.src_vocab), self.embed_size)

        self.action_nt_embed = nn.Embedding(len(self.nt_action_vocab), self.action_embed_size)
        self.action_t_embed = nn.Embedding(len(self.t_action_vocab), self.action_embed_size)

        nn.init.xavier_normal_(self.src_embed.weight.data)

        nn.init.xavier_normal_(self.action_nt_embed.weight.data)
        nn.init.xavier_normal_(self.action_t_embed.weight.data)

        # general action embed

        # whether to use att
        self.use_att = args.use_att
        if args.use_att:
            self.decoder_size = self.action_embed_size + self.action_embed_size
        else:
            self.decoder_size = self.action_embed_size

        self.encoder_lstm = nn.LSTM(self.embed_size, self.hidden_size, bidirectional=True)
        self.lstm = args.lstm
        if self.lstm == 'lstm':
            self.decoder_lstm = nn.LSTMCell(self.decoder_size, self.hidden_size)
        else:
            from .lstm import ParentFeedingLSTMCell
            self.decoder_lstm = ParentFeedingLSTMCell(self.decoder_size, self.hidden_size)




        # initialize the decoder's state and cells with encoder hidden states
        self.decoder_cell_init = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # attention: dot product attention
        # project source encoding to decoder rnn's h space
        self.att_src_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)

        # transformation of decoder hidden states and context vectors before reading out target words
        # this produces the `attentional vector` in (Luong et al., 2015)
        self.att_vec_linear = nn.Linear(self.hidden_size * 2 + self.hidden_size, self.action_embed_size, bias=False)

        # supervised attention
        self.sup_attention = args.sup_attention

        # dropout layer
        self.dropout = args.dropout
        if self.dropout > 0:
            self.dropout = nn.Dropout(self.dropout)

        self.dropout_i = args.dropout_i
        if self.dropout_i > 0:
            self.dropout_i = nn.Dropout(self.dropout)

        self.nt_readout_b = nn.Parameter(torch.FloatTensor(len(self.nt_action_vocab)).zero_())
        self.t_readout_b = nn.Parameter(torch.FloatTensor(len(self.t_action_vocab)).zero_())


        self.nt_readout = lambda q: F.linear(q, self.action_nt_embed.weight, self.nt_readout_b)
        self.t_readout = lambda q: F.linear(q, self.action_t_embed.weight, self.t_readout_b)
        # project the attention vector stack and reduce head into the embedding space
        # self.attention_stack_reduce_linear = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=False)
        # project the hidden vector stack and reduce head into the embedding space

        self.label_smoothing = None

        # label smoothing
        self.label_smoothing = args.label_smoothing
        if self.label_smoothing:
            self.t_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.t_action_vocab),
                                                                 ignore_indices=[0], use_cuda=self.use_cuda)
            self.nt_label_smoothing_layer = nn_utils.LabelSmoothing(self.label_smoothing, len(self.nt_action_vocab),
                                                                 ignore_indices=[0], use_cuda=self.use_cuda)

        if args.use_cuda:
            self.new_long_tensor = torch.cuda.LongTensor
            self.new_tensor = torch.cuda.FloatTensor
        else:
            self.new_long_tensor = torch.LongTensor
            self.new_tensor = torch.FloatTensor

        self.mask_action = self.args.mask_action

        if self.mask_action:
            self.mask_action_embed = nn.Parameter(self.new_tensor(self.action_embed_size))
            #print (self.mask_action.requires_grad)
            #nn.init.xavier_normal_(self.mask_action)

        self.pad_action_embed = nn.Parameter(self.new_tensor(self.action_embed_size))

        self.known_action_set = set()
        self.known_action_set.add(GenNTAction(RuleVertex('<pad>')))
        self.known_action_set.add(GenTAction(RuleVertex('<pad>')))
        self.training_state = 'normal_train'

        self.embed_type = args.embed_type
        if self.embed_type == 'bert-mini':
            self.plmm_tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')  # Initialize tokenizer
            self.plmm_model = AutoModel.from_pretrained('prajjwal1/bert-mini')
            #for param in self.plmm_model.parameters():
            #    param.requires_grad = False
        elif self.embed_type == 'bert':
            #>> > tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            #>> > model = BertModel.from_pretrained('bert-base-uncased')
            self.plmm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', cache_dir='/scratch/dz21/zl1166/cross_template_semantic_parsing/bert_model_cache', local_files_only=True)  # Initialize tokenizer
            self.plmm_model = BertModel.from_pretrained('bert-base-uncased', cache_dir='/scratch/dz21/zl1166/cross_template_semantic_parsing/bert_model_cache', local_files_only=True)
        elif self.embed_type == 'bert-mini-fix':
            self.plmm_tokenizer = AutoTokenizer.from_pretrained('prajjwal1/bert-mini')  # Initialize tokenizer
            self.plmm_model = AutoModel.from_pretrained('prajjwal1/bert-mini')
            for param in self.plmm_model.parameters():
                param.requires_grad = False
            self.weighted_score = nn.Parameter(torch.rand(4)).cuda()
        elif self.embed_type == 'bert-fix':
            #>> > tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            #>> > model = BertModel.from_pretrained('bert-base-uncased')
            self.plmm_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Initialize tokenizer
            self.plmm_model = BertModel.from_pretrained('bert-base-uncased')
            for param in self.plmm_model.parameters():
                param.requires_grad = False
            self.weighted_score = nn.Parameter(torch.rand(12)).cuda()

        self.att_weights = None

        #if args.second_iter:
        #    self.nt_stable_embed = nn.Embedding(len(self.nt_action_vocab), self.action_embed_size)
        #    self.t_stable_embed = nn.Embedding(len(self.t_action_vocab), self.action_embed_size)


    def dot_prod_attention(self, h_t, src_encoding, src_encoding_att_linear, mask=None):
        """
        :param h_t: (batch_size, hidden_size)
        :param src_encoding: (batch_size, src_sent_len, hidden_size * 2)
        :param src_encoding_att_linear: (batch_size, src_sent_len, hidden_size)
        :param mask: (batch_size, src_sent_len)
        """
        # (batch_size, src_sent_len)
        att_weight = torch.bmm(src_encoding_att_linear, h_t.unsqueeze(2)).squeeze(2)
        if mask is not None:
            att_weight.data.masked_fill_(mask, -float('inf'))

        att_weight = torch.softmax(att_weight, dim=-1)
        # print (att_weight)
        # print (torch.sum(att_weight[0]))
        att_view = (att_weight.size(0), 1, att_weight.size(1))
        # (batch_size, hidden_size)
        ctx_vec = torch.bmm(att_weight.view(*att_view), src_encoding).squeeze(1)

        return ctx_vec, att_weight

    def encode(self, plmm_src_token_embed, src_sents_var, src_sents_len):
        """
        encode the source sequence
        :return:
            src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
            dec_init_state, dec_init_cell: Variable(batch_size, hidden_size)
        """

        # (tgt_query_len, batch_size, embed_size)
        # print (src_sents_var.size())
        if self.embed_type == 'glove':
            src_token_embed = self.src_embed(src_sents_var)
            # print(src_token_embed.size())
        elif self.embed_type == 'bert-mini' or self.embed_type == 'bert' or self.embed_type == 'bert-mini-fix' or self.embed_type == 'bert-fix':
            src_token_embed = plmm_src_token_embed

        if self.dropout_i:
            src_token_embed = self.dropout_i(src_token_embed)
        packed_src_token_embed = pack_padded_sequence(src_token_embed, src_sents_len, enforce_sorted = False)
        # src_encodings: (tgt_query_len, batch_size, hidden_size)
        src_encodings, (last_state, last_cell) = self.encoder_lstm(packed_src_token_embed)
        src_encodings, _ = pad_packed_sequence(src_encodings)
        # print (last_state.size())
        # (batch_size, hidden_size * 2)
        last_state = torch.cat([last_state[0], last_state[1]], 1)
        last_cell = torch.cat([last_cell[0], last_cell[1]], 1)

        return src_encodings, (last_state, last_cell)

    def init_decoder_state(self, enc_last_state, enc_last_cell):
        """Compute the initial decoder hidden state and cell state"""

        h_0 = self.decoder_cell_init(enc_last_cell)
        h_0 = torch.tanh(h_0)

        return h_0, self.new_tensor(h_0.size()).zero_()

    def decode(self, src_encodings, src_sent_masks, dec_init_vec, batch):
        """
        compute the final softmax layer at each decoding step
        :param src_encodings: Variable(src_sent_len, batch_size, hidden_size * 2)
        :param src_sents_len: list[int]
        :param dec_init_vec: tuple((batch_size, hidden_size))
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            scores: Variable(src_sent_len, batch_size, src_vocab_size)
        """

        new_tensor = src_encodings.data.new
        batch_size = src_encodings.size(1)
        if self.lstm == 'lstm':
            h_tm1 = dec_init_vec[0], dec_init_vec[1]
        else:
            h_tm1 = dec_init_vec[0], dec_init_vec[1], \
                    self.new_tensor(batch_size, self.hidden_size).zero_(), \
                    self.new_tensor(batch_size, self.hidden_size).zero_()

        #zero_action_embed = self.new_tensor(self.action_embed_size).zero_()


        # (batch_size, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)
        # print (src_encodings.size())
        # (batch_size, query_len, hidden_size)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        # initialize the attentional vector
        att_tm1 = new_tensor(batch_size, self.action_embed_size).zero_()
        assert att_tm1.requires_grad == False, "the att_tm1 requires grad is False"


        att_weights = []

        att_vecs = []
        history_states = []

        for t in range(batch.max_action_num):
            # the input to the decoder LSTM is a concatenation of multiple signals
            # [
            #   embedding of previous action -> `a_tm1_embed`,
            #   previous attentional vector -> `att_tm1`,
            #   embedding of the current frontier (parent) constructor (rule) -> `parent_production_embed`,
            #   embedding of the frontier (parent) field -> `parent_field_embed`,
            #   embedding of the ASDL type of the frontier field -> `parent_field_type_embed`,
            #   LSTM state of the parent action -> `parent_states`
            # ]

            if t == 0:
                x = self.new_tensor(batch_size, self.decoder_lstm.input_size).zero_()
            else:
                a_tm1_embeds = []
                for example in batch.examples:
                    # action t - 1
                    #print(self.action_nt_embed.weight.size())

                    if t <= len(example.tgt_actions):
                        a_tm1 = example.tgt_actions[t - 1]
                        if self.training_state == 'init_embedding':
                            use_mask_action = not (a_tm1 in self.known_action_set)
                        elif self.training_state == 'mask_training':
                            self.known_action_set.add(a_tm1)
                            random_num = np.random.uniform()
                            if random_num < self.mask_action:
                                use_mask_action = True
                            else:
                                use_mask_action = False
                        elif self.training_state == 'normal_train':
                            use_mask_action = False
                            self.known_action_set.add(a_tm1)
                        elif self.training_state == 'init_train':
                                #or self.training_state == 'recon_train':
                            use_mask_action = False

                        if isinstance(a_tm1, GenNTAction):
                            a_tm1_embed = self.action_nt_embed.weight[self.nt_action_vocab[a_tm1]]

                        elif isinstance(a_tm1, GenTAction):
                            a_tm1_embed = self.action_t_embed.weight[self.t_action_vocab[a_tm1]]
                        else:
                            raise ValueError


                    else:
                        a_tm1_embed = self.pad_action_embed

                    a_tm1_embeds.append(a_tm1_embed)

                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if self.use_att:
                    inputs.append(att_tm1)
                if self.lstm == 'lstm':
                    h_tm1 = (h_tm1[0], h_tm1[1])
                else:
                    # append history states
                    actions_t = [e.tgt_actions[t-1] if t <= len(e.tgt_actions) else None for e in batch.examples]
                    #print ("==================================")
                    #for batch_id, p_t in enumerate(a_t.parent_t + 1 if a_t else 0 for a_t in actions_t):
                        #print (p_t)
                    #print ("==================================")
                    parent_states = torch.stack([history_states[p_t][0][batch_id]
                                                 for batch_id, p_t in
                                                 enumerate(a_t.parent_t + 1 if a_t else 0 for a_t in actions_t)])

                    parent_cells = torch.stack([history_states[p_t][1][batch_id]
                                                for batch_id, p_t in
                                                enumerate(a_t.parent_t + 1 if a_t else 0 for a_t in actions_t)])


                    h_tm1 = (h_tm1[0], h_tm1[1], parent_states, parent_cells)


                x = torch.cat(inputs, dim=-1)
            """
            print ("=======================")
            print(x.size())
            print (h_tm1[0].size())
            print (h_tm1[1].size())
            print (h_tm1[2].size())
            print (h_tm1[3].size())
            print ("=======================")
            """
            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, src_encodings,
                                                         src_encodings_att_linear,
                                                         src_sent_masks=src_sent_masks)



            history_states.append((h_t, cell_t))
            att_vecs.append(att_t)
            att_weights.append(att_weight.squeeze())
            att_tm1 = att_t
            #print (att_t.size())
            h_tm1 = (h_t, cell_t)

        att_vecs = torch.stack(att_vecs, dim=0)
        self.att_weights = torch.stack(att_weights, dim=0)
        return att_vecs

    def action_readout(self, query_vectors):
        return self.nt_readout(query_vectors), self.t_readout(query_vectors)


    def action_weight_readout(self, query_vectors, nt_weight, t_weight):
        #nt_readout = nt_linear(query_vectors)
        #t_readout = t_linear(query_vectors)

        nt_readout = F.linear(query_vectors, nt_weight, self.nt_readout_b)
        t_readout = F.linear(query_vectors, t_weight, self.t_readout_b)

        return nt_readout, t_readout

    def action_score(self, scores, batch, action_type = 'specific'):
        #test_action_prob = torch.softmax(scores, dim=-1)
        #if self.training_state == 'mask_training':
        #print ("first")
        #zero_index = (test_action_prob == 1).nonzero()

        #if zero_index.numel() != 0:
            #print(zero_index)
            #print(scores.size())
            #print (batch.nt_action_idx_matrix.unsqueeze(2)[6,23])
            #print (batch.nt_action_idx_matrix.unsqueeze(2)[8,25])
            #print (batch.nt_action_idx_matrix.unsqueeze(2)[15,19])
            #print(scores)
            #print(action_prob)

        #if self.mask_action > 0:
            #action_prob = action_prob.clamp(min=1e-45,max=0.9)

        #print ("second")
        #print ((action_prob == 0).nonzero())
        #print (action_prob)

        if action_type == 'specific':
            if self.training and self.label_smoothing:
                action_prob = torch.log_softmax(scores, dim=-1)
                # (tgt_action_len, batch_size)
                # this is actually the negative KL divergence size we will flip the sign later
                # tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                #     gen_from_vocab_prob.view(-1, gen_from_vocab_prob.size(-1)).log(),
                #     batch.primitive_idx_matrix.view(-1)).view(-1, len(batch))
                #print (batch.t_action_idx_matrix.size())
                tgt_t_action_log_prob = -self.t_label_smoothing_layer(
                    action_prob,
                    batch.t_action_idx_matrix)

            else:
                action_prob = torch.log_softmax(scores, dim=-1)
                # (tgt_action_len, batch_size)
                tgt_t_action_log_prob = torch.gather(action_prob, dim=2,
                                                 index=batch.t_action_idx_matrix.unsqueeze(2)).squeeze(2)
                #print (tgt_t_action_prob)
                #tgt_t_action_log_prob = tgt_t_action_prob.log()
            return -(tgt_t_action_log_prob * batch.t_action_mask).sum(dim=0)
        elif action_type == 'general':
            if self.training and self.label_smoothing:
                # (tgt_action_len, batch_size)
                # this is actually the negative KL divergence size we will flip the sign later
                # tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
                #     gen_from_vocab_prob.view(-1, gen_from_vocab_prob.size(-1)).log(),
                #     batch.primitive_idx_matrix.view(-1)).view(-1, len(batch))

                action_prob = torch.log_softmax(scores, dim=-1)
                tgt_nt_action_log_prob = -self.nt_label_smoothing_layer(
                    action_prob,
                    batch.nt_action_idx_matrix)

            else:
                action_prob = torch.log_softmax(scores, dim=-1)
                tgt_nt_action_log_prob = torch.gather(action_prob, dim=2,
                                                  index=batch.nt_action_idx_matrix.unsqueeze(2)).squeeze(2)
                #print (tgt_nt_action_prob)
                #tgt_nt_action_log_prob = tgt_nt_action_prob.log()
            return -(tgt_nt_action_log_prob * batch.nt_action_mask).sum(dim=0)

    def score_decoding_results(self, nt_scores, t_scores, batch):
        """
        :param scores: Variable(src_sent_len, batch_size, tgt_vocab_size)
        :param tgt_sents_var: Variable(src_sent_len, batch_size)
        :return:
            tgt_sent_log_scores: Variable(batch_size)
        """
        # (tgt_sent_len, batch_size, tgt_vocab_size)


        # ApplyRule (i.e., ApplyConstructor) action probabilities
        # (tgt_action_len, batch_size, grammar_size)
        #print (query_vectors.size())

        #nt_action_prob = torch.softmax(nt_scores, dim=-1)

        #print (nt_action_prob.size())
        #print ((batch.nt_action_idx_matrix.unsqueeze(2)).size())
        # probabilities of target (gold-standard) ApplyRule actions
        # (tgt_action_len, batch_size)
        #print (batch.t_action_idx_matrix)

        #tgt_nt_action_prob = torch.gather(nt_action_prob, dim=2, index=batch.nt_action_idx_matrix.unsqueeze(2)).squeeze(2)

        #### compute generation and copying probabilities

        # (tgt_action_len, batch_size, primitive_vocab_size)
        #t_action_prob = torch.softmax(t_scores, dim=-1)


        """
        if self.training and self.label_smoothing:
            # (tgt_action_len, batch_size)
            # this is actually the negative KL divergence size we will flip the sign later
            # tgt_primitive_gen_from_vocab_log_prob = -self.label_smoothing(
            #     gen_from_vocab_prob.view(-1, gen_from_vocab_prob.size(-1)).log(),
            #     batch.primitive_idx_matrix.view(-1)).view(-1, len(batch))

            tgt_t_action_log_prob = -self.t_label_smoothing_layer(
                t_action_prob.log(),
                batch.t_action_idx_matrix)

            tgt_nt_action_log_prob = -self.nt_label_smoothing_layer(
                nt_action_prob.log(),
                batch.nt_action_idx_matrix)

        else:
            # (tgt_action_len, batch_size)
            tgt_t_action_prob = torch.gather(t_action_prob, dim=2,
                                             index=batch.t_action_idx_matrix.unsqueeze(2)).squeeze(2)

            tgt_t_action_log_prob = tgt_t_action_prob.log()

            tgt_nt_action_prob = torch.gather(nt_action_prob, dim=2,
                                              index=batch.nt_action_idx_matrix.unsqueeze(2)).squeeze(2)

            tgt_nt_action_log_prob = tgt_nt_action_prob.log()

        # (tgt_action_len, batch_size)
        action_prob = tgt_nt_action_log_prob * batch.nt_action_mask + \
                      tgt_t_action_log_prob * batch.t_action_mask
        """

        loss = self.action_score(t_scores, batch, action_type = 'specific') + self.action_score(nt_scores, batch, action_type = 'general')

        #print (action_prob.size())
        #scores = action_prob.sum(dim=0)
        #print (scores)
        #loss = -scores
        return loss


    def step(self, x, h_tm1, src_encodings, src_encodings_att_linear, src_sent_masks=None):
        """
        a single LSTM decoding step
        """
        # h_t: (batch_size, hidden_size)
        h_t, cell_t = self.decoder_lstm(x, h_tm1)

        ctx_t, alpha_t = self.dot_prod_attention(h_t, src_encodings, src_encodings_att_linear, mask=src_sent_masks)
        # print (src_sent_masks)
        # vector for action prediction
        att_t = torch.tanh(self.att_vec_linear(torch.cat([h_t, ctx_t], 1)))  # E.q. (5)
        if self.dropout:
            att_t = self.dropout(att_t)

        return (h_t, cell_t), att_t, alpha_t

    def forward(self, batch):
        """
        encode source sequence and compute the decoding log likelihood
        :param src_sents_var: Variable(src_sent_len, batch_size)
        :param src_sents_len: list[int]
        :param tgt_sents_var: Variable(tgt_sent_len, batch_size)
        :return:
            tgt_token_scores: Variable(tgt_sent_len, batch_size, tgt_vocab_size)
        """
        #src_batch = Batch(examples, self.vocab, use_cuda=self.use_cuda)
        # src_batch = Batch(examples, self.vocab, use_cuda=self.use_cuda)
        src_sents_var = batch.src_sents_var
        src_sents_len = batch.src_sents_len
        plmm_src_token_embed = None
        if self.embed_type == 'bert-mini' or self.embed_type == 'bert':
            batch_src_sents = [" ".join(scent_list) for scent_list in batch.src_sents]
            #print(batch_src_sents)
            inputs = self.plmm_tokenizer(batch_src_sents, return_tensors='pt',padding=True, truncation=True, return_length=True)
            src_sents_len = inputs['attention_mask'].sum(dim=1).tolist()
            if self.use_cuda:
                inputs['input_ids'] = inputs['input_ids'].cuda()
            plmm_src_token_embed = self.plmm_model(inputs['input_ids']).last_hidden_state
            plmm_src_token_embed = plmm_src_token_embed.permute(1,0,2)
            #print ("======================")
        elif self.embed_type == 'bert-mini-fix' or self.embed_type == 'bert-fix':
            batch_src_sents = [" ".join(scent_list) for scent_list in batch.src_sents]
            # print(batch_src_sents)
            inputs = self.plmm_tokenizer(batch_src_sents, return_tensors='pt', padding=True, truncation=True,
                                         return_length=True)
            src_sents_len = inputs['attention_mask'].sum(dim=1).tolist()
            if self.use_cuda:
                inputs['input_ids'] = inputs['input_ids'].cuda()
            hidden_states = self.plmm_model(inputs['input_ids'], output_hidden_states=True).hidden_states[1:]
            stacked_hidden_state = torch.stack(hidden_states, dim=-1)

            att_weight = torch.softmax(self.weighted_score, dim=0)

            plmm_src_token_embed = torch.matmul(stacked_hidden_state, att_weight)

            plmm_src_token_embed = plmm_src_token_embed.permute(1, 0, 2)
        # action_seq_var = batch.action_seq_var

        src_encodings, (last_state, last_cell) = self.encode(plmm_src_token_embed, src_sents_var, src_sents_len)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        self.utterance_embed = dec_init_vec[0]
        src_sent_masks = nn_utils.length_array_to_mask_tensor(src_sents_len, use_cuda=self.use_cuda)


        query_vectors = self.decode(src_encodings, src_sent_masks, dec_init_vec, batch)
        #loss = self.score_decoding_results(query_vectors, src_batch)
        #ret_val = [loss]
        # print(loss.data)
        return query_vectors

    def beam_search(self, example, decode_max_time_step, t_offset1 = -1, t_offset2 = -1, nt_offset = -1, beam_size=5):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        # print ("============================")
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.src_vocab,
                                                   use_cuda=self.use_cuda, append_boundary_sym=True, mode='token')

        """
        print ("===============================preview===================================")
        print(" ".join(example.src_sent))
        print(example.tgt_code_no_var_str)
        print ("===============================preview====================================")
        """
        # TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
        src_sents_len = [src_sents_var.size(0)]
        plmm_src_token_embed = None
        if self.embed_type == 'bert-mini' or self.embed_type == 'bert':
            batch_src_sents = [" ".join(example.src_sent)]
            # print(batch_src_sents)
            inputs = self.plmm_tokenizer(batch_src_sents, return_tensors='pt', padding=True, truncation=True,
                                         return_length=True)
            # print(inputs)
            src_sents_len = inputs['attention_mask'].sum(dim=1).tolist()
            # print (src_sents_len)
            # print (src_sents_len)
            if self.use_cuda:
                inputs['input_ids'] = inputs['input_ids'].cuda()
            plmm_src_token_embed = self.plmm_model(inputs['input_ids']).last_hidden_state
            plmm_src_token_embed = plmm_src_token_embed.permute(1, 0, 2)
        elif self.embed_type == 'bert-mini-fix' or self.embed_type == 'bert-fix':
            batch_src_sents = [" ".join(example.src_sent)]
            # print(batch_src_sents)
            inputs = self.plmm_tokenizer(batch_src_sents, return_tensors='pt', padding=True, truncation=True,
                                         return_length=True)
            src_sents_len = inputs['attention_mask'].sum(dim=1).tolist()
            if self.use_cuda:
                inputs['input_ids'] = inputs['input_ids'].cuda()
            hidden_states = self.plmm_model(inputs['input_ids'], output_hidden_states=True).hidden_states[1:]
            stacked_hidden_state = torch.stack(hidden_states, dim=-1)

            att_weight = torch.softmax(self.weighted_score, dim=0)

            plmm_src_token_embed = torch.matmul(stacked_hidden_state, att_weight)

            plmm_src_token_embed = plmm_src_token_embed.permute(1, 0, 2)


        src_encodings, (last_state, last_cell) = self.encode(plmm_src_token_embed, src_sents_var, src_sents_len)

        # (1, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)
        src_encodings_att_linear = self.att_src_linear(src_encodings)
        dec_init_vec = self.init_decoder_state(last_state, last_cell)
        if self.lstm == 'lstm':
            h_tm1 = dec_init_vec[0], dec_init_vec[1]
        else:
            h_tm1 = dec_init_vec[0], dec_init_vec[1], \
                    self.new_tensor(1, self.hidden_size).zero_(), \
                    self.new_tensor(1, self.hidden_size).zero_()


        zero_action_embed = self.new_tensor(self.action_embed_size).zero_()

        att_tm1 = torch.zeros(1, self.action_embed_size, requires_grad=True)
        hyp_scores = torch.zeros(1, requires_grad=True)

        if self.use_cuda:
            att_tm1 = att_tm1.cuda()
            hyp_scores = hyp_scores.cuda()
        # todo change it back
        # eos_id = self.action_vocab['</s>']


        first_hyp = Hypothesis()
        # first_hyp.embedding_stack.append(att_tm1)
        hyp_states = [[]]
            # hypotheses = [[bos_id]]
        hypotheses = [first_hyp]
        completed_hypotheses = []

        t = 0
        while len(completed_hypotheses) < beam_size and t < decode_max_time_step and len(hypotheses) > 0:
            # if t == 50:
            # print (t)
            hyp_num = len(hypotheses)

            expanded_src_encodings = src_encodings.expand(hyp_num, src_encodings.size(1), src_encodings.size(2))

            expanded_src_encodings_att_linear = src_encodings_att_linear.expand(hyp_num,
                                                                                src_encodings_att_linear.size(1),
                                                                                src_encodings_att_linear.size(2))


            if t == 0:
                x = self.new_tensor(1, self.decoder_lstm.input_size).zero_()
            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]
                #print (actions_tm1)
                a_tm1_embeds = []
                for a_tm1 in actions_tm1:
                    if a_tm1:
                        if isinstance(a_tm1, GenNTAction):
                            a_tm1_embed = self.action_nt_embed.weight[self.nt_action_vocab[a_tm1]]
                        elif isinstance(a_tm1, GenTAction):
                            a_tm1_embed = self.action_t_embed.weight[self.t_action_vocab[a_tm1]]
                        else:
                            raise ValueError
                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if self.use_att:
                    inputs.append(att_tm1)
                if self.lstm == 'lstm':
                    h_tm1 = (h_tm1[0], h_tm1[1])
                else:
                    #print ('==============================')
                    p_ts = [hyp.frontier_node.created_time for hyp in hypotheses]
                    #print (hypotheses[0].tree)
                    #print (hypotheses[0].actions)
                    #print (hypotheses[0].frontier_node)
                    #print (hypotheses[0].frontier_field)
                    #print(p_ts)
                    parent_states = torch.stack([hyp_states[hyp_id][p_t][0] for hyp_id, p_t in enumerate(p_ts)])
                    parent_cells = torch.stack([hyp_states[hyp_id][p_t][1] for hyp_id, p_t in enumerate(p_ts)])


                    h_tm1 = (h_tm1[0], h_tm1[1], parent_states, parent_cells)
                #for i in inputs:
                    #print (i.size())
                x = torch.cat(inputs, dim=-1)

            # h_t: (hyp_num, hidden_size)
            """
            print ("=======================")
            print(x.size())
            print (h_tm1[0].size())
            print (h_tm1[1].size())
            print (h_tm1[2].size())
            print (h_tm1[3].size())
            print ("=======================")
            """
            (h_t, cell_t), att_t, att_weight = self.step(x, h_tm1, expanded_src_encodings,
                                                         expanded_src_encodings_att_linear,
                                                         src_sent_masks=None)

            #print (self.action_nt_embed.weight.size())

            if nt_offset >= 0:
                nt_action_log_prob = torch.log_softmax(self.nt_readout(att_t)[:, :nt_offset], dim=-1)
            else:
                nt_action_log_prob = torch.log_softmax(self.nt_readout(att_t), dim=-1)

            nt_action_log_prob[:,0] = -100

            # Variable(batch_size, primitive_vocab_size)

            t_action_score = self.t_readout(att_t)

            #print (t_action_score.size())

            if t_offset1 >= 0:
                pad_t_action_score = t_action_score[:, 0].unsqueeze(-1)
                task_t_action_score = t_action_score[:, t_offset1: t_offset2]
                t_action_score = torch.cat([pad_t_action_score, task_t_action_score], -1)

                t_action_log_prob = torch.log_softmax(t_action_score, dim=-1)
            else:
                t_action_log_prob = torch.log_softmax(t_action_score, dim=-1)

            t_action_log_prob[:,0] = -100
            #print (t_action_log_prob)

                # if src_unk_pos_list:
                #     primitive_prob[:, primitive_vocab.unk_id] = 1.e-10

            nt_prev_hyp_ids = []

            t_new_hyp_scores = []
            t_new_hyp_t_action_ids = []
            t_prev_hyp_ids = []


            nt_type_hyp_scores = []
            nt_type_hyp_nt_action_ids = []
            nt_type_prev_hyp_ids = []

            #print (len(hypotheses))
            if t==0:
                nt_prev_hyp_ids.append(0)
            else:
                for hyp_id, hyp in enumerate(hypotheses):
                    # generate new continuations
                    if hyp.frontier_field.head.startswith(TYPE_SIGN):
                        # print (self.t_action_vocab.lhs2rhsid[hyp.frontier_field.head])
                        for t_id in self.t_action_vocab.lhs2rhsid[hyp.frontier_field.head]:
                            if t_offset1 >= 0:
                                if t_id >= t_offset2 or t_id < t_offset1:
                                    continue

                                t_id = t_id - t_offset1 + 1

                            t_score = t_action_log_prob[hyp_id, t_id].data.item()
                            new_hyp_score = hyp.score + t_score
                            t_new_hyp_scores.append(new_hyp_score)
                            t_new_hyp_t_action_ids.append(t_id)
                            t_prev_hyp_ids.append(hyp_id)
                    elif hyp.frontier_field.head == NT:
                        nt_prev_hyp_ids.append(hyp_id)
                    elif hyp.frontier_field.head.startswith(NT_TYPE_SIGN):
                        for t_id in self.nt_action_vocab.lhs2rhsid[hyp.frontier_field.head]:
                            if nt_offset >= 0:
                                if t_id >= nt_offset:
                                    continue

                            nt_score = nt_action_log_prob[hyp_id, t_id].data.item()
                            new_nt_type_hyp_score = hyp.score + nt_score
                            nt_type_hyp_scores.append(new_nt_type_hyp_score)
                            nt_type_hyp_nt_action_ids.append(t_id)
                            nt_type_prev_hyp_ids.append(hyp_id)
                    else:
                        print(hyp.frontier_field.head)
                        raise ValueError


            # print (t_new_hyp_scores)

            # print (nt_prev_hyp_ids)

            new_hyp_scores = None
            if t_new_hyp_scores:
                new_hyp_scores = self.new_tensor(t_new_hyp_scores)

            if nt_type_hyp_scores:
                nt_type_hyp_scores_tensor = self.new_tensor(nt_type_hyp_scores)
                if new_hyp_scores is None:
                    new_hyp_scores = nt_type_hyp_scores_tensor
                else:
                    new_hyp_scores = torch.cat([new_hyp_scores, nt_type_hyp_scores_tensor])

            if nt_prev_hyp_ids:
                nt_new_hyp_scores = (
                            hyp_scores[nt_prev_hyp_ids].unsqueeze(1) + nt_action_log_prob[nt_prev_hyp_ids,
                                                                             :]).view(-1)
                #print (nt_new_hyp_scores.size())
                if new_hyp_scores is None:
                    new_hyp_scores = nt_new_hyp_scores
                else:
                    new_hyp_scores = torch.cat([new_hyp_scores, nt_new_hyp_scores])

            if new_hyp_scores is None:
                break

            top_new_hyp_scores, top_new_hyp_pos = torch.topk(new_hyp_scores,
                                                             k=min(new_hyp_scores.size(0),
                                                                   beam_size - len(completed_hypotheses)))
            #print ("===============")
            #print (top_new_hyp_scores.size())
            #print (top_new_hyp_pos.size())
            #print (len(t_new_hyp_scores))
            live_hyp_ids = []
            new_hypotheses = []
            for new_hyp_score, new_hyp_pos in zip(top_new_hyp_scores.data.cpu(), top_new_hyp_pos.data.cpu()):
                # print (len(t_new_hyp_scores))
                #print (new_hyp_pos)
                if new_hyp_pos < len(t_new_hyp_scores):
                    # it's an ApplyRule or Reduce action
                    prev_hyp_id = t_prev_hyp_ids[new_hyp_pos]

                    prev_hyp = hypotheses[prev_hyp_id]

                    t_action_id = t_new_hyp_t_action_ids[new_hyp_pos]

                    if t_offset1 >= 0:
                        t_action_id = t_action_id + t_offset1 - 1
                    action = self.t_action_vocab.id2token[t_action_id]
                elif new_hyp_pos >= len(t_new_hyp_scores) and new_hyp_pos < len(t_new_hyp_scores) + len(nt_type_hyp_scores):

                    prev_hyp_id = nt_type_prev_hyp_ids[new_hyp_pos]

                    prev_hyp = hypotheses[prev_hyp_id]

                    t_action_id = nt_type_hyp_nt_action_ids[new_hyp_pos]

                    action = self.nt_action_vocab.id2token[t_action_id]
                else:
                    # it's a GenToken action
                    if nt_offset >= 0:

                        nt_action_id = (new_hyp_pos - len(t_new_hyp_scores)) % nt_offset
                        nt_action_id = nt_action_id.item()
                        k = (new_hyp_pos - len(t_new_hyp_scores)) // nt_offset
                    else:
                        nt_action_id = (new_hyp_pos - len(t_new_hyp_scores)) % len(self.nt_action_vocab)
                        nt_action_id = nt_action_id.item()
                        k = (new_hyp_pos - len(t_new_hyp_scores)) // len(self.nt_action_vocab)




                    # try:
                    # copy_info = gentoken_copy_infos[k]
                    prev_hyp_id = nt_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id]
                    action = self.nt_action_vocab.id2token[nt_action_id]


                new_hyp = prev_hyp.clone_and_apply_action(action)
                #print (new_hyp.actions)
                #print (new_hyp.tree)
                new_hyp.score = new_hyp_score
                #print (new_hyp.rule_completed)
                if new_hyp.rule_completed():
                    completed_hypotheses.append(new_hyp)
                else:
                    new_hypotheses.append(new_hyp)
                    live_hyp_ids.append(prev_hyp_id)

            if live_hyp_ids:
                hyp_states = [hyp_states[i] + [(h_t[i], cell_t[i])] for i in live_hyp_ids]
                h_tm1 = (h_t[live_hyp_ids], cell_t[live_hyp_ids])
                att_tm1 = att_t[live_hyp_ids]
                hypotheses = new_hypotheses
                hyp_scores = self.new_tensor([hyp.score for hyp in hypotheses])
                t += 1
            else:
                break

        if len(completed_hypotheses) == 0:
            """
            print ("======================= no parsed result !!! =================================")
            print(" ".join(example.src_sent))
            print (example.tgt_code_no_var_str)
            print("======================= no parsed result !!! =================================")
            """
            dummy_hyp = Hypothesis()
            completed_hypotheses.append(dummy_hyp)
        else:
            # completed_hypotheses = [hyp for hyp in completed_hypotheses if hyp.completed()]
            # todo: check the rank order
            completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses

    def save(self, path):
        dir_name = os.path.dirname(path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        params = {
            'state_dict': self.state_dict(),
            'args': self.args,
            'vocab': self.vocab
        }

        torch.save(params, path)

    @classmethod
    def load(cls, model_path, use_cuda=False):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        vocab = params['vocab']
        saved_args = params['args']
        # update saved args
        update_args(saved_args, init_arg_parser())
        saved_state = params['state_dict']
        saved_args.use_cuda = use_cuda
        parser = cls(vocab, saved_args)
        parser.load_state_dict(saved_state)

        if use_cuda: parser = parser.cuda()

        return parser


