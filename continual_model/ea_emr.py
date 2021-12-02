# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time

import torch
import sys
import numpy as np
import random
import quadprog

import evaluation
from common.registerable import Registrable
from components.dataset import Batch, Dataset
from continual_model.seq2seq_topdown import Seq2SeqModel
from continual_model.utils import read_domain_vocab, read_vocab_list
from grammar.action import GenNTAction, GenTAction
from grammar.consts import TYPE_SIGN, NT
from grammar.hypothesis import Hypothesis
from model import nn_utils
from model.utils import GloveHelper
import torch.nn.functional as F

class AlignmentModel(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(AlignmentModel, self).__init__()
        # Calling Super Class's constructor
        self.linear = torch.nn.Linear(input_dim, output_dim)
        # nn.linear is defined in nn.Module

    def forward(self, x):
        # Here the forward pass is simply a linear function
        out = self.linear(x)
        return out

@Registrable.register('ea_emr')
class Net(torch.nn.Module):

    # Re-implementation of
    # S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert.
    # iCaRL: Incremental classifier and representation learning.
    # CVPR, 2017.

    def __init__(self,
                 args, continuum):
        super(Net, self).__init__()

        self.args = args

        self.domains = args.domains

        self.num_exemplars_per_class = args.num_exemplars_per_class

        self.num_exemplars_per_task = args.num_exemplars_per_task

        self.memory_data = {}  # stores exemplars class by class

        self.mem_class_x_t = {}  # stores exemplars class by class

        self.mem_class_x_nt = {}  # stores exemplars class by class

        self.mem_class_t = {}

        self.mem_class_nt = {}

        self.mem_align_weight = None

        self.vocab = continuum.vocab

        self.t_nc_per_task = continuum.t_nc_per_task

        self.nt_nc_per_task = continuum.nt_nc_per_task

        # self.vocab, self.nc_per_task

        self.align_t_embed = None

        self.align_nt_embed = None

        self.cos = torch.nn.CosineSimilarity(dim=-1)

        self.model = Seq2SeqModel(self.vocab, args)

        if args.uniform_init:
            print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init),
                  file=sys.stderr)
            nn_utils.uniform_init(-args.uniform_init, args.uniform_init, self.parameters())
        elif args.glorot_init:
            print('use glorot initialization', file=sys.stderr)
            nn_utils.glorot_init(self.parameters())

        print("load pre-trained word embedding (optional)")
        if args.glove_embed_path:
            print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
            glove_embedding = GloveHelper(args.glove_embed_path, args.embed_size)
            glove_embedding.load_to(self.model.src_embed, self.vocab.source)

        self.evaluator = Registrable.by_name(args.evaluator)(args=args)
        if args.use_cuda and torch.cuda.is_available():
            self.model.cuda()
        self.batch_size = args.batch_size

        self.use_cuda = args.use_cuda

        self.clip_grad = args.clip_grad

        optimizer_cls = eval('torch.optim.%s' % args.optimizer)

        self.align_t_optimizer = None

        self.align_nt_optimizer = None

        self.align_criterion = torch.nn.MSELoss()

        if args.optimizer == 'RMSprop':
            self.optimizer = optimizer_cls(self.model.parameters(), lr=args.lr, alpha=args.alpha)
        else:
            self.optimizer = optimizer_cls(self.model.parameters(), lr=args.lr)

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1


    def compute_offsets(self, task):
        offset1 = offset2 = 1
        for i in range(task + 1):
            if i < task:
                offset1 += self.t_nc_per_task[i]

            offset2 += self.t_nc_per_task[i]

        return int(offset1), int(offset2)


    def ea_emr_beam_search(self, example, decode_max_time_step, t_offset1 = -1, t_offset2 = -1, nt_offset = -1, beam_size=5, task = 0):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        # print ("============================")
        src_sents_var = nn_utils.to_input_variable([example.src_sent], self.model.src_vocab,
                                                   use_cuda=self.model.use_cuda, append_boundary_sym=True, mode='token')

        """
        print ("===============================preview===================================")
        print(" ".join(example.src_sent))
        print(example.tgt_code_no_var_str)
        print ("===============================preview====================================")
        """
        # TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
        src_encodings, (last_state, last_cell) = self.model.encode(src_sents_var, [src_sents_var.size(0)])
        # (1, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)
        src_encodings_att_linear = self.model.att_src_linear(src_encodings)
        dec_init_vec = self.model.init_decoder_state(last_state, last_cell)
        if self.model.lstm == 'lstm':
            h_tm1 = dec_init_vec[0], dec_init_vec[1]
        else:
            h_tm1 = dec_init_vec[0], dec_init_vec[1], \
                    self.model.new_tensor(1, self.model.hidden_size).zero_(), \
                    self.model.new_tensor(1, self.model.hidden_size).zero_()


        zero_action_embed = self.model.new_tensor(self.model.action_embed_size).zero_()

        att_tm1 = torch.zeros(1, self.model.action_embed_size, requires_grad=True)
        hyp_scores = torch.zeros(1, requires_grad=True)

        if self.model.use_cuda:
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
                x = self.model.new_tensor(1, self.model.decoder_lstm.input_size).zero_()
            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]
                #print (actions_tm1)
                a_tm1_embeds = []
                for a_tm1 in actions_tm1:
                    if a_tm1:
                        if isinstance(a_tm1, GenNTAction):
                            a_tm1_embed = self.model.action_nt_embed.weight[self.model.nt_action_vocab[a_tm1]]
                        elif isinstance(a_tm1, GenTAction):
                            a_tm1_embed = self.model.action_t_embed.weight[self.model.t_action_vocab[a_tm1]]
                        else:
                            raise ValueError
                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if self.model.use_att:
                    inputs.append(att_tm1)
                if self.model.lstm == 'lstm':
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
            (h_t, cell_t), att_t, att_weight = self.model.step(x, h_tm1, expanded_src_encodings,
                                                         expanded_src_encodings_att_linear,
                                                         src_sent_masks=None)

            #print (self.action_nt_embed.weight.size())

            # nt_embed_weight = model.action_nt_embed.weight

            # t_embed_weight = model.action_t_embed.weight

            # if self.align_embed is not None:
                # query_vectors = self.align_embed(query_vectors)

                # nt_embed_weight = self.align_embed(nt_embed_weight)

                # t_embed_weight = self.align_embed(t_embed_weight)

            # nt_scores = self.cos(query_vectors, nt_embed_weight)

            # t_scores = self.cos(query_vectors, t_embed_weight)

            nt_embed_weight = self.model.action_nt_embed.weight

            t_embed_weight = self.model.action_t_embed.weight

            nt_att_t = att_t

            t_att_t = att_t

            if self.align_t_embed is not None:

                t_att_t = self.align_t_embed(att_t)

                nt_att_t = self.align_nt_embed(att_t)

                nt_embed_weight = self.align_nt_embed(nt_embed_weight)
                t_embed_weight = self.align_t_embed(t_embed_weight)
            # print (nt_embed_weight.expand(att_t.size(0), nt_embed_weight.size(0), nt_embed_weight.size(1)).size())
            # att_t_expand = att_t.unsqueeze(1)
            # print (att_t_expand.expand(att_t.size(0), t_embed_weight.size(0), t_embed_weight.size(1)).size())
            # nt_action_scores = self.cos(att_t_expand.expand(att_t.size(0), nt_embed_weight.size(0), nt_embed_weight.size(1)), nt_embed_weight.expand(att_t.size(0), nt_embed_weight.size(0), nt_embed_weight.size(1)))

            # t_action_score = self.cos(att_t_expand.expand(att_t.size(0), t_embed_weight.size(0), t_embed_weight.size(1)), t_embed_weight.expand(att_t.size(0), t_embed_weight.size(0), t_embed_weight.size(1)))

            nt_action_scores = F.linear(nt_att_t, nt_embed_weight)

            t_action_score = F.linear(t_att_t, t_embed_weight)


            if nt_offset >= 0:
                nt_action_log_prob = torch.log_softmax(nt_action_scores[:, :nt_offset], dim=-1)
            else:
                nt_action_log_prob = torch.log_softmax(nt_action_scores, dim=-1)

            nt_action_log_prob[:,0] = -100

            # Variable(batch_size, primitive_vocab_size)

            #t_action_score = self.model.t_readout(att_t)


            if t_offset1 >= 0:
                # print (t_action_score.size())
                pad_t_action_score = t_action_score[:, 0].unsqueeze(-1)
                #print (pad_t_action_score.size())
                task_t_action_score = t_action_score[:, t_offset1: t_offset2]

                # print (pad_t_action_score.size())
                # print (task_t_action_score.size())

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
            #print (len(hypotheses))
            if t==0:
                nt_prev_hyp_ids.append(0)
            else:
                for hyp_id, hyp in enumerate(hypotheses):
                    # generate new continuations
                    if hyp.frontier_field.head.startswith(TYPE_SIGN):
                        #print (self.t_action_vocab.lhs2rhsid[hyp.frontier_field.head])
                        for t_id in self.model.t_action_vocab.lhs2rhsid[hyp.frontier_field.head]:
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
                    else:
                        raise ValueError



            new_hyp_scores = None
            if t_new_hyp_scores:
                new_hyp_scores = self.model.new_tensor(t_new_hyp_scores)
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
                    # print (t_action_id)
                    if t_offset1 >= 0:
                        t_action_id = t_action_id + t_offset1 - 1
                    # print (t_action_id)
                    action = self.model.t_action_vocab.id2token[t_action_id]
                else:
                    # it's a GenToken action
                    nt_action_id = (new_hyp_pos - len(t_new_hyp_scores)) % nt_offset
                    nt_action_id = nt_action_id.item()
                    k = (new_hyp_pos - len(t_new_hyp_scores)) // nt_offset
                    # try:
                    # copy_info = gentoken_copy_infos[k]
                    prev_hyp_id = nt_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id]
                    action = self.model.nt_action_vocab.id2token[nt_action_id]


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
                hyp_scores = self.model.new_tensor([hyp.score for hyp in hypotheses])
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



    def decode(self, x, t):
        # nearest neighbor
        t_offset1, t_offset2 = self.compute_offsets(t)

        nt_offset = self.nt_nc_per_task[t]

        # print (t_offset1)
        # print (t_offset2)
        temp_align_weight = None

        temp_align_embed = None
        """
        if len(self.observed_tasks) > 1:
            if t == self.observed_tasks[-1]:
                if self.mem_align_weight is not None:
                    print ("check when enter this before", t)
                    temp_align_weight = self.align_embed.linear.weight.clone()

                    self.align_embed.linear.weight.data.copy_(self.mem_align_weight.data)
                else:
                    temp_align_embed = self.align_embed
                    self.align_embed = None
       """
        decode_results = []
        for e in x:
            decode_results.append(
                self.ea_emr_beam_search(e, self.args.decode_max_time_step, t_offset1, t_offset2, nt_offset,
                                       beam_size=self.args.beam_size, task=t))

        """
        if len(self.observed_tasks) > 1:
            if t == self.observed_tasks[-1]:
                if self.mem_align_weight is not None:
                    print ("check when enter this after", t)
                    self.align_embed.linear.weight.data.copy_(temp_align_weight.data)
                else:
                    self.align_embed = temp_align_embed
        """
        return decode_results  # return 1-of-C code, ns x nc

    def reset_label_smoothing(self, t_offset1, t_offset2, nt_offset):
        if self.args.label_smoothing:
            self.model.t_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing,
                                                                         t_offset2 - t_offset1 + 1,
                                                                         ignore_indices=[0], use_cuda=self.args.use_cuda)

            self.model.nt_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing, nt_offset,
                                                                          ignore_indices=[0], use_cuda=self.args.use_cuda)

    def train_iter_process(self, batch_examples, report_loss, report_examples, t_offset1, t_offset2, nt_offset):
        # print (batch_examples)
        batch = Batch(batch_examples, self.vocab, use_cuda=self.use_cuda)

        nt_scores, t_scores = self.forward_with_offset(self.model, t_offset1, t_offset2, nt_offset,
                                                       batch)

        batch.t_action_idx_matrix[batch.t_action_idx_matrix.nonzero(as_tuple=True)] = batch.t_action_idx_matrix[
                                                                                          batch.t_action_idx_matrix.nonzero(
                                                                                              as_tuple=True)] - t_offset1 + 1

        t_action_loss = self.model.action_score(t_scores, batch, action_type='specific')

        nt_action_loss = self.model.action_score(nt_scores, batch, action_type='general')

        loss_val = torch.sum(t_action_loss).data.item() + torch.sum(nt_action_loss).data.item()

        t_action_loss = torch.mean(t_action_loss)

        nt_action_loss = torch.mean(nt_action_loss)

        report_loss += loss_val

        report_examples += len(batch_examples)

        loss = t_action_loss + nt_action_loss

        return report_loss, report_examples, loss

    def update_alignment_model(self):

        epoch = 0
        replay_report_loss = 0.

        while True:
            epoch += 1
            epoch_begin = time.time()

            for tt in range(len(self.observed_tasks)):

                self.align_t_optimizer.zero_grad()

                self.align_nt_optimizer.zero_grad()
                # fwd/bwd on the examples in the memory
                past_task = self.observed_tasks[tt]

                replay_batch_examples = self.memory_data[past_task].examples

                stored_seq_embed_t = self.mem_class_x_t[past_task].cuda()

                stored_seq_embed_nt = self.mem_class_x_nt[past_task].cuda()

                stored_t_action_embed = self.mem_class_t[past_task].cuda()

                stored_nt_action_embed = self.mem_class_nt[past_task].cuda()

                # replay_report_loss, replay_report_examples, replay_loss = self.train_iter_process(replay_batch_examples, replay_report_loss, replay_report_examples, p_t_offset1, p_t_offset2, p_nt_offset)

                replay_batch = Batch(replay_batch_examples, self.vocab, use_cuda=self.use_cuda)

                current_seq_embed_t = self.get_seq_embed(self.model, replay_batch, do_align=True,
                                                                    detach=False,action_type='t')

                current_seq_embed_nt = self.get_seq_embed(self.model, replay_batch, do_align=True,
                                                                    detach=False,action_type='nt')

                current_t_action_embed = self.get_action_embed(self.model, do_align=True, detach=False,action_type='t')

                current_nt_action_embed = self.get_action_embed(self.model, do_align=True, detach=False,action_type='nt')

                #if tt < len(self.observed_tasks) - 2:
                #    replay_loss.backward(retain_graph=True)
                #else:
                # replay_loss.backward()

                t_action_loss = self.align_criterion(current_t_action_embed, stored_t_action_embed)

                nt_action_loss = self.align_criterion(current_nt_action_embed, stored_nt_action_embed)

                t_seq_loss = self.align_criterion(current_seq_embed_t, stored_seq_embed_t)

                nt_seq_loss = self.align_criterion(current_seq_embed_nt, stored_seq_embed_nt)

                loss = t_action_loss + nt_action_loss + t_seq_loss + nt_seq_loss

                loss.backward()

                replay_report_loss += loss.item()

                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.align_t_embed.parameters(), self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.align_t_embed.parameters(), self.args.clip_grad)

                self.align_t_optimizer.step()

                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.align_nt_embed.parameters(), self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.align_nt_embed.parameters(), self.args.clip_grad)

                self.align_nt_optimizer.step()

                del stored_seq_embed_t

                del stored_seq_embed_nt

                del stored_t_action_embed

                del stored_nt_action_embed

                torch.cuda.empty_cache()

            if epoch % 10 == 0:
                log_str = '[Epoch %d] align model loss=%.5f' % (epoch, replay_report_loss / 10)

                print(log_str, file=sys.stderr)
                replay_report_loss = 0.

            print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)


            if epoch == self.args.max_align_epoch:
                print('reached max epoch, stop!', file=sys.stderr)
                break

        for tt in range(len(self.observed_tasks)):

            past_task = self.observed_tasks[tt]

            assert past_task == tt

            stored_examples = self.memory_data[past_task].examples

            stored_batch = Batch(stored_examples, self.vocab, use_cuda=self.use_cuda)

            self.mem_class_x_t[past_task] = self.get_seq_embed(self.model, stored_batch, do_align=True,
                                                                detach=True, action_type='t').cpu()

            self.mem_class_x_nt[past_task] = self.get_seq_embed(self.model, stored_batch, do_align=True,
                                                                detach=True, action_type='nt').cpu()

            self.mem_class_t[past_task] = self.get_action_embed(self.model, do_align=True, detach=True, action_type='t').cpu()

            self.mem_class_nt[past_task] = self.get_action_embed(self.model, do_align=True, detach=True, action_type='nt').cpu()

    def reset_learning_rate(self, lr):
        print('reset learning rate to %f' % lr, file=sys.stderr)

        # set new lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def observe(self, task_data, t):

        # if self.align_embed is not None:
        #    self.mem_align_weight = self.align_embed.linear.weight.clone()

        # self.reset_learning_rate(self.args.lr)

        self.forward_train(task_data, t)

        if len(self.observed_tasks) > 1:
            print ("task=========================")
            print (t)
            if self.align_t_embed is None:

                self.align_t_embed = AlignmentModel(self.model.action_embed_size, self.model.action_embed_size)

                self.align_nt_embed = AlignmentModel(self.model.action_embed_size, self.model.action_embed_size)

                if self.args.use_cuda and torch.cuda.is_available():

                    self.align_t_embed.cuda()

                    self.align_nt_embed.cuda()

                optimizer_cls = eval('torch.optim.%s' % self.args.optimizer)

                if self.args.optimizer == 'RMSprop':
                    self.align_t_optimizer = optimizer_cls(self.align_t_embed.parameters(), lr=self.args.align_lr, alpha=self.args.alpha)
                else:
                    self.align_t_optimizer = optimizer_cls(self.align_t_embed.parameters(), lr=self.args.align_lr)

                if self.args.optimizer == 'RMSprop':
                    self.align_nt_optimizer = optimizer_cls(self.align_nt_embed.parameters(), lr=self.args.align_lr, alpha=self.args.alpha)
                else:
                    self.align_nt_optimizer = optimizer_cls(self.align_nt_embed.parameters(), lr=self.args.align_lr)

            self.update_alignment_model()

        model_file = self.args.save_to + '.bin'
        print('save the current model ..', file=sys.stderr)
        print('save model to [%s]' % model_file, file=sys.stderr)
        self.model.save(model_file)


    def forward_train(self, task_data, t):

        # reset label smoothing

        train_set = task_data.train

        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        n_memories = int(len(train_set) * self.args.num_exemplars_ratio)

        print ("The memory size is {0}".format(n_memories))

        if t in self.memory_data:
            print ("T in memory data")
            self.memory_data[t].add(train_set.random_sample_batch_iter(n_memories))
        else:
            self.memory_data[t] = Dataset(train_set.random_sample_batch_iter(n_memories))

        print("setting model to training mode")
        self.model.train()

        # now compute the grad on the current minibatch

        # if self.memx is None:
        #    self.memx = x.data.clone()
        #    self.memy = y.data.clone()
        # else:
        #    self.memx = torch.cat((self.memx, x.data.clone()))
        #    self.memy = torch.cat((self.memy, y.data.clone()))

        print('begin training, %d training examples' % (len(train_set)), file=sys.stderr)
        print('vocab: %s' % repr(self.vocab), file=sys.stderr)

        epoch = train_iter = 0
        report_loss = report_examples = replay_report_loss = replay_report_examples = 0.

        t_offset1, t_offset2 = self.compute_offsets(t)

        nt_offset = self.nt_nc_per_task[t]

        while True:
            epoch += 1
            epoch_begin = time.time()

            for batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=True):
                self.reset_label_smoothing(t_offset1, t_offset2, nt_offset)
                # print(offset1)
                # print(offset2)
                train_iter += 1
                self.optimizer.zero_grad()
                if self.align_t_embed is not None:
                    self.align_t_embed.zero_grad()
                    self.align_nt_embed.zero_grad()

                report_loss, report_examples, loss = self.train_iter_process(batch_examples, report_loss, report_examples, t_offset1, t_offset2, nt_offset)

                #loss.backward()
                """
                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_grad)
                
                self.optimizer.step()
                """
                # replay batch
                # sample_size = int(self.args.batch_size / len(self.observed_tasks) - 1)
                if len(self.observed_tasks) > 1:
                    for tt in range(len(self.observed_tasks) - 1):
                        self.optimizer.zero_grad()
                        # fwd/bwd on the examples in the memory
                        past_task = self.observed_tasks[tt]

                        p_t_offset1, p_t_offset2 = self.compute_offsets(past_task)

                        p_nt_offset = self.nt_nc_per_task[past_task]

                        self.reset_label_smoothing(p_t_offset1, p_t_offset2, p_nt_offset)

                        replay_batch_examples = self.memory_data[past_task].examples

                        replay_report_loss, replay_report_examples, replay_loss = self.train_iter_process(replay_batch_examples, replay_report_loss, replay_report_examples, p_t_offset1, p_t_offset2, p_nt_offset)

                        #if tt < len(self.observed_tasks) - 2:
                        #    replay_loss.backward(retain_graph=True)
                        #else:
                        #replay_loss.backward()
                        loss += replay_loss
                loss.backward()
                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_grad)

                self.optimizer.step()

                if train_iter % self.args.log_every == 0:
                    log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                    if len(self.observed_tasks) > 1:
                        log_str += '[Iter %d] replay encoder loss=%.5f' % (
                            train_iter, replay_report_loss / replay_report_examples)

                    print(log_str, file=sys.stderr)
                    report_loss = report_examples = replay_report_loss = replay_report_examples = 0.

            print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)


            if self.args.decay_lr_every_epoch and epoch > self.args.lr_decay_after_epoch:
                lr = self.optimizer.param_groups[0]['lr'] * self.args.lr_decay
                print('decay learning rate to %f' % lr, file=sys.stderr)

                # set new lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr


            if epoch == self.args.max_epoch:
                print('reached max epoch, stop!', file=sys.stderr)
                break

        current_task = self.observed_tasks[-1]

        assert current_task == t

        current_stored_examples = self.memory_data[current_task].examples

        current_stored_batch = Batch(current_stored_examples, self.vocab, use_cuda=self.use_cuda)

        self.mem_class_x_t[current_task] = self.get_seq_embed(self.model, current_stored_batch, do_align=True, detach=True, action_type='t').cpu()

        self.mem_class_x_nt[current_task] = self.get_seq_embed(self.model, current_stored_batch, do_align=True, detach=True, action_type='nt').cpu()

        self.mem_class_t[current_task] = self.get_action_embed(self.model, do_align=True, detach=True,action_type='t').cpu()

        self.mem_class_nt[current_task] = self.get_action_embed(self.model, do_align=True, detach=True,
                                                               action_type='nt').cpu()


    def get_seq_embed(self, model, batch, do_align=False, detach=False, action_type='t'):

        query_vectors = model(batch)

        if self.align_t_embed is not None and do_align:

            assert len(self.observed_tasks) > 1
            if action_type == 't':
                query_vectors = self.align_t_embed(query_vectors)
            else:
                query_vectors = self.align_nt_embed(query_vectors)
        if detach:
            return query_vectors.detach()
        else:
            return query_vectors

    def get_action_embed(self, model, do_align=False, detach=False, action_type='t'):

        if action_type == 't':
            embed_weight = model.action_t_embed.weight
        else:
            embed_weight = model.action_nt_embed.weight

        if self.align_t_embed is not None and do_align:

            assert len(self.observed_tasks) > 1
            if action_type == 't':
                embed_weight = self.align_t_embed(embed_weight)
            else:
                embed_weight = self.align_nt_embed(embed_weight)
        if detach:
            return embed_weight.detach()
        else:
            return embed_weight

        # check whether this is the last minibatch of the current task
        # We assume only 1 epoch!
        # if self.examples_seen == self.samples_per_task:
        #    self.examples_seen = 0
        # get labels from previous task; we assume labels are consecutive

        # Reduce exemplar set by updating value of num. exemplars per class
        # self.num_exemplars = int(self.n_memories / (num_classes + len(self.mem_class_x.keys())))

    def forward_with_offset(self, model, t_offset1, t_offset2, nt_offset, batch):

        query_vectors = model(batch)

        nt_embed_weight = model.action_nt_embed.weight

        t_embed_weight = model.action_t_embed.weight

        nt_query_vectors = query_vectors

        t_query_vectors = query_vectors

        if self.align_t_embed is not None:
            t_query_vectors = self.align_t_embed(query_vectors)

            nt_query_vectors = self.align_nt_embed(query_vectors)

            nt_embed_weight = self.align_nt_embed(nt_embed_weight)

            t_embed_weight = self.align_t_embed(t_embed_weight)

        #print (query_vectors.size())

        # query_vectors_expand = query_vectors.unsqueeze(2)

        # nt_embed_weight_expand = nt_embed_weight.unsqueeze(0).unsqueeze(0)

        # t_embed_weight_expand = t_embed_weight.unsqueeze(0).unsqueeze(0)

        # nt_scores = self.cos(query_vectors_expand.expand(query_vectors.size(0),query_vectors.size(1), nt_embed_weight.size(0), nt_embed_weight.size(1)), nt_embed_weight_expand.expand(query_vectors.size(0), query_vectors.size(1), nt_embed_weight.size(0), nt_embed_weight.size(1)))

        # t_scores = self.cos(query_vectors_expand.expand(query_vectors.size(0),query_vectors.size(1), t_embed_weight.size(0), t_embed_weight.size(1)), t_embed_weight_expand.expand(query_vectors.size(0),query_vectors.size(1), t_embed_weight.size(0), t_embed_weight.size(1)))
        # todo: debug
        nt_scores = F.linear(nt_query_vectors, nt_embed_weight)

        t_scores = F.linear(t_query_vectors, t_embed_weight)

        # nt_scores = self.cos(query_vectors, nt_embed_weight)

        # t_scores = self.cos(query_vectors, t_embed_weight)

        pad_t_scores = t_scores[:, :, 0].unsqueeze(-1)

        task_t_scores = t_scores[:, :, t_offset1: t_offset2]

        t_action_scores = torch.cat((pad_t_scores, task_t_scores), -1)

        nt_scores = nt_scores[:, :, :nt_offset]

        return nt_scores, t_action_scores