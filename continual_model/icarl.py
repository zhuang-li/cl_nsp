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

from torch.nn.utils.rnn import pad_sequence

import evaluation
from common.registerable import Registrable
from components.dataset import Batch, Dataset
from continual_model.utils import read_domain_vocab, read_vocab_list
from grammar.action import GenNTAction, GenTAction
from grammar.consts import TYPE_SIGN, NT
from grammar.hypothesis import Hypothesis
from model import nn_utils
from continual_model.seq2seq_topdown import Seq2SeqModel
from model.utils import GloveHelper


@Registrable.register('icarl')
class Net(torch.nn.Module):

    # Re-implementation of
    # S.-A. Rebuffi, A. Kolesnikov, G. Sperl, and C. H. Lampert.
    # iCaRL: Incremental classifier and representation learning.
    # CVPR, 2017.

    def __init__(self,
                 args, continuum):
        super(Net, self).__init__()


        self.domains = args.domains

        self.num_exemplars_per_class = args.num_exemplars_per_class

        self.vocab = continuum.vocab

        self.t_nc_per_task = continuum.t_nc_per_task

        self.nt_nc_per_task = continuum.nt_nc_per_task

        self.reg = args.memory_strength

        self.args = args

        self.mem_class = {}  # stores exemplars class by class
        self.mem_class_y = {}



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
            glove_embedding.load_to(self.src_embed, self.vocab.source)

        self.evaluator = Registrable.by_name(args.evaluator)(args=args)
        if args.use_cuda and torch.cuda.is_available():
            self.model.cuda()
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda

        self.clip_grad = args.clip_grad

        optimizer_cls = eval('torch.optim.%s' % args.optimizer)

        if args.optimizer == 'RMSprop':
            self.optimizer = optimizer_cls(self.parameters(), lr=args.lr, alpha=args.alpha)
        else:
            self.optimizer = optimizer_cls(self.parameters(), lr=args.lr)

        # setup losses
        self.bce = torch.nn.CrossEntropyLoss()
        self.kl = torch.nn.KLDivLoss()  # for distillation
        self.lsm = torch.nn.LogSoftmax(dim=1)
        self.sm = torch.nn.Softmax(dim=1)

    def reset_label_smoothing(self, t_offset1, t_offset2, nt_offset):
        if self.args.label_smoothing:
            self.model.t_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing, t_offset2 - t_offset1 + 1,
                                                                 ignore_indices=[0])

            self.model.nt_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing, nt_offset,
                                                                 ignore_indices=[0])

    def compute_offsets(self, task):
        offset1 = offset2 = 1
        for i in range(task + 1):
            if i < task:
                offset1 += self.t_nc_per_task[i]

            offset2 += self.t_nc_per_task[i]

        return int(offset1), int(offset2)

    def icarl_beam_search(self, model, example, decode_max_time_step, t_offset1, t_offset2, t_means, nt_offset, nt_means, beam_size=5):
        """
        given a not-batched source, sentence perform beam search to find the n-best
        :param src_sent: List[word_id], encoded source sentence
        :return: list[list[word_id]] top-k predicted natural language sentence in the beam
        """
        # print ("============================")
        src_sents_var = nn_utils.to_input_variable([example.src_sent], model.src_vocab,
                                                   use_cuda=model.use_cuda, append_boundary_sym=True, mode='token')


        # TODO(junxian): check if src_sents_var(src_seq_length, embed_size) is ok
        src_encodings, (last_state, last_cell) = model.encode(src_sents_var, [src_sents_var.size(0)])
        # (1, query_len, hidden_size * 2)
        src_encodings = src_encodings.permute(1, 0, 2)
        src_encodings_att_linear = model.att_src_linear(src_encodings)
        dec_init_vec = model.init_decoder_state(last_state, last_cell)
        if model.lstm == 'lstm':
            h_tm1 = dec_init_vec[0], dec_init_vec[1]
        else:
            h_tm1 = dec_init_vec[0], dec_init_vec[1], \
                    model.new_tensor(1, model.hidden_size).zero_(), \
                    model.new_tensor(1, model.hidden_size).zero_()


        zero_action_embed = model.new_tensor(model.action_embed_size).zero_()

        att_tm1 = torch.zeros(1, model.action_embed_size, requires_grad=True)
        hyp_scores = torch.zeros(1, requires_grad=True)

        if model.use_cuda:
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
                x = model.new_tensor(1, model.decoder_lstm.input_size).zero_()
            else:
                actions_tm1 = [hyp.actions[-1] for hyp in hypotheses]
                #print (actions_tm1)
                a_tm1_embeds = []
                for a_tm1 in actions_tm1:
                    if a_tm1:
                        if isinstance(a_tm1, GenNTAction):
                            a_tm1_embed = model.action_nt_embed.weight[model.nt_action_vocab[a_tm1]]
                        elif isinstance(a_tm1, GenTAction):
                            a_tm1_embed = model.action_t_embed.weight[model.t_action_vocab[a_tm1]]
                        else:
                            raise ValueError
                    else:
                        a_tm1_embed = zero_action_embed
                    a_tm1_embeds.append(a_tm1_embed)
                a_tm1_embeds = torch.stack(a_tm1_embeds)

                inputs = [a_tm1_embeds]
                if model.use_att:
                    inputs.append(att_tm1)
                if model.lstm == 'lstm':
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

            (h_t, cell_t), att_t, att_weight = model.step(x, h_tm1, expanded_src_encodings,
                                                         expanded_src_encodings_att_linear,
                                                         src_sent_masks=None)

            #print (self.action_nt_embed.weight.size())
            nt_action_log_prob = torch.log_softmax(model.nt_readout(att_t)[:, :nt_offset], dim=-1)

            nt_action_log_prob[:,0] = -10e10

            t_nc_per_task = t_offset2 - t_offset1
            # Variable(batch_size, primitive_vocab_size)
            # ns data size
            # nd class size
            # means: nc_per_task, len(model.t_action_vocab)

            preds = model.t_readout(att_t).unsqueeze(1)

            dist = (means.expand(hyp_num, t_nc_per_task, len(model.t_action_vocab)) - preds.expand(hyp_num, t_nc_per_task, len(model.t_action_vocab))).norm(2, -1)


            t_action_log_prob = torch.log_softmax(-dist, dim=-1)

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
                        for t_id in model.t_action_vocab.lhs2rhsid[hyp.frontier_field.head]:
                            #print (t_action_log_prob.size())
                            # example : 1, 55
                            if t_id >= t_offset2 or t_id < t_offset1:
                                continue

                            t_id = t_id - t_offset1

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
                new_hyp_scores = model.new_tensor(t_new_hyp_scores)
            if nt_prev_hyp_ids:
                nt_new_hyp_scores = (
                            hyp_scores[nt_prev_hyp_ids].unsqueeze(1) + nt_action_log_prob[nt_prev_hyp_ids,
                                                                             :]).view(-1)
                #print (nt_new_hyp_scores.size())
                if new_hyp_scores is None:
                    new_hyp_scores = nt_new_hyp_scores
                else:
                    new_hyp_scores = torch.cat([new_hyp_scores, nt_new_hyp_scores])

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

                    t_action_id = t_action_id + t_offset1

                    action = model.t_action_vocab.id2token[t_action_id]
                else:
                    # it's a GenToken action
                    nt_action_id = (new_hyp_pos - len(t_new_hyp_scores)) % nt_offset
                    nt_action_id = nt_action_id.item()
                    k = (new_hyp_pos - len(t_new_hyp_scores)) // nt_offset
                    # try:
                    # copy_info = gentoken_copy_infos[k]
                    prev_hyp_id = nt_prev_hyp_ids[k]
                    prev_hyp = hypotheses[prev_hyp_id]
                    action = model.nt_action_vocab.id2token[nt_action_id]


                #print (action)
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
                hyp_scores = model.new_tensor([hyp.score for hyp in hypotheses])
                t += 1
            else:
                break

        if len(completed_hypotheses) == 0:
            dummy_hyp = Hypothesis()
            completed_hypotheses.append(dummy_hyp)
        else:
            completed_hypotheses.sort(key=lambda hyp: -hyp.score)

        return completed_hypotheses


    def decode(self, x, t):
        # nearest neighbor
        t_offset1, t_offset2 = self.compute_offsets(t)

        nt_offset = self.nt_nc_per_task[t]

        if t_offset1 not in self.mem_class.keys():
            # no exemplar, return a dummy hypothesis
            return [[Hypothesis()]]
        means = torch.ones(t_offset2 - t_offset1, len(self.model.t_action_vocab)) * float('inf')
        if self.use_cuda:
            means = means.cuda()

        for cc in range(t_offset1, t_offset2):
            exemplars_batch = Batch(self.mem_class[cc], self.vocab, use_cuda=self.use_cuda)

            means[cc -
                  t_offset1] = self.get_model_output(exemplars_batch, self.model, 1, len(self.model.t_action_vocab), nt_offset, cc).data.mean(0)

        decode_results = []
        for e in x:
            decode_results.append(self.icarl_beam_search(self.model, e, self.args.decode_max_time_step, t_offset1, t_offset2, means.unsqueeze(0), nt_offset, beam_size=self.args.beam_size))
        return decode_results  # return 1-of-C code, ns x nc

    def reset_learning_rate(self, lr):
        print('reset learning rate to %f' % lr, file=sys.stderr)

        # set new lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def observe(self, task_data, t):
        print("setting model to training mode")
        self.model.train()

        # self.reset_learning_rate(self.args.lr)

        train_set = task_data.train

        # print (type(train_set))

        if task_data.dev:
            dev_set = task_data.dev
        else:
            dev_set = Dataset(examples=[])

        #if self.memx is None:
        #    self.memx = x.data.clone()
        #    self.memy = y.data.clone()
        #else:
        #    self.memx = torch.cat((self.memx, x.data.clone()))
        #    self.memy = torch.cat((self.memy, y.data.clone()))

        print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
        print('vocab: %s' % repr(self.vocab), file=sys.stderr)

        epoch = train_iter = 0
        report_loss = report_examples = 0.
        history_dev_scores = []
        num_trial = patience = 0

        t_offset1, t_offset2 = self.compute_offsets(t)

        nt_offset = self.nt_nc_per_task[t]

        self.reset_label_smoothing(t_offset1, t_offset2, nt_offset)

        while True:
            epoch += 1
            epoch_begin = time.time()

            for batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=True):
                train_iter += 1
                self.optimizer.zero_grad()

                batch = Batch(batch_examples, self.vocab, use_cuda=self.use_cuda)

                nt_scores, t_scores = self.forward_with_offset(self.model, t_offset1, t_offset2, nt_offset,
                                                                                    batch)

                batch.t_action_idx_matrix[batch.t_action_idx_matrix.nonzero(as_tuple=True)] = batch.t_action_idx_matrix[batch.t_action_idx_matrix.nonzero(as_tuple=True)] - t_offset1 + 1
                loss = self.model.score_decoding_results(nt_scores, t_scores, batch)

                loss_val = torch.sum(loss).data.item()
                # print(loss.size())
                loss = torch.mean(loss)

                report_loss += loss_val

                report_examples += len(batch_examples)

                if t > 0:
                # distillation
                    for tt in range(t):
                        # first generate a minibatch with one example per class from
                        # previous tasks

                        p_t_offset1, p_t_offset2 = self.compute_offsets(tt)

                        p_nt_offset = self.nt_nc_per_task[tt]

                        indx_buffer=  []
                        exemplars = []

                        for cc in range(self.t_nc_per_task[tt]):
                            indx = random.randint(0, len(self.mem_class[cc + p_t_offset1]) - 1)
                            exemplars.append(self.mem_class[cc + p_t_offset1][indx])
                            indx_buffer.append(indx)
                            # print (self.mem_class_y[cc+p_t_offset1].size())
                            # print (len(self.mem_class[cc + p_t_offset1]))



                        exemplars_batch = Batch(exemplars, self.vocab, use_cuda=self.use_cuda)

                        # seq_len * batch size * vocab_size
                        # target_dist = torch.zeros(exemplars_batch.max_action_num, self.t_nc_per_task[tt], len(self.model.t_action_vocab))

                        # if self.use_cuda:
                            # target_dist = target_dist.cuda()
                        # print (target_dist.size())
                        target_dist_list = []
                        for idx, target_indx in enumerate(indx_buffer):
                            # print (exemplars_batch.max_action_num)
                            # print (target_indx)
                            # target_dist[idx] = \
                            #print ("======")
                            #print(exemplars_batch.max_action_num)
                            #print (self.mem_class_y[idx + p_t_offset1].size())
                            target_dist_list.append(self.mem_class_y[idx + p_t_offset1][:exemplars_batch.max_action_num, target_indx,:].clone().detach())

                        target_dist = pad_sequence(target_dist_list)

                        # print (target_dist.size())

                        exemplars_nt_scores, exemplars_t_vectors = self.forward_with_offset(self.model, p_t_offset1, p_t_offset2, p_nt_offset, exemplars_batch)

                        exemplars_pad_dist = target_dist[:,:,0].unsqueeze(-1)
                        exemplars_task_dist = target_dist[:,:,p_t_offset1: p_t_offset2]
                        exemplars_target_dist = torch.cat([exemplars_pad_dist, exemplars_task_dist], -1)

                        #self.model.score_decoding_results(torch.cat(exemplars_pad_vector, exemplars_t_vectors, 2), exemplars_batch)
                        #print (loss)
                        # Add distillation loss

                        loss += self.reg * self.kl(
                            self.lsm(exemplars_t_vectors),
                            self.sm(exemplars_target_dist)) * self.t_nc_per_task[tt]
                        #print (loss)
                # bprop and update
                loss.backward()

                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(), self.clip_grad)

                self.optimizer.step()

                if train_iter % self.args.log_every == 0:
                    log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)

                    print(log_str, file=sys.stderr)
                    report_loss = report_examples = 0.

            print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

            if self.args.save_all_models:
                model_file = self.args.save_to + '.iter%d.bin' % train_iter
                print('save model to [%s]' % model_file, file=sys.stderr)
                self.model.save(model_file)

            # perform validation
            if self.args.dev_file:
                if epoch % self.args.valid_every_epoch == 0:
                    print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                    eval_start = time.time()
                    decode_results = self.decode(dev_set.examples, t)
                    eval_results = self.evaluator.evaluate_dataset(dev_set.examples, decode_results, fast_mode=self.args.eval_top_pred_only)
                    #eval_results = evaluation.evaluate(dev_set.examples, model, self.evaluator, self.args, verbose=True, eval_top_pred_only=self.args.eval_top_pred_only)
                    dev_score = eval_results[self.evaluator.default_metric]

                    print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                        epoch, eval_results,
                        self.evaluator.default_metric,
                        dev_score,
                        time.time() - eval_start), file=sys.stderr)

                    is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                    history_dev_scores.append(dev_score)
            else:
                is_better = True

            if self.args.decay_lr_every_epoch and epoch > self.args.lr_decay_after_epoch:
                lr = self.optimizer.param_groups[0]['lr'] * self.args.lr_decay
                print('decay learning rate to %f' % lr, file=sys.stderr)

                # set new lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

            if is_better:

                patience = 0
                model_file = self.args.save_to + '.bin'
                print('save the current model ..', file=sys.stderr)
                print('save model to [%s]' % model_file, file=sys.stderr)
                self.model.save(model_file)
                # also save the optimizers' state
                torch.save(self.optimizer.state_dict(), self.args.save_to + '.optim.bin')

            elif patience < self.args.patience and epoch >= self.args.lr_decay_after_epoch:
                patience += 1
                print('hit patience %d' % patience, file=sys.stderr)

            if epoch == self.args.max_epoch:
                print('reached max epoch, stop!', file=sys.stderr)
                break

            if patience >= self.args.patience and epoch >= self.args.lr_decay_after_epoch:
                num_trial += 1
                print('hit #%d trial' % num_trial, file=sys.stderr)
                if num_trial == self.args.max_num_trial:
                    print('early stop!', file=sys.stderr)
                    break

                # decay lr, and restore from previously best checkpoint
                lr = self.optimizer.param_groups[0]['lr'] * self.args.lr_decay
                print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                # load model
                params = torch.load(self.args.save_to + '.sample.bin', map_location=lambda storage, loc: storage)
                self.model.load_state_dict(params['state_dict'])
                if self.args.use_cuda: self.model = self.model.cuda()

                # load optimizers
                if self.args.reset_optimizer:
                    print('reset optimizer', file=sys.stderr)
                    reset_optimizer_cls = eval('torch.optim.%s' % self.args.optimizer)  # FIXME: this is evil!
                    if self.args.optimizer == 'RMSprop':
                        self.optimizer = reset_optimizer_cls(self.parameters(), lr=self.args.lr, alpha=self.args.alpha)
                    else:
                        self.optimizer = reset_optimizer_cls(self.parameters(), lr=self.args.lr)
                    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
                else:
                    print('restore parameters of the optimizers', file=sys.stderr)
                    self.optimizer.load_state_dict(torch.load(self.args.save_to + '.sample.optim.bin'))

                # set new lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # reset patience
                patience = 0

        # check whether this is the last minibatch of the current task
        # We assume only 1 epoch!
        #if self.examples_seen == self.samples_per_task:
        #    self.examples_seen = 0
            # get labels from previous task; we assume labels are consecutive



        # Reduce exemplar set by updating value of num. exemplars per class
        # self.num_exemplars = int(self.n_memories / (num_classes + len(self.mem_class_x.keys())))
        for ll in range(self.t_nc_per_task[t]):
            lab = t_offset1 + ll

            lab_examples = train_set.class_examples[self.model.t_action_vocab.id2token[lab]]

            ll_batch = Batch(lab_examples, self.vocab, use_cuda=self.use_cuda)

            nd = self.t_nc_per_task[t] + 1 # number of class per task

            exemplars = []

            ntr = len(ll_batch) # exemplars number
            # used to keep track of which examples we have already used
            taken = torch.zeros(ntr)

            model_output = self.get_model_output(ll_batch, self.model, t_offset1, t_offset2, nt_offset, lab)

            # 1 * vocab_size

            mean_feature = model_output.data.clone().mean(0)

            for ee in range(self.num_exemplars_per_class):
                prev = torch.zeros(1, nd)
                if self.use_cuda:
                    prev = prev.cuda()
                if ee > 0:

                    prev_batch = Batch(train_set.class_examples[self.model.t_action_vocab.id2token[lab]][:ee], self.vocab, use_cuda=self.use_cuda)
                    # ee * vocab_size
                    prev = self.get_model_output(prev_batch, self.model, t_offset1, t_offset2, nt_offset, lab).data.clone().sum(0)



                cost = (mean_feature.expand(ntr, nd) - (model_output
                                                        + prev.expand(ntr, nd)) / (ee + 1)).norm(2, 1).squeeze()
                _, indx = cost.sort(0)
                winner = 0
                while winner < indx.size(0) and taken[indx[winner]] == 1:
                    winner += 1

                if winner < indx.size(0):
                    taken[indx[winner]] = 1
                    exemplars.append(lab_examples[indx[winner].item()])
                else:
                    exemplars = exemplars[:indx.size(0)]
                    #self.num_exemplars_per_class = indx.size(0)
                    break
            # update memory with exemplars
            self.mem_class[lab] = exemplars

        # recompute outputs for distillation purposes
        for cc in self.mem_class.keys():
            cc_exemplars = self.mem_class[cc]
            cc_exemplars.sort(key=lambda e: -len(e.src_sent))
            cc_exemplars_batch = Batch(cc_exemplars, self.vocab, use_cuda=self.use_cuda)
            cc_query_vectors = self.model(cc_exemplars_batch)
            cc_exemplars_nt_scores, cc_exemplars_t_scores = self.model.action_readout(cc_query_vectors)
            self.mem_class_y[cc] = cc_exemplars_t_scores.clone()


    def forward_with_offset(self, model, t_offset1, t_offset2, nt_offset, batch):
        query_vectors = model(batch)
        nt_scores, t_scores = model.action_readout(query_vectors)
        pad_t_scores = t_scores[:, :, 0].unsqueeze(-1)
        task_t_scores = t_scores[:, :, t_offset1: t_offset2]
        t_action_scores = torch.cat((pad_t_scores, task_t_scores), -1)

        nt_scores = nt_scores[: ,: ,:nt_offset]

        return nt_scores, t_action_scores

    def get_model_output(self, batch, model, t_offset1, t_offset2, nt_offset, lab):
        nt_scores, t_action_scores = self.forward_with_offset(model, t_offset1, t_offset2, nt_offset, batch)

        indxs_mask = (batch.t_action_idx_matrix != lab).unsqueeze(-1)


        indxs = (batch.t_action_idx_matrix == lab).nonzero(as_tuple=True)

        # print (batch.t_action_idx_matrix.size())

        elem_num_of_indxs = torch.unique(indxs[1], return_counts=True)[1]

        # for i in range(0, elem_num_of_indxs.size(0)):
            # print (elem_num_of_indxs[i])

        return t_action_scores.masked_fill_(indxs_mask.expand(t_action_scores.size()), 0).sum(0) / elem_num_of_indxs.view(
            elem_num_of_indxs.size(0), 1)