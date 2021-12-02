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


@Registrable.register('independent')
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

        self.memory_data = {}  # stores exemplars class by class

        self.vocab = continuum.vocab

        self.t_nc_per_task = continuum.t_nc_per_task

        self.nt_nc_per_task = continuum.nt_nc_per_task

        # self.vocab, self.nc_per_task
        n_tasks = len(continuum.task_permutation)

        self.models = torch.nn.ModuleList()

        for _ in range(n_tasks):
            self.models.append(Seq2SeqModel(self.vocab, args))

        if args.uniform_init:
            print('uniformly initialize parameters [-%f, +%f]' % (args.uniform_init, args.uniform_init),
                  file=sys.stderr)
            nn_utils.uniform_init(-args.uniform_init, args.uniform_init, self.parameters())
        elif args.glorot_init:
            print('use glorot initialization', file=sys.stderr)
            nn_utils.glorot_init(self.parameters())

        print("load pre-trained word embedding (optional)")
        if args.glove_embed_path:
            for i in range(n_tasks):
                print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
                glove_embedding = GloveHelper(args.glove_embed_path, args.embed_size)
                glove_embedding.load_to(self.models[i].src_embed, self.vocab.source)

        self.evaluator = Registrable.by_name(args.evaluator)(args=args)

        if args.use_cuda and torch.cuda.is_available():
            for i in range(n_tasks):
                self.models[i].cuda()
        self.batch_size = args.batch_size
        self.use_cuda = args.use_cuda

        self.clip_grad = args.clip_grad

        optimizer_cls = eval('torch.optim.%s' % args.optimizer)

        self.optimizers = []

        for i in range(n_tasks):
            if args.optimizer == 'RMSprop':
                self.optimizers.append(optimizer_cls(self.models[i].parameters(), lr=args.lr, alpha=args.alpha))
            else:
                self.optimizers.append(optimizer_cls(self.models[i].parameters(), lr=args.lr))

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

    def decode(self, x, t):
        # nearest neighbor
        t_offset1, t_offset2 = self.compute_offsets(t)

        nt_offset = self.nt_nc_per_task[t]

        #current_task = self.observed_tasks[-1]

        decode_results = []
        for e in x:
            decode_results.append(
                self.models[t].beam_search(e, self.args.decode_max_time_step, t_offset1, t_offset2, nt_offset,
                                       beam_size=self.args.beam_size))
        return decode_results  # return 1-of-C code, ns x nc

    def reset_label_smoothing(self, t_offset1, t_offset2, nt_offset, task):
        if self.args.label_smoothing:
            self.models[task].t_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing,
                                                                         t_offset2 - t_offset1 + 1,
                                                                         ignore_indices=[0])

            self.models[task].nt_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing, nt_offset,
                                                                          ignore_indices=[0])

    def train_iter_process(self, batch_examples, report_loss, report_examples, t_offset1, t_offset2, nt_offset, task):
        # print (batch_examples)
        batch = Batch(batch_examples, self.vocab, use_cuda=self.use_cuda)

        nt_scores, t_scores = self.forward_with_offset(self.models[task], t_offset1, t_offset2, nt_offset,
                                                       batch)

        batch.t_action_idx_matrix[batch.t_action_idx_matrix.nonzero(as_tuple=True)] = batch.t_action_idx_matrix[
                                                                                          batch.t_action_idx_matrix.nonzero(
                                                                                              as_tuple=True)] - t_offset1 + 1

        t_action_loss = self.models[task].action_score(t_scores, batch, action_type='specific')

        nt_action_loss = self.models[task].action_score(nt_scores, batch, action_type='general')

        loss_val = torch.sum(t_action_loss).data.item() + torch.sum(nt_action_loss).data.item()

        t_action_loss = torch.mean(t_action_loss)

        nt_action_loss = torch.mean(nt_action_loss)

        report_loss += loss_val

        report_examples += len(batch_examples)

        loss = t_action_loss + nt_action_loss

        return report_loss, report_examples, loss

    def observe(self, task_data, t):

        # reset label smoothing

        train_set = task_data.train

        # print (type(train_set))

        if task_data.dev:
            dev_set = task_data.dev
        else:
            dev_set = Dataset(examples=[])

        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        n_memories = self.t_nc_per_task[t] * self.num_exemplars_per_class
        if t in self.memory_data:
            print ("T in memory data")
            self.memory_data[t].add(train_set.random_sample_batch_iter(n_memories))
        else:
            self.memory_data[t] = Dataset(train_set.random_sample_batch_iter(n_memories))

        print("setting model to training mode")
        self.models[t].train()

        # now compute the grad on the current minibatch

        # if self.memx is None:
        #    self.memx = x.data.clone()
        #    self.memy = y.data.clone()
        # else:
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

        while True:
            epoch += 1
            epoch_begin = time.time()

            for batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=True):
                self.reset_label_smoothing(t_offset1, t_offset2, nt_offset, t)
                # print(offset1)
                # print(offset2)
                train_iter += 1
                self.optimizers[t].zero_grad()

                report_loss, report_examples, loss = self.train_iter_process(batch_examples, report_loss, report_examples, t_offset1, t_offset2, nt_offset, t)

                loss.backward()

                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.models[t].parameters(), self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.models[t].parameters(), self.args.clip_grad)

                self.optimizers[t].step()


                if train_iter % self.args.log_every == 0:
                    log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)

                    print(log_str, file=sys.stderr)
                    report_loss = report_examples = 0.

            print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)


            if self.args.decay_lr_every_epoch and epoch > self.args.lr_decay_after_epoch:
                lr = self.optimizers[t].param_groups[0]['lr'] * self.args.lr_decay
                print('decay learning rate to %f' % lr, file=sys.stderr)

                # set new lr
                for param_group in self.optimizers[t].param_groups:
                    param_group['lr'] = lr


            if epoch == self.args.max_epoch:
                print('reached max epoch, stop!', file=sys.stderr)
                break

        model_file = self.args.save_to + '.{0}.bin'.format(str(t))
        print('save the current model ..', file=sys.stderr)
        print('save model to [%s]' % model_file, file=sys.stderr)
        self.models[t].save(model_file)
        # also save the optimizers' state
        torch.save(self.optimizers[t].state_dict(), self.args.save_to + '.{0}.optim.bin'.format(str(t)))

        # check whether this is the last minibatch of the current task
        # We assume only 1 epoch!
        # if self.examples_seen == self.samples_per_task:
        #    self.examples_seen = 0
        # get labels from previous task; we assume labels are consecutive

        # Reduce exemplar set by updating value of num. exemplars per class
        # self.num_exemplars = int(self.n_memories / (num_classes + len(self.mem_class_x.keys())))

    def forward_with_offset(self, model, t_offset1, t_offset2, nt_offset, batch):
        query_vectors = model(batch)
        nt_scores, t_scores = model.action_readout(query_vectors)
        pad_t_scores = t_scores[:, :, 0].unsqueeze(-1)
        task_t_scores = t_scores[:, :, t_offset1: t_offset2]
        t_action_scores = torch.cat((pad_t_scores, task_t_scores), -1)

        nt_scores = nt_scores[:, :, :nt_offset]

        return nt_scores, t_action_scores