# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import sys
import time
from collections import Counter

import miosqp
import numpy
import numpy as np
import quadprog
import scipy.sparse as spa
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch import nn
from torch.distributions import Categorical

from common.registerable import Registrable
from components.dataset import Batch, Dataset, reweight_t_data, sample_balance_data
from continual_model.seq2seq_topdown import Seq2SeqModel
from model import nn_utils
from model.utils import GloveHelper

miosqp_settings = {
    # integer feasibility tolerance
    'eps_int_feas': 1e-03,
    # maximum number of iterations
    'max_iter_bb': 1000,
    # tree exploration rule
    #   [0] depth first
    #   [1] two-phase: depth first until first incumbent and then  best bound
    'tree_explor_rule': 1,
    # branching rule
    #   [0] max fractional part
    'branching_rule': 0,
    'verbose': False,
    'print_interval': 1}

osqp_settings = {'eps_abs': 2e-03,
                 'eps_rel': 2e-03,
                 'eps_prim_inf': 2e-04,
                 'verbose': False}

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return numpy.allclose(a, a.T, rtol=rtol, atol=atol)

def store_grad(pp, grads, grad_dims, tid):
    """
        This stores parameter gradients of past tasks.
        pp: parameters
        grads: gradients
        grad_dims: list with number of parameters per layers
        tid: task id
    """
    # store the gradients
    grads[:, tid].fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en, tid].copy_(param.grad.data.view(-1))
        cnt += 1


def fill_diagonal(source_array, diagonal):
    copy = source_array.copy()
    np.fill_diagonal(copy, diagonal)
    return copy

def cosine_similarity_selector_IQP_Exact(G, solver, nb_selected, sample_method='normal'):
    """
    Integer programming
    """
    print ("Begin to solve !!! ")
    if sample_method == 'topk_max':
        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        values, inds = torch.topk(max_G, nb_selected)
        return inds
    elif sample_method == 'topk_sum':
        sum_G = torch.sum(G, dim=1)
        values, inds = torch.topk(sum_G, nb_selected)
        return inds
    elif sample_method == 'ne_topk_max':
        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        values, inds = torch.topk(-max_G, nb_selected)
        return inds
    elif sample_method == 'ne_topk_min':
        min_G, min_G_inds = torch.min(G, dim=1)
        values, inds = torch.topk(-min_G, nb_selected)
        return inds
    elif sample_method == 'ne_topk_sum':
        sum_G = torch.sum(G, dim=1)
        values, inds = torch.topk(-sum_G, nb_selected)
        return inds
    else:
        t = G.size(0)

        inds = torch.arange(0, t)
        #print (G)
        G = G.cpu().double().numpy()
        print (check_symmetric(G))
        # print (G)

        if sample_method == 'sum':
            print ("Using greedy method")
            a = np.sum(G, axis=1)
        elif sample_method == 'max':
            a = np.max(fill_diagonal(G, -np.inf), axis=1)
        elif sample_method == 'ne_sum':
            a = -1*np.sum(G, axis=1)
        elif sample_method == 'ne_max':
            a = -1*np.max(fill_diagonal(G, -np.inf), axis=1)
        else:
            a = np.zeros(t)
        # a=np.ones(t)*-1

        # a=((w1-torch.min(w1))/(torch.max(w1)-torch.min(w1))).squeeze().double().numpy()*-0.01
        C = np.ones((t, 1))
        h = np.zeros(1) + nb_selected
        C2 = np.eye(t)

        hlower = np.zeros(t)
        hupper = np.ones(t)
        idx = np.arange(t)

        #################
        C = np.concatenate((C2, C), axis=1)
        C = np.transpose(C)
        h_final_lower = np.concatenate((hlower, h), axis=0)
        h_final_upper = np.concatenate((hupper, h), axis=0)
        #################
        csc_G = spa.csc_matrix(G)

        C = spa.csc_matrix(C)

        #print (C)
        #print (h_final_lower)
        #print (h_final_upper)

        solver.setup(csc_G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper, miosqp_settings, osqp_settings)

        # prob = mpbpy.QuadprogProblem(G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper)

        if sample_method == 'init_max':
            max_G = -np.max(fill_diagonal(G, -np.inf), axis=1)
            max_index = max_G.argsort()[-nb_selected:][::-1]
            x0 = np.zeros(t)
            x0[max_index] = 1
            # prob = mpbpy.QuadprogProblem(G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper, x0=x0)
            solver.set_x0(x0)
        elif sample_method == 'init_min':
            min_G = -np.min(G, axis=1)
            min_index = min_G.argsort()[-nb_selected:][::-1]
            x0 = np.zeros(t)
            x0[min_index] = 1
            # prob = mpbpy.QuadprogProblem(G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper, x0=x0)
            solver.set_x0(x0)

        elif sample_method == 'init_sum':
            sum_G = -np.sum(G, axis=1)
            sum_index = sum_G.argsort()[-nb_selected:][::-1]
            x0 = np.zeros(t)
            x0[sum_index] = 1
            # prob = mpbpy.QuadprogProblem(G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper, x0=x0)
            solver.set_x0(x0)
        # else:
            # inderr = torch.randperm(t).tolist()
            # x0 = np.zeros(t)
            # x0[inderr[:nb_selected]] = 1
            # solver.set_x0(x0)

        # results = prob.solve(solver=mpbpy.GUROBI, verbose=False, Threads=6)

        results = solver.solve()
        print("STATUS", results.status)
        coeffiecents_np = results.x
        # print (coeffiecents_np)
        coeffiecents = torch.nonzero(torch.Tensor(coeffiecents_np), as_tuple=False)
        print("number of selected items is", sum(coeffiecents_np))
        if "Infeasible" in results.status:
            return inds

        # print  (coeffiecents)

        # print (inds[coeffiecents.squeeze()])
        return inds[coeffiecents.squeeze()]

def get_G(x1, x2):
    # print ("get G begin")

    w1 = x1.norm(p=2, dim=1, keepdim=True)

    # inds = torch.nonzero(torch.gt(w1, 0.0),as_tuple=False)[:, 0]

    # assert inds.size(0) == w1.size(0)
    # print (inds.size(0))
    # if inds.size(0) < nb_selected:
        # print("WARNING GRADIENTS ARE TOO SMALL!!!!!!!!")
    # inds = torch.arange(0, x1.size(0))
    # w1 = w1[inds]
    # x1 = x1[inds]
    # x2 = x1 if x2 is None else x2
    w2 = x2.norm(p=2, dim=1, keepdim=True)
    # print(x1.size())
    # print (x2.size())
    # print(w1.size())
    # print (w2.size())
    G = torch.mm(x1, x2.t()) / (w1 * w2.t())  # .clamp(min=eps)
    # print ("get G end")
    return G

def get_grad_vector(pp, grad_dims):
    """
     gather the gradients in one vector
    """
    get_grad_ti = time.time()
    grads = torch.cuda.FloatTensor(sum(grad_dims))
    grads.fill_(0.0)
    cnt = 0


    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])

            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    # print ("Gradient vector time is", time.time() - get_grad_ti)
    return grads


def add_memory_grad(pp, mem_grads, grad_dims, idx):
    """
        This stores the gradient of a new memory and compute the dot product with the previously stored memories.
        pp: parameters
        mem_grads: gradients of previous memories
        grad_dims: list with number of parameters per layers
    """

    # gather the gradient of the new memory
    grads = get_grad_vector(pp, grad_dims)
    add_mem_time = time.time()

    # mem_grads = torch.Tensor(batch_size, sum(grad_dims))


    # grads = grads.unsqueeze(dim=0)

    mem_grads[idx].copy_(grads.data.view(-1))
    #print("Add memory time is", time.time() - add_mem_time)
    return mem_grads


def overwrite_grad(pp, newgrad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            this_grad = newgrad[beg: en].contiguous().view(
                param.grad.data.size())
            param.grad.data.copy_(this_grad)
        cnt += 1


def project2cone2(gradient, memories, margin=0.5, eps=1e-3):
    """
        Solves the GEM dual QP described in the paper given a proposed
        gradient "gradient", and a memory of task gradients "memories".
        Overwrites "gradient" with the final projected update.
        input:  gradient, p-vector
        input:  memories, (t * p)-vector
        output: x, p-vector
    """

    memories_np = memories.cpu().t().double().numpy()
    gradient_np = gradient.cpu().contiguous().view(-1).double().numpy()
    t = memories_np.shape[0]
    P = np.dot(memories_np, memories_np.transpose())
    P = 0.5 * (P + P.transpose()) + np.eye(t) * eps
    q = np.dot(memories_np, gradient_np) * -1
    G = np.eye(t)
    h = np.zeros(t) + margin
    v = quadprog.solve_qp(P, q, G, h)[0]
    x = np.dot(v, memories_np) + gradient_np
    gradient.copy_(torch.Tensor(x).view(-1, 1))


@Registrable.register('emr_wo_t')
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

        self.n_memories = 0

        self.memory_data = []  # stores exemplars class by task

        self.memory_buffer = []

        self.vocab = continuum.vocab

        self.t_nc_per_task = continuum.t_nc_per_task

        self.nt_nc_per_task = continuum.nt_nc_per_task

        self.mem_grads = None

        self.new_mem_grads = None

        self.mem_cnt = 0

        m = miosqp.MIOSQP()
        self.solver = m
        # self.vocab, self.nc_per_task

        self.old_mem_grads = None

        self.mem_cnt = 0

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

        optimizer_cls = eval('torch.optim.%s' % self.args.optimizer)

        if self.args.optimizer == 'RMSprop':
            self.optimizer = optimizer_cls(self.parameters(), lr=self.args.lr, alpha=self.args.alpha)
        else:
            self.optimizer = optimizer_cls(self.parameters(), lr=self.args.lr)

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1

        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())


        self.grads = torch.Tensor(sum(self.grad_dims), 2)
        if args.use_cuda:
            self.grads = self.grads.cuda()



    def decode(self, x, t):
        # nearest neighbor

        decode_results = []
        for e in x:
            decode_results.append(
                self.model.beam_search(e, self.args.decode_max_time_step, -1, -1, -1,
                                       beam_size=self.args.beam_size))
        return decode_results  # return 1-of-C code, ns x nc


    def train_iter_process(self, model, batch_examples, report_loss, report_examples):
        # print (batch_examples)
        batch = Batch(batch_examples, self.vocab, use_cuda=self.use_cuda)

        nt_scores, t_scores = self.forward(model, batch)

        t_action_loss = model.action_score(t_scores, batch, action_type='specific')

        nt_action_loss = model.action_score(nt_scores, batch, action_type='general')

        loss_val = torch.sum(t_action_loss).data.item() + torch.sum(nt_action_loss).data.item()

        non_avg_loss = t_action_loss + nt_action_loss

        t_action_loss = torch.mean(t_action_loss)

        nt_action_loss = torch.mean(nt_action_loss)

        report_loss += loss_val

        report_examples += len(batch_examples)

        loss = t_action_loss + nt_action_loss

        return report_loss, report_examples, loss, non_avg_loss


    def reset_learning_rate(self, lr):
        print('reset learning rate to %f' % lr, file=sys.stderr)

        # set new lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def observe(self, task_data, t):

        # self.reset_learning_rate(self.args.lr)
        # reset label smoothing

        # self.reset_learning_rate(self.args.lr)

        train_set = task_data.train

        # print (type(train_set))

        if task_data.dev:
            dev_set = task_data.dev
        else:
            dev_set = Dataset(examples=[])

        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t
        # print (self.args.num_exemplars)
        self.n_memories = int(self.args.num_exemplars)
            #int(len(train_set) * self.args.num_exemplars_ratio)

        # reset memory buffer
        # self.memory_buffer = [None] * self.args.num_memory_buffer


        print("setting model to training mode")
        self.model.train()


        print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
        print('vocab: %s' % repr(self.vocab), file=sys.stderr)

        train_iter = 0
        report_loss = report_examples = replay_report_loss = replay_report_examples = 0.


        task_begin = time.time()

        for batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=True):

            bz = len(batch_examples)
            endcnt = min(self.mem_cnt + bz, self.args.num_memory_buffer)
            effbsz = endcnt - self.mem_cnt
            self.memory_buffer.extend(batch_examples[:effbsz])
            #print (len(self.memory_buffer))
            #print (endcnt)
            assert len(self.memory_buffer) == endcnt

            self.mem_cnt += effbsz

            for i in range(self.args.max_epoch):
                # self.reset_label_smoothing(self.model, t_offset1, t_offset2, nt_offset)
                train_iter += 1
                self.optimizer.zero_grad()

                report_loss, report_examples, loss, non_avg_loss = self.train_iter_process(self.model ,batch_examples, report_loss,
                                                                             report_examples)

                loss.backward()

                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_grad)

                self.optimizer.step()



                if len(self.memory_data) > 0:

                    replay_batch_examples = self.memory_data

                    # print (len(replay_batch_examples))

                    replay_report_loss, replay_report_examples, replay_loss, non_avg_loss = self.train_iter_process(
                        self.model, replay_batch_examples, replay_report_loss, replay_report_examples)


                    replay_loss.backward()

                    if self.args.clip_grad > 0.:
                        if self.args.clip_grad_mode == 'norm':
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                       self.args.clip_grad)
                        elif self.args.clip_grad_mode == 'value':
                            grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                                        self.args.clip_grad)

                    self.optimizer.step()

            if train_iter % self.args.log_every == 0:
                log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                if len(self.memory_data) > 0:
                    log_str += '[Iter %d] replay encoder loss=%.5f' % (
                        train_iter, replay_report_loss / replay_report_examples)

                print(log_str, file=sys.stderr)
                report_loss = report_examples = replay_report_loss = replay_report_examples = 0.

            # update buffer
            if self.mem_cnt == self.args.num_memory_buffer:

                self.sample_examples(self.memory_buffer, self.memory_data)

                self.mem_cnt = 0
                self.memory_buffer = []
                print("ring buffer is full, re-estimating of the constrains, we are at task", t)
                self.old_mem_grads = None


        print('[Task %d] task elapsed %ds' % (t, time.time() - task_begin), file=sys.stderr)

        model_file = self.args.save_to + '.bin'
        print('save the current model ..', file=sys.stderr)
        print('save model to [%s]' % model_file, file=sys.stderr)
        self.model.save(model_file)

    def sample_examples(self, buffer_examples, replay_examples):
        #print(len(buffer_examples))
        #print(len(replay_examples))

        if self.args.subselect and len(self.memory_data) > 0:
            buffer_examples = self.subselect_examples(self.model, Dataset(buffer_examples)).examples

        train_set = Dataset(buffer_examples + replay_examples)
        #print(len(train_set))
        if self.args.sample_method == 'random':
            pruned_train_set = train_set
            self.memory_data = pruned_train_set.random_sample_batch_iter(self.n_memories)

            # print (len(self.memory_data))


        elif self.args.sample_method == 'entropy' or self.args.sample_method == 'ne_entropy':
            self.eval()

            entropy = torch.zeros(len(train_set))

            # nt_entropy = torch.zeros(len(train_set))

            beg = end = 0
            for feature_batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=False, sort=False):
                batch = Batch(feature_batch_examples, self.vocab, use_cuda=self.use_cuda)
                nt_scores, t_scores = self.forward(self.model, batch)

                nt_prob = torch.softmax(nt_scores, dim=-1)

                t_prob = torch.softmax(t_scores, dim=-1)

                end += len(feature_batch_examples)

                # print (beg)
                # print (end)
                # print (replay_loss)
                # print (nt_prob)
                # print (nt_prob.size())
                # print (Categorical(probs=nt_prob).entropy().size())
                # print (torch.mean(Categorical(probs=nt_prob).entropy(), dim=0))
                # print(torch.mean(Categorical(probs=nt_prob).entropy(), dim=0).size())

                batch_entropy = (torch.mean(Categorical(probs=nt_prob).entropy(), dim=0) + torch.mean(
                    Categorical(probs=t_prob).entropy(), dim=0)).cpu().detach()

                entropy[beg:end].copy_(batch_entropy)

                # t_entropy[beg:end].copy_(torch.sum(Categorical(probs=t_prob).entropy(), dim=0).cpu().detach())

                beg += len(feature_batch_examples)
            assert end == len(train_set)

            # entropy = nt_entropy + t_entropy

            # print (entropy)
            if self.args.sample_method == 'entropy':
                entropy_index = [i[0] for i in sorted(enumerate(entropy.tolist()), key=lambda x: x[1], reverse=True)]
            elif self.args.sample_method == 'ne_entropy':
                entropy_index = [i[0] for i in sorted(enumerate(entropy.tolist()), key=lambda x: x[1])]

            temp_examples = []

            for top_idx in entropy_index[:self.n_memories]:
                temp_examples.append(train_set.examples[top_idx])

            assert len(temp_examples) == self.n_memories


            self.memory_data = temp_examples

            self.train()
        elif self.args.sample_method == 'f_cluster':
            self.eval()

            feature_vectors = torch.zeros(len(train_set), self.args.hidden_size)
            beg = end = 0
            for feature_batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=False, sort=False):
                batch = Batch(feature_batch_examples, self.vocab, use_cuda=self.use_cuda)
                src_sents_var = batch.src_sents_var
                src_sents_len = batch.src_sents_len
                src_encodings, (last_state, last_cell) = self.model.encode(src_sents_var, src_sents_len)
                dec_init_vec = self.model.init_decoder_state(last_state, last_cell)

                end += len(feature_batch_examples)


                feature_vectors[beg:end, :].copy_(dec_init_vec[0].detach())

                beg += len(feature_batch_examples)
            assert end == len(train_set)
            X = feature_vectors.cpu().numpy()

            km = KMeans(n_clusters=self.n_memories).fit(X)

            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)

            temp_examples = []

            for top_idx in closest.tolist():
                temp_examples.append(train_set.examples[top_idx])

            assert len(temp_examples) == self.n_memories

            self.memory_data = temp_examples

            self.train()
        elif self.args.sample_method == 'balance':
            self.memory_data = sample_balance_data(train_set.examples, self.n_memories)

        else:
            final_examples, inds = self.gss_select_examples(self.model, train_set,
                                                      nb_selected=self.n_memories)
            # selection_model.load_state_dict(model_state)
            begin_time = time.time()


            self.memory_data = final_examples


            print("Final selected instances number is {0}".format(len(self.memory_data)), file=sys.stderr)


            self.mem_grads = None
            print("Total selection cost time is {0}".format(time.time() - begin_time), file=sys.stderr)

    def gss_select_examples(self, selection_model, train_set, nb_selected):

        pruned_train_set = train_set


        final_examples, added_inds_list = self.select_samples_per_group(selection_model, pruned_train_set, nb_selected=nb_selected)

        final_examples.sort(key=lambda e: -len(e.src_sent))

        return final_examples, added_inds_list

    def subselect_examples(self, selection_model, train_set):
        selection_model.zero_grad()

        memory_examples = self.memory_data

        _, _, replay_loss, non_avg_loss = self.train_iter_process(selection_model, memory_examples, 0, 0)

        replay_loss.backward()

        store_grad(selection_model.parameters, self.grads, self.grad_dims,0)



        index_list = []


        idx = 0


        for example in train_set.examples:
            selection_model.zero_grad()

            _, _, ptloss, non_avg_loss = self.train_iter_process(selection_model, [example], 0, 0)

            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            # print (self.grad_dims)
            # copy gradient
            store_grad(selection_model.parameters, self.grads, self.grad_dims, 1)
            dotp = torch.mm(self.grads[:, 1].unsqueeze(0), self.grads[:, 0].unsqueeze(0))
            # cosine = get_G(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx).t())
            # cosine = get_G(self.grads[:, t].unsqueeze(0), a_g.unsqueeze(0).t())

            # print (self.grads[:, t], a_g)
            # cosine_list.append(cosine.sum().item())
            # print(dotp)
            if (dotp < 0).sum() == 0:  # or (dotp > 0.01).sum() == 0:
                index_list.append(idx)
            idx += 1

        # index_list = [i[0] for i in sorted(enumerate(cosine_list), key=lambda x: x[1])]

        print("Original dataset length is {0}".format(len(train_set.examples)), file=sys.stderr)
        pruned_train_set = Dataset([train_set.examples[i] for i in index_list])
        print("Pruned dataset length is {0}".format(len(pruned_train_set.examples)), file=sys.stderr)

        return pruned_train_set

    def get_complete_G(self, model, train_examples):

        train_set = Dataset(train_examples)

        G = torch.cuda.FloatTensor(len(train_set), len(train_set)).fill_(0.)

        # get gradients from the ring buffer
        i = out_beg = out_end = 0
        for out_examples in train_set.batch_iter(batch_size=self.args.gradient_buffer_size, shuffle=False, sort=False):
            j = in_beg = in_end = 0

            out_end += len(out_examples)

            out_mem_grads = torch.cuda.FloatTensor(len(out_examples), sum(self.grad_dims))


            # out_mask = torch.zeros(len(out_examples), device='cuda')

            # out_weight = 1 / len(out_examples)
            out_idx = 0
            for example in out_examples:
                model.zero_grad()

                # out_mask[out_idx] = out_weight

                _, _, ptloss, non_avg_loss = self.train_iter_process(model, [example], 0, 0)

                ptloss.backward()
                # add the new grad to the memory grads and add it is cosine similarity
                # print (self.grad_dims)
                add_memory_grad(model.parameters, out_mem_grads, self.grad_dims, out_idx)
                out_idx+=1

            for inner_examples in train_set.batch_iter(batch_size=self.args.gradient_buffer_size, shuffle=False, sort=False):
                in_end += len(inner_examples)
                if j >= i:
                    if i == j:
                        sub_G = get_G(out_mem_grads, out_mem_grads)
                    else:
                        in_mem_grads = torch.cuda.FloatTensor(len(inner_examples), sum(self.grad_dims))

                        # in_mask = torch.zeros(len(inner_examples), device='cuda')

                        # in_weight = 1 / len(inner_examples)

                        in_idx = 0
                        for example in inner_examples:
                            model.zero_grad()

                            # in_mask[in_idx] = in_weight
                            _, _, ptloss, non_avg_loss = self.train_iter_process(model, [example], 0, 0)

                            ptloss.backward()
                            # print (i)
                            # print (j)
                            # add the new grad to the memory grads and add it is cosine similarity
                            # be_ti = time.time()
                            add_memory_grad(model.parameters, in_mem_grads, self.grad_dims, in_idx)
                            # print ("Add memory time is ", time.time() - be_ti)
                            in_idx+=1

                        sub_G = get_G(out_mem_grads, in_mem_grads)
                        #print (sub_G.size())
                        del in_mem_grads

                    G[out_beg:out_end, in_beg: in_end].copy_(sub_G)
                    G[in_beg: in_end, out_beg:out_end].copy_(sub_G.T)

                in_beg += len(inner_examples)
                j += 1
            out_beg += len(out_examples)
            i += 1
            del out_mem_grads
        model.zero_grad()
        # print (G)
        # print (torch.nonzero((G==0), as_tuple=False).size(0))
        assert torch.nonzero((G==0), as_tuple=False).size(0) == 0

        return G


    def select_samples_per_group(self, model, train_set, nb_selected):
        """
        Assuming a ring buffer, backup constraints and constrains,
        re-estimate the backup constrains and constrains
        """

        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0

        begin_time = time.time()

        print("constraints selector")

        G = self.get_complete_G(model, train_set.examples)

        added_inds = cosine_similarity_selector_IQP_Exact(G,
                                                          nb_selected=nb_selected,
                                                          solver=self.solver, sample_method=self.args.sample_method)

        added_inds_list = added_inds.squeeze().tolist()

        selected_examples = [train_set.examples[indx] for indx in added_inds_list]

        print("Individual selection cost time {0}".format(time.time() - begin_time), file=sys.stderr)

        for name, module in model.named_modules():
            # print (name)
            if name == 'dropout':
                module.p = self.args.dropout


        return selected_examples, added_inds_list

    # check whether this is the last minibatch of the current task
    # We assume only 1 epoch!
    # if self.examples_seen == self.samples_per_task:
    #    self.examples_seen = 0
    # get labels from previous task; we assume labels are consecutive

    # Reduce exemplar set by updating value of num. exemplars per class
    # self.num_exemplars = int(self.n_memories / (num_classes + len(self.mem_class_x.keys())))

    def forward(self, model, batch):
        query_vectors = model(batch)
        nt_scores, t_scores = model.action_readout(query_vectors)

        return nt_scores, t_scores