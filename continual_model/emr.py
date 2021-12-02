# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import time
from collections import Counter

import numpy
import torch
import sys
import numpy as np
import random
import scipy.sparse as spa
import quadprog
import miosqp
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from torch.distributions import Categorical

import evaluation
from common.registerable import Registrable
from components.dataset import Batch, Dataset, data_augmentation, generate_augment_samples, generate_concat_samples, \
    reweight_data, reweight_t_data
from continual_model.seq2seq_topdown import Seq2SeqModel
from continual_model.utils import read_domain_vocab, read_vocab_list
from grammar.action import GenNTAction, GenTAction
from grammar.consts import TYPE_SIGN, NT
from grammar.hypothesis import Hypothesis
import mathprogbasepy as mpbpy
import miosqp
from model import nn_utils
from model.utils import GloveHelper
import scipy as sp
import scipy.sparse as spa

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

osqp_settings = {'eps_abs': 1e-03,
                 'eps_rel': 1e-03,
                 'eps_prim_inf': 1e-04,
                 'verbose': False}

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


def cosine_similarity_selector_IQP_Exact(x1, solver, nb_selected, slack=0.0):
	"""
	Integer programming
	"""

	x2 = None

	w1 = x1.norm(p=2, dim=1, keepdim=True)

	inds = torch.nonzero(torch.gt(w1, slack),as_tuple=False)[:, 0]
	if inds.size(0) < nb_selected:
		print("WARNING GRADIENTS ARE TOO SMALL!!!!!!!!")
		inds = torch.arange(0, x1.size(0))
	w1 = w1[inds]
	x1 = x1[inds]
	x2 = x1 if x2 is None else x2
	w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
	G = torch.mm(x1, x2.t()) / (w1 * w2.t())  # .clamp(min=eps)
	t = G.size(0)

	G = G.double().numpy()

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
	G = spa.csc_matrix(G)

	C = spa.csc_matrix(C)

	# solver.setup(G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper, miosqp_settings, osqp_settings)
	# results = solver.solve()

	prob = mpbpy.QuadprogProblem(G, a, C, h_final_lower, h_final_upper, idx, hlower, hupper)
	results = prob.solve(solver=mpbpy.GUROBI,
							verbose=False, Threads=6)

	print ("Solved time is", 1e3 * results.cputime)
	print("STATUS", results.status)
	coeffiecents_np = results.x
	coeffiecents = torch.nonzero(torch.Tensor(coeffiecents_np), as_tuple=False)
	print("number of selected items is", sum(coeffiecents_np))
	if "Infeasible" in results.status:
		return inds

	return inds[coeffiecents.squeeze()]


def get_grad_vector(pp, grad_dims):
	"""
	 gather the gradients in one vector
	"""
	grads = torch.Tensor(sum(grad_dims))
	grads.fill_(0.0)
	cnt = 0


	for param in pp():
		if param.grad is not None:
			beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
			en = sum(grad_dims[:cnt + 1])

			grads[beg: en].copy_(param.grad.data.view(-1))
		cnt += 1
	return grads


def add_memory_grad(pp, mem_grads, grad_dims):
	"""
		This stores the gradient of a new memory and compute the dot product with the previously stored memories.
		pp: parameters
		mem_grads: gradients of previous memories
		grad_dims: list with number of parameters per layers
	"""

	# gather the gradient of the new memory
	grads = get_grad_vector(pp, grad_dims)

	if mem_grads is None:

		mem_grads = grads.unsqueeze(dim=0)


	else:

		grads = grads.unsqueeze(dim=0)

		mem_grads = torch.cat((mem_grads, grads), dim=0)

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


@Registrable.register('emr')
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

        self.augment_memory_data = {}

        self.rebalanced_memory_data = {}

        self.memory_buffer = {}

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

        self.n_memories = 0

        # self.vocab, self.nc_per_task

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

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1

        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

        optimizer_cls = eval('torch.optim.%s' % self.args.optimizer)

        if self.args.optimizer == 'RMSprop':
            self.optimizer = optimizer_cls(self.parameters(), lr = self.args.lr, alpha = self.args.alpha)
        else:
            self.optimizer = optimizer_cls(self.parameters(), lr = self.args.lr)

    def select_samples_per_group(self, task, train_set):
        """
		Assuming a ring buffer, backup constraints and constrains,
		re-estimate the backup constrains and constrains
		"""

        begin_time = time.time()
        t_offset1, t_offset2 = self.compute_offsets(task)

        nt_offset = self.nt_nc_per_task[task]

        self.reset_label_smoothing(t_offset1, t_offset2, nt_offset)

        print("constraints selector")
        self.mem_grads = None
        # get gradients from the ring buffer


        for example in train_set.examples:
            self.zero_grad()

            _, _, ptloss = self.train_iter_process([example], 0, 0, t_offset1, t_offset2,
                                                   nt_offset)

            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            # print (self.grad_dims)
            self.mem_grads = add_memory_grad(self.parameters, self.mem_grads, self.grad_dims)


        added_inds = cosine_similarity_selector_IQP_Exact(self.mem_grads,
                                                              nb_selected=self.n_memories,
                                                              solver=self.solver)



        added_inds_list = added_inds.squeeze().tolist()

        self.memory_data[task] = Dataset([train_set.examples[indx] for indx in added_inds_list])

        print ("Selected instances number is", len(self.memory_data[task]))

        self.mem_grads = None
        print("Selection cost time", time.time() - begin_time)


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

        decode_results = []
        for e in x:
            decode_results.append(
                self.model.beam_search(e, self.args.decode_max_time_step, t_offset1, t_offset2, nt_offset,
                                       beam_size=self.args.beam_size))
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

    def reset_learning_rate(self, lr):
        print('reset learning rate to %f' % lr, file=sys.stderr)

        # set new lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def observe(self, task_data, t):


        # self.reset_learning_rate(self.args.lr)

        # reset label smoothing

        train_set = task_data.train
        # self.reset_learning_rate(self.args.lr)
        # print (type(train_set))

        if task_data.dev:
            dev_set = task_data.dev
        else:
            dev_set = Dataset(examples=[])

        if t != self.old_task:
            self.observed_tasks.append(t)
            self.old_task = t

        n_memories = int(len(train_set) * self.args.num_exemplars_ratio)

        self.n_memories = n_memories

        # self.memory_data[t] = train_set.random_sample_batch_iter(n_memories)
        # self.rebalanced_memory_data[t] = Dataset(reweight_t_data(train_set.examples))
        """
        if t in self.memory_data:
            print ("T in memory data")
            self.memory_data[t].add(train_set.random_sample_batch_iter(n_memories))
        else:
            self.memory_data[t] = Dataset(train_set.random_sample_batch_iter(n_memories))
        """
        print("setting model to training mode")
        self.model.train()

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

                report_loss, report_examples, loss = self.train_iter_process(batch_examples, report_loss, report_examples, t_offset1, t_offset2, nt_offset)

                loss.backward()

                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_grad)

                self.optimizer.step()

                # replay batch
                # sample_size = int(self.args.batch_size / len(self.observed_tasks) - 1)
                # rand_num = numpy.random.uniform()

                # if rand_num > 1 - self.args.replay_dropout:

                if len(self.observed_tasks) > 1:

                    for tt in range(len(self.observed_tasks) - 1):
                        self.optimizer.zero_grad()
                        # fwd/bwd on the examples in the memory
                        past_task = self.observed_tasks[tt]

                        p_t_offset1, p_t_offset2 = self.compute_offsets(past_task)

                        p_nt_offset = self.nt_nc_per_task[past_task]

                        self.reset_label_smoothing(p_t_offset1, p_t_offset2, p_nt_offset)

                        assert past_task != t

                        replay_batch_examples = self.memory_data[past_task]

                        if self.args.augment and train_iter%self.args.augment == 0:
                            #print (replay_batch_examples)
                            #print ("====================")
                            replay_batch_examples = generate_concat_samples(replay_batch_examples, batch_examples)
                            #print (replay_batch_examples)

                        if self.args.rebalance:
                            ori_sample_size = len(replay_batch_examples)
                            rebalance_data = self.rebalanced_memory_data[past_task]

                            replay_batch_examples = rebalance_data.random_sample_batch_iter(ori_sample_size)




                        replay_report_loss, replay_report_examples, replay_loss = self.train_iter_process(replay_batch_examples, replay_report_loss, replay_report_examples, p_t_offset1, p_t_offset2, p_nt_offset)

                        if self.args.ada_emr:
                            ada_ratio = 1/(t-past_task)
                            replay_loss = ada_ratio * replay_loss

                        # if tt < len(self.observed_tasks) - 2:
                            #replay_loss.backward(retain_graph=True)
                        #else:
                        replay_loss.backward()

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


        model_file = self.args.save_to + '.bin'
        print('save the current model ..', file=sys.stderr)
        print('save model to [%s]' % model_file, file=sys.stderr)
        self.model.save(model_file)

        if self.args.sample_method == 'random':
            self.memory_data[t] = train_set.random_sample_batch_iter(n_memories)
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

                # print (beg)
                # print (end)
                # print (replay_loss)

                feature_vectors[beg:end, :].copy_(dec_init_vec[0].detach())

                beg += len(feature_batch_examples)
            assert end == len(train_set)
            X = feature_vectors.cpu().numpy()

            km = KMeans(n_clusters=n_memories).fit(X)

            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)

            temp_examples = []

            for top_idx in closest.tolist():
                temp_examples.append(train_set.examples[top_idx])

            assert len(temp_examples) == n_memories

            self.memory_data[t] = temp_examples

            self.train()
        elif self.args.sample_method == 'entropy' or self.args.sample_method == 'ne_entropy':
            self.eval()

            entropy = torch.zeros(len(train_set))

            # nt_entropy = torch.zeros(len(train_set))

            beg = end = 0
            for feature_batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=False, sort=False):

                batch = Batch(feature_batch_examples, self.vocab, use_cuda=self.use_cuda)
                nt_scores, t_scores = self.forward_with_offset(self.model, t_offset1, t_offset2, nt_offset,
                                                               batch)

                nt_prob = torch.softmax(nt_scores, dim=-1)

                t_prob = torch.softmax(t_scores, dim=-1)

                end += len(feature_batch_examples)

                # print (beg)
                # print (end)
                # print (replay_loss)
                #print (nt_prob)
                # print (nt_prob.size())
                # print (Categorical(probs=nt_prob).entropy().size())
                # print (torch.mean(Categorical(probs=nt_prob).entropy(), dim=0))
                # print(torch.mean(Categorical(probs=nt_prob).entropy(), dim=0).size())

                batch_entropy = (torch.mean(Categorical(probs=nt_prob).entropy(), dim=0) + torch.mean(Categorical(probs=t_prob).entropy(), dim=0)).cpu().detach()

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

            for top_idx in entropy_index[:n_memories]:
                temp_examples.append(train_set.examples[top_idx])

            assert len(temp_examples) == n_memories

            template_count = Counter(
                [e.tgt_ast_t_seq.to_lambda_expr if e.tgt_ast_t_seq.to_lambda_expr else e.to_logic_form for e in
                 task_data.train.examples])

            instance_count = [template_count[e.tgt_ast_t_seq.to_lambda_expr] for e in temp_examples]
            print(template_count.values())
            print(instance_count)

            self.memory_data[t] = temp_examples

            self.train()


        if self.args.augment:
            self.augment_memory_data[t] = data_augmentation(self.memory_data[t], self.model.t_action_vocab)

        if self.args.rebalance:
            self.rebalanced_memory_data[t] = Dataset(reweight_t_data(self.memory_data[t]))
            # print (len(self.rebalanced_memory_data[t]))
            # print ([e for e in self.memory_data[t]])
            # print (len(self.memory_data[t]))
            # print ("==================")
            # print (len(reweight_data(self.memory_data[t])))
            # print ([e for e in reweight_data(self.memory_data[t])])

# elif self.args.sample_method == 'gss':
            # self.select_samples_per_group(t, train_set)

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