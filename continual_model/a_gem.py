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


@Registrable.register('a_gem')
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

		if args.optimizer == 'RMSprop':
			self.optimizer = optimizer_cls(self.parameters(), lr=args.lr, alpha=args.alpha)
		else:
			self.optimizer = optimizer_cls(self.parameters(), lr=args.lr)

		self.kl = torch.nn.KLDivLoss()  # for distillation
		self.lsm = torch.nn.LogSoftmax(dim=1)
		self.sm = torch.nn.Softmax(dim=1)

		self.margin = args.memory_strength
		self.grad_dims = []
		for param in self.parameters():
			self.grad_dims.append(param.data.numel())
		n_tasks = len(self.t_nc_per_task)
		self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
		if args.use_cuda:
			self.grads = self.grads.cuda()

		# allocate counters
		self.observed_tasks = []
		self.old_task = -1
		self.mem_cnt = 0

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
			if t_offset2 - t_offset1 <= 1:
				# a hack
				self.model.t_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing,
				                                                        t_offset2 - t_offset1 + 1,
				                                                        use_cuda=self.args.use_cuda)
			else:
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

		n_memories = int(self.args.num_exemplars_per_task)

		self.memory_data[t] = []

		self.memory_data[t].extend(train_set.random_sample_batch_iter(n_memories))

		print("setting model to training mode")
		self.model.train()

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


		while True:
			epoch += 1
			epoch_begin = time.time()

			for batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=True):

				# compute gradient on previous tasks
				if len(self.observed_tasks) > 1:
					for tt in range(len(self.observed_tasks) - 1):
						self.zero_grad()
						# fwd/bwd on the examples in the memory
						past_task = self.observed_tasks[tt]

						p_t_offset1, p_t_offset2 = self.compute_offsets(past_task)

						p_nt_offset = self.nt_nc_per_task[past_task]

						self.reset_label_smoothing(p_t_offset1, p_t_offset2, p_nt_offset)

						memory_examples = self.memory_data[past_task]

						_, _, replay_loss = self.train_iter_process(memory_examples, 0, 0, p_t_offset1, p_t_offset2,
																	p_nt_offset)

						replay_loss.backward()

						store_grad(self.parameters, self.grads, self.grad_dims,
								   past_task)

				# now compute the grad on the current minibatch
				self.zero_grad()

				t_offset1, t_offset2 = self.compute_offsets(t)

				nt_offset = self.nt_nc_per_task[t]

				self.reset_label_smoothing(t_offset1, t_offset2, nt_offset)

				# print(offset1)
				# print(offset2)
				train_iter += 1
				# self.optimizer.zero_grad()
				# print (batch_examples)

				report_loss, report_examples, loss = self.train_iter_process(batch_examples, report_loss,
																			 report_examples, t_offset1, t_offset2,
																			 nt_offset)

				loss.backward()


				if len(self.observed_tasks) > 1:
					# copy gradient
					store_grad(self.parameters, self.grads, self.grad_dims, t)
					indx = self.model.new_long_tensor(self.observed_tasks[:-1])

					example_num = sum([len(self.memory_data[t]) for t in self.observed_tasks[:-1]])

					a_g = self.grads.index_select(1, indx).sum(-1)/example_num

					dotp = torch.dot(self.grads[:, t], a_g)

					# print (dotp)

					# print (len(self.observed_tasks))

					if (dotp < 0).sum() != 0:
						# project2cone2(self.grads[:, t].unsqueeze(1), self.grads.index_select(1, indx), self.margin)
						# print (self.grads[:, t].size())
						# print (a_g.size())

						# print (torch.dot(self.grads[:, t], a_g).size())

						# print (torch.dot(a_g, a_g).size())

						# print ((self.grads[:, t] - torch.div(torch.dot(self.grads[:, t], a_g), torch.dot(a_g, a_g)) * a_g).size())

						# print ((self.grads[:, t] - torch.div(torch.dot(self.grads[:, t], a_g), torch.dot(a_g, a_g)) * a_g).contiguous().view(-1, 1).size())

						updated_grad = (self.grads[:, t] - torch.div(torch.dot(self.grads[:, t], a_g), torch.dot(a_g, a_g)) * a_g).detach()

						self.grads[:, t].copy_(updated_grad)

						# copy gradients back
						overwrite_grad(self.parameters, self.grads[:, t],
									   self.grad_dims)

						updated_dotp = torch.dot(self.grads[:, t], a_g)

						assert updated_dotp.item() > -0.01, "updated gradient is {0}".format(updated_dotp.item())
					# print(dotp)

				# bprop and update
				# nt_action_loss.backward()

				if self.args.clip_grad > 0.:
					if self.args.clip_grad_mode == 'norm':
						grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
					elif self.args.clip_grad_mode == 'value':
						grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_grad)

				self.optimizer.step()

				if train_iter % self.args.log_every == 0:
					log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)

					print(log_str, file=sys.stderr)
					report_loss = report_examples = 0.

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
		# also save the optimizers' state
		torch.save(self.optimizer.state_dict(), self.args.save_to + '.optim.bin')

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