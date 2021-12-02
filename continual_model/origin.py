# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import pdb
import time
from collections import Counter
import copy

from sklearn_extra.cluster import KMedoids
from zss import simple_distance

from components.evaluator import DefaultEvaluator, ActionEvaluator, SmatchEvaluator
import numpy
import torch
import sys
import math
import numpy as np
import random
import quadprog
import miosqp

from continual_model.irnet import IrNetModel
from continual_model.seq2seq_topdown import Seq2SeqModel
from continual_model.utils import ori_tree2edit_tree
from grammar.action import GenTAction, GenNTAction
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import OneHotEncoder
from torch import nn
from torch.distributions import Categorical

import evaluation
from common.registerable import Registrable
from components.dataset import Batch, Dataset, reweight_data, reweight_t_data, sample_balance_data, generate_concat_samples
from continual_model.OR_test import select_uniform_data
import scipy as sp
import mathprogbasepy as mpbpy
import scipy.sparse as spa
from model import nn_utils
from model.utils import GloveHelper
from preprocess_data.utils import parse_overnight_query_helper, parse_nlmap_query_helper

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

def cosine_similarity_selector_IQP_Exact(G, solver, nb_selected, sample_method='normal', cos_G = None, smooth = 0):
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
    elif sample_method == 'pro_max':
        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        del G
        torch.cuda.empty_cache()
        #print(max_G)
        max_G = -max_G + 1
        #print (max_G)
        #max_G[max_G<1] = 0
        #print (max_G)
        prob = (max_G / torch.sum(max_G).item()).cpu().numpy()
        #print (prob)
        inds = np.random.choice(t, nb_selected, replace=False, p=prob)
        #print (inds)
        # values, inds = torch.topk(max_G, nb_selected)
        return inds
    elif sample_method == 'pro_max_gem':

        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        del G
        torch.cuda.empty_cache()
        #print(max_G.size())
        max_G = -max_G + 1
        #max_G[max_G<1] = 0
        #print (max_G)
        max_prob = (max_G / torch.sum(max_G).item()).cpu().numpy()
        prob = max_prob
        #print(max_prob)
        if cos_G is not None:
            assert torch.nonzero((cos_G == 0), as_tuple=False).size(0) == 0
            cos_G = cos_G + 1

            cos_prob = (cos_G / torch.sum(cos_G).item()).cpu().numpy()

            #print (cos_prob)
            sum_prob = max_prob + cos_prob
            prob = (sum_prob / sum_prob.sum())
            #print (prob)
        inds = np.random.choice(t, nb_selected, replace=False, p=prob)
        #print (inds)
        # values, inds = torch.topk(max_G, nb_selected)
        return inds
    elif sample_method == 'pro_max_gem_sub':

        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        del G
        torch.cuda.empty_cache()
        #print(max_G.size())
        max_G = -max_G + 1
        #max_G[max_G<1] = 0
        #print (max_G)
        max_prob = (max_G / torch.sum(max_G).item()).cpu().numpy()
        prob = max_prob
        #print(max_prob)
        if cos_G is not None:
            assert torch.nonzero((cos_G == 0), as_tuple=False).size(0) == 0
            cos_G[cos_G<0] = 0
            # print (cos_G)
            cos_prob = (cos_G / torch.sum(cos_G).item()).cpu().numpy()

            #print (cos_prob)
            sum_prob = max_prob + cos_prob
            prob = (sum_prob / sum_prob.sum())
            #print (prob)
        inds = np.random.choice(t, nb_selected, replace=False, p=prob)
        #print (inds)
        # values, inds = torch.topk(max_G, nb_selected)
        return inds
    elif sample_method == 'pro_max_gem_reverse':

        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        del G
        torch.cuda.empty_cache()
        #print(max_G.size())
        max_G = 1/(max_G - torch.min(max_G).item() + smooth)
        #max_G[max_G<1] = 0
        #print (max_G)
        #print(max_prob)
        if cos_G is not None:
            assert torch.nonzero((cos_G == 0), as_tuple=False).size(0) == 0
            cos_G_index = cos_G < 0
            cos_G = cos_G - torch.min(cos_G).item()
            cos_G[cos_G_index] = 0
            cos_G = cos_G + smooth
            reverse_G = cos_G.cpu()*max_G.cpu()

            #reverse_G[cos_G < 1] = 0

            reverse_G[reverse_G < 0] = 0
            # print (cos_G)
            # cos_prob = (cos_G / torch.sum(cos_G).item()).cpu().numpy()

            #print (cos_prob)
            # sum_prob = max_prob + cos_prob
            prob = (reverse_G / torch.sum(reverse_G).item()).cpu().numpy()
        else:
            prob = (max_G / torch.sum(max_G).item()).cpu().numpy()
            #(sum_prob / sum_prob.sum())
            #print (prob)
        inds = np.random.choice(t, nb_selected, replace=False, p=prob)
        #print (inds)
        # values, inds = torch.topk(max_G, nb_selected)
        return inds
    elif sample_method == 'pro_max_gem_rescale':

        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        del G
        torch.cuda.empty_cache()
        #print(max_G.size())
        #max_G[max_G<1] = 0
        #print (max_G)
        #print(max_prob)
        if cos_G is not None:
            assert torch.nonzero((cos_G == 0), as_tuple=False).size(0) == 0

            max_G = - max_G
            max_G = (max_G - torch.min(max_G).item()) / (torch.max(max_G).item() - torch.min(max_G).item())

            cos_G_index = cos_G < 0
            cos_G[cos_G_index] = 0
            cos_G = (cos_G - torch.min(cos_G).item())/(torch.max(cos_G).item() - torch.min(cos_G).item())
            reverse_G = cos_G.cpu() + max_G.cpu() + smooth
            print (reverse_G)
            # reverse_G[cos_G_index] = 0

            #reverse_G[cos_G < 1] = 0

            #reverse_G[reverse_G < 0] = 0
            # print (cos_G)
            # cos_prob = (cos_G / torch.sum(cos_G).item()).cpu().numpy()

            #print (cos_prob)
            # sum_prob = max_prob + cos_prob
            prob = (reverse_G / torch.sum(reverse_G).item()).cpu().numpy()
        else:
            max_G = - max_G
            max_G = (max_G - torch.min(max_G).item()) / (torch.max(max_G).item() - torch.min(max_G).item()) + smooth
            prob = (max_G / torch.sum(max_G).item()).cpu().numpy()
            #(sum_prob / sum_prob.sum())
            #print (prob)
        inds = np.random.choice(t, nb_selected, replace=False, p=prob)
        #print (inds)
        # values, inds = torch.topk(max_G, nb_selected)
        return inds
    elif sample_method == 'pro_max_gem_turn':
        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        del G
        torch.cuda.empty_cache()
        #print(max_G.size())
        #max_G[max_G<1] = 0
        #print (max_G)
        #print(max_prob)
        if cos_G is not None:
            assert torch.nonzero((cos_G == 0), as_tuple=False).size(0) == 0

            max_G = - max_G
            max_G = (max_G - torch.min(max_G).item()) / (torch.max(max_G).item() - torch.min(max_G).item())

            cos_G_index = cos_G < 0
            cos_G[cos_G_index] = 0
            cos_G = (cos_G - torch.min(cos_G).item())/(torch.max(cos_G).item() - torch.min(cos_G).item())

            reverse_G = cos_G.cpu() * max_G.cpu()

            reverse_G = (reverse_G - torch.min(reverse_G).item())/(torch.max(reverse_G).item() - torch.min(reverse_G).item()) + smooth

            # + smooth
            print (reverse_G)
            # reverse_G[cos_G_index] = 0

            #reverse_G[cos_G < 1] = 0

            #reverse_G[reverse_G < 0] = 0
            # print (cos_G)
            # cos_prob = (cos_G / torch.sum(cos_G).item()).cpu().numpy()

            #print (cos_prob)
            # sum_prob = max_prob + cos_prob
            prob = (reverse_G / torch.sum(reverse_G).item()).cpu().numpy()
        else:
            max_G = - max_G
            max_G = (max_G - torch.min(max_G).item()) / (torch.max(max_G).item() - torch.min(max_G).item()) + smooth
            prob = (max_G / torch.sum(max_G).item()).cpu().numpy()
            #(sum_prob / sum_prob.sum())
            #print (prob)
        inds = np.random.choice(t, nb_selected, replace=False, p=prob)
        #print (inds)
        # values, inds = torch.topk(max_G, nb_selected)
        return inds
    elif sample_method == 'pro_max_gem_reverse_v2':

        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        del G
        torch.cuda.empty_cache()
        #print(max_G.size())
        #max_G[max_G<1] = 0
        #print (max_G)
        #print(max_prob)
        if cos_G is not None:
            assert torch.nonzero((cos_G == 0), as_tuple=False).size(0) == 0

            max_G = - max_G
            max_G = (max_G - torch.min(max_G).item()) / (torch.max(max_G).item() - torch.min(max_G).item())

            cos_G_index = cos_G < 0
            cos_G[cos_G_index] = 0
            cos_G = (cos_G - torch.min(cos_G).item())/(torch.max(cos_G).item() - torch.min(cos_G).item())
            reverse_G = cos_G.cpu() + max_G.cpu()
            reverse_G = (reverse_G - torch.min(reverse_G).item()) / (torch.max(reverse_G).item() - torch.min(reverse_G).item())
            reverse_G = reverse_G + smooth
            # print (reverse_G)
            # reverse_G[cos_G_index] = 0

            #reverse_G[cos_G < 1] = 0

            #reverse_G[reverse_G < 0] = 0
            # print (cos_G)
            # cos_prob = (cos_G / torch.sum(cos_G).item()).cpu().numpy()

            #print (cos_prob)
            # sum_prob = max_prob + cos_prob
            prob = (reverse_G / torch.sum(reverse_G).item()).cpu().numpy()
        else:
            max_G = - max_G
            max_G = (max_G - torch.min(max_G).item()) / (torch.max(max_G).item() - torch.min(max_G).item()) + smooth
            prob = (max_G / torch.sum(max_G).item()).cpu().numpy()
            #(sum_prob / sum_prob.sum())
            #print (prob)
        inds = np.random.choice(t, nb_selected, replace=False, p=prob)
        #print (inds)
        # values, inds = torch.topk(max_G, nb_selected)
        return inds
    elif sample_method == 'pro_max_gem_reverse_v1':

        t = G.size(0)
        G[torch.eye(t).bool()] = -float('inf')
        max_G, max_G_inds = torch.max(G, dim=1)
        del G
        torch.cuda.empty_cache()
        #print(max_G.size())
        max_G = 1/(max_G + 1)
        #max_G[max_G<1] = 0
        #print (max_G)
        max_prob = (max_G / torch.sum(max_G).item()).cpu().numpy()
        prob = max_prob
        #print(max_prob)
        if cos_G is not None:
            assert torch.nonzero((cos_G == 0), as_tuple=False).size(0) == 0
            cos_G = cos_G + 1
            cos_G[cos_G<1] = 0
            # print (cos_G)
            cos_prob = (cos_G / torch.sum(cos_G).item()).cpu().numpy()

            #print (cos_prob)
            sum_prob = max_prob + cos_prob
            prob = (sum_prob / sum_prob.sum())
            #print (prob)
        inds = np.random.choice(t, nb_selected, replace=False, p=prob)
        #print (inds)
        # values, inds = torch.topk(max_G, nb_selected)
        return inds
    elif sample_method == 'topk_sum':
        sum_G = torch.sum(G, dim=1)
        values, inds = torch.topk(sum_G, nb_selected)
        return inds
    elif sample_method == 'pro_sum':
        t = G.size(0)
        sum_G = torch.sum(G, dim=1)
        #print(max_G.size())
        prob = torch.softmax(sum_G, dim=-1).cpu().numpy()
        #print (prob)
        inds = np.random.choice(t, nb_selected, replace=False, p=prob)
        #print (inds)
        # values, inds = torch.topk(max_G, nb_selected)
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
        elif sample_method == 'init_random':
            inderr = torch.randperm(t).tolist()
            x0 = np.zeros(t)
            x0[inderr[:nb_selected]] = 1
            solver.set_x0(x0)

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


@Registrable.register('origin')
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

        self.dev_list = []

        self.memory_data = {}  # stores exemplars class by class

        self.memory_random_data = {}  # temp random data

        self.rebalanced_memory_data = {}

        self.memory_buffer = set()

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

        if args.base_model == 'seq2seq':
            self.model = Seq2SeqModel(self.vocab, args)
        elif args.base_model == 'irnet':
            self.model = IrNetModel(self.vocab, args)
        else:
            raise ValueError

        if args.embed_type == 'glove':
            # print ("=============================================glove golver dadas =======================")
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
            self.src_ids = glove_embedding.load_to(self.model.src_embed, self.vocab.source)

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

        self.fisher = {}
        self.optpar = {}

        # allocate counters
        self.observed_tasks = []
        self.old_task = -1
        self.current_task = 0
        # allocate temporary synaptic memory
        self.grad_dims = []
        for param in self.parameters():
            self.grad_dims.append(param.data.numel())

        self.query_vector = None

        n_tasks = len(self.t_nc_per_task)

        self.n_tasks = n_tasks


        self.grads = torch.Tensor(sum(self.grad_dims), n_tasks)
        if args.use_cuda:
            self.grads = self.grads.cuda()


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

    def reset_label_smoothing(self, model, t_offset1, t_offset2, nt_offset):
        if self.args.label_smoothing:
            if t_offset2 - t_offset1 <= 1:
                # a hack
                model.t_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing,
                                                                        t_offset2 - t_offset1 + 1, use_cuda=self.args.use_cuda)
            else:
                model.t_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing,
                                                                         t_offset2 - t_offset1 + 1,
                                                                         ignore_indices=[0],use_cuda=self.args.use_cuda)

            model.nt_label_smoothing_layer = nn_utils.LabelSmoothing(self.args.label_smoothing, nt_offset,
                                                                          ignore_indices=[0],use_cuda=self.args.use_cuda)

    def train_iter_process(self, model, batch_examples, report_loss, report_examples, t_offset1, t_offset2, nt_offset):
        # print (batch_examples)
        batch = Batch(batch_examples, self.vocab, use_cuda=self.use_cuda)

        nt_scores, t_scores = self.forward_with_offset(model, t_offset1, t_offset2, nt_offset,
                                                       batch)

        batch.t_action_idx_matrix[batch.t_action_idx_matrix.nonzero(as_tuple=True)] = batch.t_action_idx_matrix[
                                                                                          batch.t_action_idx_matrix.nonzero(
                                                                                              as_tuple=True)] - t_offset1 + 1

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

        self.current_task = t

        print("========= Current task is {0} ============".format(self.current_task))

        train_set = task_data.train

        if task_data.dev:
            dev_set = task_data.dev
        else:
            dev_set = Dataset(examples=[])


        t_offset1, t_offset2 = self.compute_offsets(t)

        nt_offset = self.nt_nc_per_task[t]


        print("setting model to training mode")
        self.model.train()


        self.model.training_state = 'normal_train'

        print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
        print('vocab: %s' % repr(self.vocab), file=sys.stderr)

        epoch = train_iter = 0
        report_loss = report_examples = 0.
        num_trial = patience = 0



        while True:
            epoch += 1
            epoch_begin = time.time()


            for batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=True):

                self.reset_label_smoothing(self.model, t_offset1, t_offset2, nt_offset)
                train_iter += 1
                self.optimizer.zero_grad()

                report_loss, report_examples, loss, non_avg_loss = self.train_iter_process(self.model ,batch_examples, report_loss,
                                                                             report_examples, t_offset1, t_offset2,
                                                                             nt_offset)


                loss.backward()

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

                    print(log_str, file=sys.stderr)
                    report_loss = report_examples = 0.


            print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

            if self.args.save_all_models:
                model_file = self.args.save_to + '.iter%d.bin' % train_iter
                print('save model to [%s]' % model_file, file=sys.stderr)
                self.model.save(model_file)

            # perform validation
            if False:
                # dev_set = task_data.dev
                if epoch % self.args.valid_every_epoch == 0:
                    # dev_score = 0
                    # for dev_set in self.dev_list:
                    print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                    eval_start = time.time()
                    eval_results = evaluation.evaluate(dev_set.examples, self.model, self.evaluator, self.args,
                                                       verbose=True, eval_top_pred_only=self.args.eval_top_pred_only)
                    dev_score = eval_results[self.evaluator.default_metric]

                    print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                        epoch, eval_results,
                        self.evaluator.default_metric,
                        dev_score,
                        time.time() - eval_start), file=sys.stderr)
                    # dev_score = dev_score/len(self.dev_list)
                    is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                    history_dev_scores.append(dev_score)
            else:
                is_better = False

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
                params = torch.load(self.args.save_to + '.bin', map_location=lambda storage, loc: storage)
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
                    self.optimizer.load_state_dict(torch.load(self.args.save_to + '.optim.bin'))

                # set new lr
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr

                # reset patience
                patience = 0



        model_file = self.args.save_to + '.bin'
        print('save the current model ..', file=sys.stderr)
        print('save model to [%s]' % model_file, file=sys.stderr)
        self.model.save(model_file)

        self.mem_cnt = 0



    def forward_with_offset(self, model, t_offset1, t_offset2, nt_offset, batch):
        query_vectors = model(batch)
        self.query_vector = query_vectors
        #if model.training_state == 'recon_train':
        #    nt_scores, t_scores = model.action_weight_readout(query_vectors, self.model.nt_stable_embed.weight, self.model.t_stable_embed.weight)
        #else:
        nt_scores, t_scores = model.action_readout(query_vectors)
        pad_t_scores = t_scores[:, :, 0].unsqueeze(-1)
        task_t_scores = t_scores[:, :, t_offset1: t_offset2]
        t_action_scores = torch.cat((pad_t_scores, task_t_scores), -1)

        nt_scores = nt_scores[:, :, :nt_offset]

        return nt_scores, t_action_scores