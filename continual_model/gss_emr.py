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
from continual_model.irnet_q import IrNetQModel
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


@Registrable.register('gss_emr')
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
        elif args.base_model == 'irnetq':
            self.model = IrNetQModel(self.vocab, args)
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

        """
        if self.args.second_iter:
            self.nt_action_embed = {}
            self.t_action_embed = {}
        """

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


    def init_action_embedding(self, examples, init_all=False):
        self.model.training_state = 'init_embedding'
        att_logits_list = []
        instance_batch_list = []
        for idx, example in enumerate(examples):
            self.model.zero_grad()
            single_instance_batch = Batch([example], self.vocab, use_cuda=self.use_cuda)
            query_vectors = self.model(single_instance_batch)
            att_logits_list.append(query_vectors.detach())
            instance_batch_list.append(single_instance_batch)

        nt_action_embedding = {}
        t_action_embedding = {}
        #print (self.model.known_action_set)
        for template_id, template_batch in enumerate(instance_batch_list):
            template_seq = template_batch.action_seq
            att_seq = att_logits_list[template_id]
            #print (att_seq.size())
            #print (len(template_seq[0]))
            #if (len(list(self.model.known_action_set & set(template_seq[0]))) < len(template_seq[0])//2):
            #    print (len(list(self.model.known_action_set & set(template_seq[0]))))
            #    print (len(template_seq[0]))
            #    continue
            for att_id, action in enumerate(template_seq[0]):
                #print (att_seq[att_id].size())
                if (not (action in self.model.known_action_set)) or init_all:
                    #print ("=============new action==============")
                    #print (action)
                    if isinstance(action, GenNTAction):
                        if action in nt_action_embedding:
                            nt_action_embedding[action].append(att_seq[att_id])
                        else:
                            nt_action_embedding[action] = []
                            nt_action_embedding[action].append(att_seq[att_id])
                    elif isinstance(action, GenTAction):
                        if action in t_action_embedding:
                            t_action_embedding[action].append(att_seq[att_id])
                        else:
                            t_action_embedding[action] = []
                            t_action_embedding[action].append(att_seq[att_id])
                #else:
                    #continue
                    #print("=============known action==============")
                    #print(action)
                    #print(action)

        for action, action_embedding_list in nt_action_embedding.items():
            #print (action)
            action_id = self.model.nt_action_vocab[action]
            #print (action_id)
            #print (len(action_embedding_list))
            #print (torch.stack(action_embedding_list).size())
            #action_embedding_list.append(self.model.action_nt_embed.weight[action_id].data.unsqueeze(0).detach())
            action_embedding = torch.stack(action_embedding_list).mean(dim=0)
            #print(action_embedding)
            self.model.action_nt_embed.weight[action_id].data.copy_(action_embedding.squeeze().detach())

        for action, action_embedding_list in t_action_embedding.items():
            action_id = self.model.t_action_vocab[action]
            #action_embedding_list.append(self.model.action_t_embed.weight[action_id].data.unsqueeze(0).detach())
            action_embedding = torch.stack(action_embedding_list).mean(dim=0)
            #print (action_embedding)
            #print (len(action_embedding_list))
            self.model.action_t_embed.weight[action_id].data.copy_(action_embedding.squeeze().detach())
        self.model.training_state = 'normal_train'

    def observe(self, task_data, t):

        self.current_task = t

        #if self.current_task == 0:
        #    self.args.max_epoch = 50
        #else:
        #    self.args.max_epoch = 10

        print("========= Current task is {0} ============".format(self.current_task))

        model_state = None

        # self.reset_learning_rate(self.args.lr)
        # reset label smoothing

        # self.reset_learning_rate(self.args.lr)

        previous_examples = []
        for pre_t, examples in self.memory_data.items():
            previous_examples.extend(examples)

        previous_dataset = Dataset(previous_examples)

        last_templates = previous_dataset.class_examples.keys()
        # print (task_data.train.class_examples[0])
        if isinstance(task_data.train.class_examples, tuple):
            current_templates = task_data.train.class_examples[0].keys()
        else:
            current_templates = task_data.train.class_examples.keys()
        print("Previous Task template number is : ", len(last_templates))
        print("Current Task template number is : ", len(current_templates))
        #print("Dataset number is : ", len(task_data.train))
        print("Overlap instance number : ", len(list(set(last_templates) & set(current_templates))))

        train_set = task_data.train

        if task_data.dev:
            dev_set = task_data.dev
        else:
            dev_set = Dataset(examples=[])

        #train_length = int(0.8*len(train_set))
        # dev_length = len(train_set) - train_length
        #shuffle_index_arr = np.arange(len(task_data.train.examples))
        # print (index_arr)
        #np.random.shuffle(shuffle_index_arr)

        # print(batch_ids)
        #task_data.train.examples = [task_data.train.examples[i] for i in shuffle_index_arr]


        #train_set = Dataset(task_data.train.examples[:train_length])
        #dev_set = Dataset(task_data.train.examples[train_length:])
        #self.dev_list.append(dev_set)
        t_offset1, t_offset2 = self.compute_offsets(t)

        nt_offset = self.nt_nc_per_task[t]
        if t != self.old_task:
            if len(self.observed_tasks) > 0 and self.args.ewc:
                self.optimizer.zero_grad()
                old_task = self.observed_tasks[self.old_task]

                p_t_offset1, p_t_offset2 = self.compute_offsets(old_task)

                p_nt_offset = self.nt_nc_per_task[old_task]

                self.reset_label_smoothing(self.model, p_t_offset1, p_t_offset2, p_nt_offset)

                assert old_task != t

                old_replay_batch_examples = self.memory_data[old_task]

                index_arr = np.arange(len(old_replay_batch_examples))

                np.random.shuffle(index_arr)

                old_replay_batch_examples = [old_replay_batch_examples[i] for i in index_arr]


                replay_report_loss, replay_report_examples, replay_loss, non_avg_loss = self.train_iter_process(
                    self.model, old_replay_batch_examples, 0, 0, p_t_offset1,
                    p_t_offset2, p_nt_offset)


                replay_loss.backward()

                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                   self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                                    self.args.clip_grad)
                self.optpar[self.old_task] = {}
                self.fisher[self.old_task] = {}
                for n, p in self.model.named_parameters():
                    if self.args.embed_type == 'glove':
                        pd = p.data.clone()
                        pg = p.grad.data.clone().pow(2)
                        self.optpar[self.old_task][n] = pd
                        self.fisher[self.old_task][n] = pg
                    elif self.args.embed_type == 'bert' or self.args.embed_type == 'bert-mini':
                        if not (n.startswith('src_embed') or n.startswith('plmm_model.pooler.dense')):
                            pd = p.data.clone()
                            pg = p.grad.data.clone().pow(2)
                            self.optpar[self.old_task][n] = pd
                            self.fisher[self.old_task][n] = pg
                    else:
                        if not (n.startswith('src_embed') or n.startswith('plmm_model')):
                            # print(n)
                            pd = p.data.clone()
                            pg = p.grad.data.clone().pow(2)
                            self.optpar[self.old_task][n] = pd
                            self.fisher[self.old_task][n] = pg
                    #pd = p.data.clone()
                    #pg = p.grad.data.clone().pow(2)
                    #self.optpar[self.old_task].append(pd)
                    #self.fisher[self.old_task].append(pg)
            if len(self.observed_tasks) > 0 and self.args.mask_action:
                self.init_action_embedding(train_set.examples)
            self.observed_tasks.append(t)
            self.old_task = t

            # alpha = 1/(len(self.observed_tasks) + 1)
        if self.args.mask_action:
            self.model.training_state = 'mask_training'

        self.n_memories = int(self.args.num_exemplars_per_task)
            #int(len(train_set) * self.args.num_exemplars_ratio)

        self.memory_data[t] = []

        # reset memory buffer
        self.memory_buffer = [None] * self.args.num_memory_buffer

        vote_index_list = []
        """
        if t in self.memory_data:
            print ("T in memory data")
            self.memory_data[t].add(train_set.random_sample_batch_iter(n_memories))
        else:
            self.memory_data[t] = Dataset(train_set.random_sample_batch_iter(n_memories))
        """
        print("setting model to training mode")
        self.model.train()

        if self.args.init_proto and len(self.observed_tasks) > 1:
            self.init_action_embedding(train_set.examples)

        self.model.training_state = 'init_train'

        if not self.args.initial_epoch == 0:
            for name, param in self.model.named_parameters():
                #print (name)
                if not (name == "action_nt_embed.weight" or name == "action_t_embed.weight"):
                    param.requires_grad = False

            epoch = train_iter = 0
            report_loss = report_examples = 0.
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
                    if self.args.embed_fixed and self.src_ids:
                        self.model.src_embed.weight.grad[self.model.new_long_tensor(self.src_ids)] = 0

                    nt_action_ids = []
                    t_action_ids = []
                    for action in self.model.known_action_set:
                        if isinstance(action, GenNTAction):
                            nt_action_ids.append(self.model.nt_action_vocab[action])
                        elif isinstance(action, GenTAction):
                            t_action_ids.append(self.model.t_action_vocab[action])

                    self.model.action_nt_embed.weight.grad[self.model.new_long_tensor(nt_action_ids)] = 0

                    self.model.action_t_embed.weight.grad[self.model.new_long_tensor(t_action_ids)] = 0

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

                if epoch == self.args.initial_epoch:
                    print('reached max initial epoch, stop!', file=sys.stderr)
                    break
            for name, param in self.model.named_parameters():
                param.requires_grad = True
        # now compute the grad on the current minibatch

        # if self.memx is None:
        #    self.memx = x.data.clone()
        #    self.memy = y.data.clone()
        # else:
        #    self.memx = torch.cat((self.memx, x.data.clone()))
        #    self.memy = torch.cat((self.memy, y.data.clone()))

        #known_actions = self.model.known_action_set.copy()

        known_actions = set()
        for action in self.model.known_action_set:
            known_actions.add(action)

        self.model.training_state = 'normal_train'

        print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
        print('vocab: %s' % repr(self.vocab), file=sys.stderr)

        epoch = train_iter = 0
        report_loss = report_examples = replay_report_loss = replay_report_examples = 0.
        history_dev_scores = []
        num_trial = patience = 0


        gss_sum_G = 0
        temp_action_set = set()
        temp_nt_weight = None
        temp_t_weight = None
        while True:
            epoch += 1
            epoch_begin = time.time()
        # train_set = Dataset(train_set.examples[:300])

            for batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=True):

                if self.args.init_proto:
                    if train_iter%2==0:
                        proto_examples = train_set.random_sample_batch_iter(self.n_memories)
                        temp_nt_weight = self.model.action_nt_embed.weight.data.clone().detach()
                        temp_t_weight = self.model.action_t_embed.weight.data.clone().detach()
                        for e in proto_examples:
                            for action in e.tgt_actions:
                                temp_action_set.add(action)

                        self.init_action_embedding(proto_examples,init_all=True)
                    else:
                        if temp_action_set:
                            for action in temp_action_set:
                                if isinstance(action, GenNTAction):
                                    action_id = self.model.nt_action_vocab[action]
                                    self.model.action_nt_embed.weight[action_id].data.copy_(
                                        temp_nt_weight[action_id].squeeze())
                                elif isinstance(action, GenTAction):
                                    action_id = self.model.t_action_vocab[action]
                                    self.model.action_t_embed.weight[action_id].data.copy_(
                                        temp_t_weight[action_id].squeeze())
                        temp_action_set = set()
                        temp_nt_weight = None
                        temp_t_weight = None

            # set index

            # bsz = len(batch_examples)

            # cendcnt = min(self.mem_cnt + bsz, self.args.num_memory_buffer)

            # effbsz = endcnt - self.mem_cnt

            # self.memory_buffer[self.mem_cnt: endcnt] = batch_examples[:effbsz]
            # self.mem_cnt += effbsz


            # print(offset1)
            # print(offset2)
            # for i in range(10):
                self.reset_label_smoothing(self.model, t_offset1, t_offset2, nt_offset)
                train_iter += 1
                self.optimizer.zero_grad()

                report_loss, report_examples, loss, non_avg_loss = self.train_iter_process(self.model ,batch_examples, report_loss,
                                                                             report_examples, t_offset1, t_offset2,
                                                                             nt_offset)
                #if len(self.observed_tasks) == 2 or len(self.observed_tasks) == 4:
                    #loss = 0.9 * loss
                # loss = alpha * loss
                """
                loss.backward()

                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(), self.args.clip_grad)

                if self.args.embed_fixed and self.src_ids:
                    self.model.src_embed.weight.grad[self.model.new_long_tensor(self.src_ids)] = 0

                self.optimizer.step()
                """


                if len(self.observed_tasks) > 1:

                    self.model.training_state = 'normal_train'

                    for tt in range(len(self.observed_tasks) - 1):
                        # self.optimizer.zero_grad()
                        # fwd/bwd on the examples in the memory
                        past_task = self.observed_tasks[tt]

                        p_t_offset1, p_t_offset2 = self.compute_offsets(past_task)

                        p_nt_offset = self.nt_nc_per_task[past_task]

                        self.reset_label_smoothing(self.model, p_t_offset1, p_t_offset2, p_nt_offset)

                        assert past_task != t

                        replay_batch_examples = self.memory_data[past_task]
                        replay_dataset = Dataset(replay_batch_examples)

                        replay_batch_examples = replay_dataset.random_sample_batch_iter(self.batch_size)
                        # index_arr = np.arange(len(replay_batch_examples))
                        # print (index_arr)
                        #np.random.shuffle(index_arr)

                        # print(batch_ids)
                        # replay_batch_examples = [replay_batch_examples[i] for i in index_arr]

                        # gss_examples = self.memory_data[t]

                        # random_examples = self.memory_random_data[past_task]

                        #gss_G = self.get_complete_G(self.model, replay_batch_examples, t_offset1=p_t_offset1, t_offset2=p_t_offset2, nt_offset=p_nt_offset)

                        # random_G = self.get_complete_G(self.model, random_examples, t_offset1=p_t_offset1, t_offset2=p_t_offset2, nt_offset=p_nt_offset)

                        # print ("GSS G")
                        # print(gss_G.sum().item())

                        #gss_sum_G += gss_G.sum().item()

                        # print ("random G")
                        # print(random_G.sum().item())
                        # if t % 2 == 0:
                        # 	replay_batch_examples = random_examples
                        # else:
                        #	replay_batch_examples = gss_examples

                        #if self.args.augment:
                            #print (replay_batch_examples)
                            #print ("====================")
                            #replay_batch_examples = generate_concat_samples(replay_batch_examples, replay_batch_examples)


                        if self.args.rebalance:
                            ori_sample_size = len(replay_batch_examples)
                            rebalance_data = self.rebalanced_memory_data[past_task]

                            replay_batch_examples = rebalance_data.random_sample_batch_iter(ori_sample_size)

                        replay_report_loss, replay_report_examples, replay_loss, non_avg_loss = self.train_iter_process(
                            self.model, replay_batch_examples, replay_report_loss, replay_report_examples, p_t_offset1,
                            p_t_offset2, p_nt_offset)

                        # if tt < len(self.observed_tasks) - 2:
                        # replay_loss.backward(retain_graph=True)
                        # else:
                        # if len(self.observed_tasks) == 2 or len(self.observed_tasks) == 4:
                        #	replay_loss = 0.8*replay_loss
                        # replay_loss = (1 - alpha) * replay_loss
                        #replay_loss.backward()
                        loss += replay_loss
                    if self.args.mask_action:
                        self.model.training_state = 'mask_training'
                if self.args.ewc and len(self.observed_tasks) > 1:
                    reg_loss = 0
                    for tt in range(len(self.observed_tasks) - 1):
                        for n, p in self.model.named_parameters():
                            if self.args.embed_type == 'glove':
                                l = self.fisher[tt][n]
                                l = l * (p - self.optpar[tt][n]).pow(2)
                                reg_loss += l.sum()
                            elif self.args.embed_type == 'bert' or self.args.embed_type == 'bert-mini':
                                if not (n.startswith('src_embed') or n.startswith('plmm_model.pooler.dense')):
                                    l = self.fisher[tt][n]
                                    l = l * (p - self.optpar[tt][n]).pow(2)
                                    reg_loss += l.sum()
                            else:
                                if not (n.startswith('src_embed') or n.startswith('plmm_model')):
                                    l = self.fisher[tt][n]
                                    l = l * (p - self.optpar[tt][n]).pow(2)
                                    reg_loss += l.sum()

                        """
                        for i, p in enumerate(self.model.parameters()):
                            l = self.fisher[tt][i]
                            l = l * (p - self.optpar[tt][i]).pow(2)
                            reg_loss += l.sum()
                        """
                    last_t_offset1, last_t_offset2 = self.compute_offsets(t - 1)

                    last_nt_offset = self.nt_nc_per_task[t - 1]

                    current_vocab_size = t_offset2 - t_offset1 + nt_offset - last_nt_offset

                    previous_vocab_size = last_t_offset2 + last_nt_offset

                    adap_reg_ratio = math.sqrt(previous_vocab_size/current_vocab_size)

                    #print (adap_reg_ratio)

                    loss = loss + self.args.reg * adap_reg_ratio * reg_loss
                loss.backward()

                if self.args.clip_grad > 0.:
                    if self.args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                   self.args.clip_grad)
                    elif self.args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                                    self.args.clip_grad)
                if self.args.embed_fixed and self.src_ids:
                    self.model.src_embed.weight.grad[self.model.new_long_tensor(self.src_ids)] = 0

                self.optimizer.step()

                if train_iter % self.args.log_every == 0:
                    log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                    #print (len(self.observed_tasks))
                    if len(self.observed_tasks) > 1:
                        log_str += '[Iter %d] replay encoder loss=%.5f' % (
                            train_iter, replay_report_loss / replay_report_examples)

                    print(log_str, file=sys.stderr)
                    report_loss = report_examples = replay_report_loss = replay_report_examples = 0.


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

        if self.args.init_proto and temp_action_set:
            for action in temp_action_set:
                if isinstance(action, GenNTAction):
                    action_id = self.model.nt_action_vocab[action]
                    self.model.action_nt_embed.weight[action_id].data.copy_(
                        temp_nt_weight[action_id].squeeze())
                elif isinstance(action, GenTAction):
                    action_id = self.model.t_action_vocab[action]
                    self.model.action_t_embed.weight[action_id].data.copy_(
                        temp_t_weight[action_id].squeeze())


        # intersect_list = set.intersection(*map(set,vote_index_list))
        # print ("intersect list")
        # print (intersect_list)
        # pruned_train_set = Dataset([train_set.examples[i] for i in list(intersect_list)])
        print ("gss sum G")
        print (gss_sum_G)
        # print ("after multiple vote, remaining :", len(pruned_train_set.examples))
        if self.args.sample_method == 'random':
            pruned_train_set = train_set
            if self.args.subselect and len(self.observed_tasks) > 1:
                pruned_train_set = self.subselect_examples(self.model, t_offset1, t_offset2, nt_offset,
                                                           train_set, t)
            self.memory_data[t] = pruned_train_set.random_sample_batch_iter(self.n_memories)
            # for e in self.memory_data[t]:
                #print (e.tgt_actions)
            action_set = set()

            for e in self.memory_data[t]:
                # print (set(e.tgt_actions))
                action_set = action_set.union(set(e.tgt_actions))
            print ("action length")
            print(len(action_set))
            # print(action_set)

        elif self.args.sample_method == 'label_uni':
            label_examples = []
            for l, examples in train_set.class_examples.items():
                label_examples.extend(list(set(examples)))
            self.memory_data[t] = Dataset(label_examples).random_sample_batch_iter(self.n_memories)
        # for e in self.memory_data[t]:
        elif self.args.sample_method.startswith('graph_clustering'):
            pruned_train_set = train_set
            if self.args.subselect and len(self.observed_tasks) > 1:
                pruned_train_set = self.subselect_examples(self.model, t_offset1, t_offset2, nt_offset,
                                                           train_set, t)

            self.memory_data[t] = self.graph_clustering(pruned_train_set, self.n_memories, known_actions)

        elif self.args.sample_method == 'IQP_graph':
            final_examples, inds = self.IQP_graph(train_set, self.n_memories)
            self.memory_data[t] = final_examples

        elif self.args.sample_method == 'kmedoids_graph':
            final_examples = self.kmedoids_graph(train_set, self.n_memories)
            self.memory_data[t] = final_examples

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

            template_count = Counter(
                [e.tgt_ast_t_seq.to_lambda_expr if e.tgt_ast_t_seq.to_lambda_expr else e.to_logic_form for e in
                 task_data.train.examples])

            instance_count = [template_count[e.tgt_ast_t_seq.to_lambda_expr] for e in temp_examples]
            # print(template_count.values())
            # print(instance_count)

            self.memory_data[t] = temp_examples

            self.train()
        elif self.args.sample_method == 'ce_slot':
            self.eval()

            self.reset_label_smoothing(self.model, t_offset1, t_offset2, nt_offset)

            self.model.t_label_smoothing_layer = None


            self.model.nt_label_smoothing_layer = None

            ce_values = torch.zeros(len(train_set))

            slot_values = torch.zeros(len(train_set))

            for idx, example in enumerate(train_set.examples):
                self.model.zero_grad()
                slot_num = 0
                _, _, ptloss, non_avg_loss = self.train_iter_process(self.model, [example], 0, 0, t_offset1,
                                                                     t_offset2,
                                                                     nt_offset)

                # print (self.query_vector)

                ce_values[idx] = ptloss.detach().item()/len(example.tgt_actions)
                #print (ptloss.detach().item())
                #print (ce_values[idx])
                for action in example.tgt_actions:
                    if isinstance(action, GenTAction):
                        if action in self.model.t_action_vocab.action2nl:
                            slot_num += 1
                    elif isinstance(action, GenNTAction):
                        if action in self.model.nt_action_vocab.action2nl:
                            slot_num += 1
                #print (slot_num)
                slot_values[idx] = math.pow(slot_num, 0.5)
            # entropy = nt_entropy + t_entropy


            prior_index_tensor = ce_values*slot_values

            prior_index = [i[0] for i in sorted(enumerate(prior_index_tensor.tolist()), key=lambda x: x[1])]


            #print (prior_index)

            temp_examples = self.arper_select(train_set, prior_index)

            temp_examples = list(set(temp_examples))
            print(len(temp_examples))
            assert len(temp_examples) == self.n_memories

            self.memory_data[t] = temp_examples
            self.reset_label_smoothing(self.model, t_offset1, t_offset2, nt_offset)
            self.train()
        elif self.args.sample_method == 'f_cluster':
            self.eval()

            feature_vectors = torch.zeros(len(train_set), 2*self.args.hidden_size)
            beg = end = 0
            for feature_batch_examples in train_set.batch_iter(batch_size=self.batch_size, shuffle=False, sort=False):
                batch = Batch(feature_batch_examples, self.vocab, use_cuda=self.use_cuda)
                src_sents_var = batch.src_sents_var
                src_sents_len = batch.src_sents_len
                src_encodings, (last_state, last_cell) = self.model.encode(src_sents_var, src_sents_len)
                #dec_init_vec = self.model.init_decoder_state(last_state, last_cell)

                end += len(feature_batch_examples)

                # print (beg)
                # print (end)
                # print (replay_loss)

                feature_vectors[beg:end, :].copy_(last_cell.detach())

                beg += len(feature_batch_examples)
            assert end == len(train_set)
            X = feature_vectors.cpu().numpy()

            km = KMeans(n_clusters=self.n_memories, random_state=self.args.p_seed).fit(X)

            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)

            temp_examples = []

            for top_idx in closest.tolist():
                temp_examples.append(train_set.examples[top_idx])

            assert len(temp_examples) == self.n_memories

            self.memory_data[t] = temp_examples

            self.train()
        elif self.args.sample_method == 'gss_cluster':

            gss_size = self.model.att_vec_linear.weight.data.numel()

            feature_vectors = torch.zeros(len(train_set), gss_size)

            self.reset_label_smoothing(self.model, t_offset1, t_offset2, nt_offset)

            for idx, example in enumerate(train_set.examples):
                self.model.zero_grad()

                _, _, ptloss, non_avg_loss = self.train_iter_process(self.model, [example], 0, 0, t_offset1,
                                                                     t_offset2,
                                                                     nt_offset)

                ptloss.backward()

                # print (self.query_vector)

                feature_vectors[idx] = self.model.att_vec_linear.weight.grad.data.view(-1).detach()

            X = feature_vectors.cpu().numpy()

            km = KMeans(n_clusters=self.n_memories).fit(X)

            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, X)

            temp_examples = []
            print (closest.tolist())
            for top_idx in closest.tolist():
                temp_examples.append(train_set.examples[top_idx])

            assert len(temp_examples) == self.n_memories

            self.memory_data[t] = temp_examples
        elif self.args.sample_method == 'balance':
            self.memory_data[t] = sample_balance_data(train_set.examples, self.n_memories)
        elif self.args.sample_method == 'greedy_uniform':
            pruned_train_set = train_set
            if self.args.subselect and len(self.observed_tasks) > 1:
                pruned_train_set = self.subselect_examples(self.model, t_offset1, t_offset2, nt_offset,
                                                           train_set, t)

            self.memory_data[t] = self.greedy_uniform_sample(pruned_train_set, self.n_memories)
        elif self.args.sample_method == 'IQP_uniform':
            self.memory_data[t] = self.IQP_uniform_sample(train_set.examples, self.n_memories)
        else:
            final_examples, inds = self.gss_select_examples(self.model, t_offset1, t_offset2, nt_offset, t, train_set,
                                                      nb_selected=self.n_memories, smooth=self.args.reg)
            print (inds)
            # selection_model.load_state_dict(model_state)
            begin_time = time.time()


            self.memory_data[t] = final_examples

            action_set = set()

            for e in self.memory_data[t]:
                action_set = action_set.union(set(e.tgt_actions))
            print ("action length")
            print(len(action_set))
            # self.memory_random_data[t] = train_set.random_sample_batch_iter(self.n_memories)

            print("Final selected instances number is {0}".format(len(self.memory_data[t])), file=sys.stderr)


            self.mem_grads = None
            print("Total selection cost time is {0}".format(time.time() - begin_time), file=sys.stderr)
        if self.args.rebalance:
            self.rebalanced_memory_data[t] = Dataset(reweight_t_data(self.memory_data[t]))


        """
        if self.args.second_iter:
            action_set = set()
            for e in train_set.examples:
                for action in e.tgt_actions:
                    action_set.add(action)
            for action in list(action_set):
                if isinstance(action, GenNTAction):
                    action_id = self.model.nt_action_vocab[action]
                    if action in self.nt_action_embed:
                        self.nt_action_embed[action].append(self.model.action_nt_embed.weight[action_id].data.detach())
                    else:
                        self.nt_action_embed[action] = []
                        self.nt_action_embed[action].append(self.model.action_nt_embed.weight[action_id].data.detach())
                elif isinstance(action, GenTAction):
                    action_id = self.model.t_action_vocab[action]
                    if action in self.t_action_embed:
                        self.t_action_embed[action].append(self.model.action_t_embed.weight[action_id].data.detach())
                    else:
                        self.t_action_embed[action] = []
                        self.t_action_embed[action].append(self.model.action_t_embed.weight[action_id].data.detach())

            if self.current_task == self.n_tasks - 1:
                self.model.training_state = 'recon_train'
                for action, action_embedding_list in self.nt_action_embed.items():
                    # print (action)
                    action_id = self.model.nt_action_vocab[action]
                    # print (action_id)
                    #print (len(action_embedding_list))
                    # print (torch.stack(action_embedding_list).size())
                    # action_embedding_list.append(self.model.action_nt_embed.weight[action_id].data.unsqueeze(0).detach())
                    action_embedding = torch.stack(action_embedding_list).mean(dim=0)
                    # print(action_embedding)
                    self.model.nt_stable_embed.weight[action_id].data.copy_(action_embedding.squeeze().detach())

                for action, action_embedding_list in self.t_action_embed.items():
                    action_id = self.model.t_action_vocab[action]
                    # action_embedding_list.append(self.model.action_t_embed.weight[action_id].data.unsqueeze(0).detach())
                    action_embedding = torch.stack(action_embedding_list).mean(dim=0)
                    # print (action_embedding)
                    #print (len(action_embedding_list))
                    self.model.t_stable_embed.weight[action_id].data.copy_(action_embedding.squeeze().detach())


                meta_train_iter = 0
                for i in range(self.args.second_iter):
                    meta_total_loss = meta_report_loss = meta_report_examples = 0.
                    self.optimizer.zero_grad()
                    for tt in range(len(self.observed_tasks) - 1):
                        meta_train_iter += 1
                        past_task = self.observed_tasks[tt]

                        p_t_offset1, p_t_offset2 = self.compute_offsets(past_task)

                        p_nt_offset = self.nt_nc_per_task[past_task]

                        self.reset_label_smoothing(self.model, p_t_offset1, p_t_offset2, p_nt_offset)

                        replay_batch_examples = self.memory_data[past_task]
                        replay_dataset = Dataset(replay_batch_examples)

                        replay_batch_examples = replay_dataset.random_sample_batch_iter(self.batch_size)

                        meta_report_loss, meta_report_examples, meta_loss, non_avg_loss = self.train_iter_process(
                            self.model, replay_batch_examples, meta_report_loss, meta_report_examples, p_t_offset1,
                            p_t_offset2, p_nt_offset)

                        meta_total_loss += meta_loss

                        log_str = '[Reconsolidation Train Iter %d] recons loss=%.5f' % (
                            meta_train_iter, meta_report_loss / meta_report_examples)

                        print(log_str, file=sys.stderr)
                        meta_report_loss = meta_report_examples = 0.

                    meta_total_loss.backward()

                    if self.args.clip_grad > 0.:
                        if self.args.clip_grad_mode == 'norm':
                            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                       self.args.clip_grad)
                        elif self.args.clip_grad_mode == 'value':
                            grad_norm = torch.nn.utils.clip_grad_value_(self.model.parameters(),
                                                                        self.args.clip_grad)

                    self.optimizer.step()

                #self.model.action_nt_embed.weight.data.copy_((self.model.action_nt_embed.weight.data + self.model.nt_stable_embed.weight.data)/2)
                #self.model.action_t_embed.weight.data.copy_(
                #    (self.model.action_t_embed.weight.data + self.model.t_stable_embed.weight.data) / 2)
        """
        model_file = self.args.save_to + '.bin'
        print('save the current model ..', file=sys.stderr)
        print('save model to [%s]' % model_file, file=sys.stderr)
        self.model.save(model_file)
        #loaded_model = self.args.save_to + '.bin'
        #print('load model from [%s]' % loaded_model, file=sys.stderr)
        #params = torch.load(self.args.save_to + '.bin', map_location=lambda storage, loc: storage)
        #self.model.load_state_dict(params['state_dict'])
        #if self.args.use_cuda: self.model = self.model.cuda()

        self.mem_cnt = 0

    def arper_select(self, train_set, prior_index):
        temp_examples = []
        slelected_example = set()
        while len(temp_examples) < self.n_memories:
            slot_set = set()
            for ce_top_idx in prior_index:
                e = train_set.examples[ce_top_idx]
                if e in slelected_example:
                    continue
                slot_str = e.tgt_ast_t_seq.to_lambda_expr
                #print (slot_str)
                if slot_str in slot_set:
                    continue
                else:
                    slot_set.add(slot_str)
                temp_examples.append(e)
                slelected_example.add(e)

                if len(temp_examples) == self.n_memories:
                    return temp_examples

    def IQP_graph(self, train_set, n_memories):
        actions = []

        for e in train_set.examples:
            for action in e.tgt_actions:
                actions.append(action)

        action_to_id = {token: idx for idx, token in enumerate(set(actions))}

        # convert token lists to token-id lists, e.g. [[1, 2], [2, 2]] here
        # action_ids = [[action_to_id[action] for action in e.tgt_actions] for e in examples]

        data = torch.zeros(len(train_set.examples), len(action_to_id))
        # [[0 for i in range(len(action_to_id))] for j in range(len(examples))]
        # data = [[0]*len(action_to_id)]*len(examples)
        # print(len(data))
        # print(len(data[0]))
        for idx, e in enumerate(train_set.examples):
            # print("==================================================")
            # print (data[idx])
            for action in e.tgt_actions:
                # print (action_to_id[action])
                data[idx][action_to_id[action]] = 1

        sim_array = get_G(data, data)
        #print (sim_array)
        begin_time = time.time()

        added_inds = cosine_similarity_selector_IQP_Exact(sim_array,
                                                          nb_selected=n_memories,
                                                          solver=self.solver, sample_method='init_random')

        added_inds_list = added_inds.squeeze().tolist()

        selected_examples = [train_set.examples[indx] for indx in added_inds_list]

        print("Individual selection cost time {0}".format(time.time() - begin_time), file=sys.stderr)

        return selected_examples, added_inds_list


    def kmedoids_graph(self, train_set, n_memories):
        smatch = SmatchEvaluator(self.args)

        sim_array = np.ndarray((len(train_set.examples), len(train_set.examples)))
        triple_list = []
        for e in train_set.examples:
            if self.args.lang == "nlmap":
                node = parse_nlmap_query_helper(e.to_logic_form.split(' '))
            elif self.args.lang == "overnight_lambda":
                node = parse_overnight_query_helper(e.to_logic_form.split(' '))
            triple_set = smatch.get_triple_set(node)
            triple_list.append(triple_set)

        for out_idx, out_triple_set in enumerate(triple_list):
            for in_idx, in_triple_set in enumerate(triple_list):
                p, r, f_l = smatch.smatch_score(out_triple_set, in_triple_set)
                p, r, f_r = smatch.smatch_score(in_triple_set, out_triple_set)
                #print(f)
                sim_array[out_idx, in_idx] = (f_l + f_r)/2

        print ("kmedoids graph sampling")

        kmedoids = KMedoids(metric='precomputed', n_clusters=n_memories, init='k-medoids++').fit((1 - sim_array))
        #print (kmedoids.medoid_indices_)
        added_inds = kmedoids.medoid_indices_

        added_inds_list = added_inds.squeeze().tolist()

        selected_examples = [train_set.examples[indx] for indx in added_inds_list]

        return selected_examples

    def graph_clustering(self, train_set, n_memories, known_actions):
        sim_array = np.ndarray((len(train_set.examples), len(train_set.examples)))
        """
        if self.args.sample_method == 'graph_clustering_edit':
            tree_list = []
            for e in train_set.examples:
                if self.args.lang == "nlmap":
                    node = parse_nlmap_query_helper(e.to_logic_form.split(' '))
                elif self.args.lang == "overnight_lambda":
                    node = parse_overnight_query_helper(e.to_logic_form.split(' '))
                edit_node = ori_tree2edit_tree(node)
                tree_list.append(edit_node)

            for out_idx, out_tree in enumerate(tree_list):
                for in_idx, in_tree in enumerate(tree_list):
                    if in_idx >= out_idx:
                        if in_idx == out_idx:
                            sim_array[out_idx, in_idx] = 0
                        else:
                            dis_out_in = simple_distance(out_tree, in_tree)
                            #dis_in_out = simple_distance(in_tree, out_tree)
                            print(out_idx)
                            print(in_idx)
                            sim_array[out_idx, in_idx] = dis_out_in
                            sim_array[in_idx, out_idx] = dis_out_in
            check_symmetric(sim_array)
            print (sim_array)
            sim_array = np.exp(- sim_array ** 2 / (2. * 1 ** 2))
            print(sim_array)

        elif self.args.sample_method == 'graph_clustering_smatch':
        """
        smatch = SmatchEvaluator(self.args)

        triple_list = []
        for e in train_set.examples:
            if self.args.lang == "nlmap":
                node = parse_nlmap_query_helper(e.to_logic_form.split(' '))
            elif self.args.lang == "overnight_lambda":
                node = parse_overnight_query_helper(e.to_logic_form.split(' '))
            triple_set = smatch.get_triple_set(node)
            triple_list.append(triple_set)

        for out_idx, out_triple_set in enumerate(triple_list):
            for in_idx, in_triple_set in enumerate(triple_list):
                p, r, f_l = smatch.smatch_score(out_triple_set, in_triple_set)
                p, r, f_r = smatch.smatch_score(in_triple_set, out_triple_set)
                #print(f)
                sim_array[out_idx, in_idx] = (f_l + f_r)/2

        # print (sim_array)
        sc = SpectralClustering(n_memories, affinity='precomputed', random_state=self.args.p_seed)
        sc.fit(sim_array)

        labels = sc.labels_.tolist()
        # print (labels)
        index_list = [None]*n_memories

        for idx, e in enumerate(train_set.examples):
            label = labels[idx]
            index_list[label] = idx

        # filter actions
        filtered_action_set = set()
        action_freq_dict = {}
        action_len = 0
        for e in train_set.examples:
            action_len += len(list(set(e.tgt_actions)))
            for action in list(set(e.tgt_actions)):
                if action in action_freq_dict:
                    action_freq_dict[action] = action_freq_dict[action] + 1
                else:
                    action_freq_dict[action] = 1

        action_list = list(action_freq_dict.keys())
        action_freq_list = [action_freq_dict[action] for action in action_list]

        avg_action_len = action_len / len(train_set.examples)
        sample_action_num = self.args.action_num
        if self.args.clustering_filter == 3:
            action_freq_nparray = np.array(action_freq_list)
            # print (action_freq_nparray)
            action_prob = action_freq_nparray / action_freq_nparray.sum()
            selected_inds = np.random.choice(len(action_list), sample_action_num, replace=False, p=action_prob)

            for filter_index in selected_inds.tolist():
                filtered_action_set.add(action_list[filter_index])
        elif self.args.clustering_filter == 4:
            selected_inds = sorted(range(len(action_freq_list)), key=action_freq_list.__getitem__, reverse=True)[
                            :sample_action_num]
            for filter_index in selected_inds:
                filtered_action_set.add(action_list[filter_index])

        actions = []

        for e in train_set.examples:
            for action in e.tgt_actions:
                if self.args.clustering_filter == 1:
                    if isinstance(action, GenNTAction):
                        actions.append(action)
                elif self.args.clustering_filter == 2:
                    if isinstance(action, GenTAction):
                        actions.append(action)
                elif self.args.clustering_filter == 3 or self.args.clustering_filter == 4:
                    if action in filtered_action_set:
                        actions.append(action)
                elif self.args.clustering_filter == 5:
                    random_num = np.random.uniform()
                    if random_num < self.args.action_dropout:
                        actions.append(action)
                elif self.args.clustering_filter == 6:
                    if not (action in known_actions):
                        actions.append(action)
                else:
                    actions.append(action)
        action_to_id = {token: idx for idx, token in enumerate(set(actions))}

        # convert token lists to token-id lists, e.g. [[1, 2], [2, 2]] here
        # action_ids = [[action_to_id[action] for action in e.tgt_actions] for e in examples]

        data = torch.zeros(len(train_set.examples), len(action_to_id))
        # [[0 for i in range(len(action_to_id))] for j in range(len(examples))]
        # data = [[0]*len(action_to_id)]*len(examples)
        # print(len(data))
        # print(len(data[0]))
        for idx, e in enumerate(train_set.examples):
            # print("==================================================")
            # print (data[idx])
            for action in e.tgt_actions:
                # print (action_to_id[action])
                if action in action_to_id:
                    data[idx][action_to_id[action]] = 1
            # print(idx)
            # print(data[idx])

        freq_m = data[index_list].clone().detach()
        print (freq_m.sum(dim=0))
        print (freq_m.sum(dim=0).size())
        freq_sum = freq_m.sum(dim=0)
        freq_prob = freq_sum / freq_m.sum()
        current_entropy = Categorical(probs=freq_prob).entropy()
        print (current_entropy)
        print ("===========")

        added_inds_list = index_list
        entropy_sum = 1000
        while entropy_sum > 0:

            entropy_sum = 0
            for label_index in range(n_memories):
                entropy_tensor = []
                entropy_tensor_indx = []
                freq_list = []
                for train_idx in range(len(train_set.examples)):
                    if labels[train_idx] == label_index:
                        freq_sum = freq_m.sum(dim=0)
                        freq_prob = freq_sum/freq_m.sum()
                        current_entropy = Categorical(probs=freq_prob).entropy()
                        example_freq = data[train_idx]

                        # entropy_tensor = torch.zeros(n_memories)

                        temp_freq_sum = freq_sum - freq_m[label_index] + example_freq
                        temp_freq_prob = temp_freq_sum / temp_freq_sum.sum()
                        temp_entropy = Categorical(probs=temp_freq_prob).entropy()
                        entropy_tensor.append(temp_entropy - current_entropy)
                        entropy_tensor_indx.append(train_idx)
                        freq_list.append(example_freq)

                entropy_tensor = torch.Tensor(entropy_tensor)
                max_entropy, max_entropy_ind = torch.max(entropy_tensor,dim=-1)
                #print (max_entropy_ind.item())
                max_selected_indx = entropy_tensor_indx[max_entropy_ind.item()]
                if entropy_tensor[max_entropy_ind.item()].item() > 0 and not (max_selected_indx in added_inds_list):
                    added_inds_list[label_index] = max_selected_indx
                    freq_m[label_index] = freq_list[max_entropy_ind.item()]
                    entropy_sum += entropy_tensor[max_entropy_ind.item()].item()



            print(entropy_sum)

        # break
        # print (data[idx])
        # print (data)
        # print (type(train_set))
        # print (len(data[0]))
        #added_inds_list = select_uniform_data(nb_selected, data)
        print (added_inds_list)
        freq_m = data[added_inds_list].clone().detach()
        print(freq_m.sum(dim=0))
        print (freq_m.sum(dim=0).size())
        freq_sum = freq_m.sum(dim=0)
        freq_prob = freq_sum / freq_m.sum()
        current_entropy = Categorical(probs=freq_prob).entropy()
        print (current_entropy)
        selected_examples = [train_set.examples[indx] for indx in added_inds_list]

        return selected_examples


    def gss_select_examples(self, selection_model, t_offset1, t_offset2, nt_offset, t, train_set, nb_selected, smooth=0):

        pruned_train_set = train_set
        cosine_G = None
        if self.args.subselect and len(self.observed_tasks) > 1:
            cosine_G = torch.zeros(len(train_set))
            pruned_train_set = self.subselect_examples(selection_model, t_offset1, t_offset2, nt_offset, train_set, t, cos_G=cosine_G)
        # debug
        # pruned_train_set = Dataset(pruned_train_set.examples[:201])
        """
        while len(pruned_train_set.examples) > self.args.num_memory_buffer:

            temp_examples = []

            for batch_examples in pruned_train_set.batch_iter(batch_size=self.args.num_memory_buffer, shuffle=False, sort=False):
                temp_examples.extend(self.select_samples_per_group(t, batch_examples, nb_selected = int(len(batch_examples)/2)))

            pruned_train_set = Dataset(temp_examples)
        """


        final_examples, added_inds_list = self.select_samples_per_group(selection_model ,t, pruned_train_set, nb_selected=nb_selected, cos_G=cosine_G, smooth=smooth)

        final_examples.sort(key=lambda e: -len(e.src_sent))

        return final_examples, added_inds_list

    def subselect_examples(self, selection_model, t_offset1, t_offset2, nt_offset, train_set, t, cos_G = None):
        for tt in range(len(self.observed_tasks) - 1):
            selection_model.zero_grad()
            # fwd/bwd on the examples in the memory
            past_task = self.observed_tasks[tt]

            p_t_offset1, p_t_offset2 = self.compute_offsets(past_task)

            p_nt_offset = self.nt_nc_per_task[past_task]

            self.reset_label_smoothing(selection_model, p_t_offset1, p_t_offset2, p_nt_offset)

            memory_examples = self.memory_data[past_task]

            _, _, replay_loss, non_avg_loss = self.train_iter_process(selection_model, memory_examples, 0, 0,
                                                                      p_t_offset1, p_t_offset2,
                                                                      p_nt_offset)

            replay_loss.backward()

            store_grad(selection_model.parameters, self.grads, self.grad_dims,
                       past_task)

        """
        store_grad(self.parameters, self.grads, self.grad_dims, t)
        indx = self.model.new_long_tensor(self.observed_tasks[:-1])

        example_num = sum([len(self.memory_data[t]) for t in self.observed_tasks[:-1]])

        a_g = self.grads.index_select(1, indx).sum(-1)/example_num

        dotp = torch.dot(self.grads[:, t], a_g)
        """

        index_list = []


        idx = 0

        self.reset_label_smoothing(selection_model, t_offset1, t_offset2, nt_offset)

        for example in train_set.examples:
            selection_model.zero_grad()

            _, _, ptloss, non_avg_loss = self.train_iter_process(selection_model, [example], 0, 0, t_offset1, t_offset2,
                                                                 nt_offset)

            ptloss.backward()
            # add the new grad to the memory grads and add it is cosine similarity
            # print (self.grad_dims)
            # copy gradient
            store_grad(selection_model.parameters, self.grads, self.grad_dims, t)
            indx = self.model.new_long_tensor(self.observed_tasks[:-1])
            if self.args.subselect == 1:
                dotp = torch.mm(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx))
                if (dotp < 0).sum() == 0:  # or (dotp > 0.01).sum() == 0:
                    index_list.append(idx)
            # cosine = get_G(self.grads[:, t].unsqueeze(0), self.grads.index_select(1, indx).t())
            elif self.args.subselect == 2:
                example_num = sum([len(self.memory_data[t]) for t in self.observed_tasks[:-1]])
                a_g = self.grads.index_select(1, indx).sum(-1) / example_num
                dotp = torch.dot(self.grads[:, t], a_g)
                if (dotp < 0).sum() == 0:  # or (dotp > 0.01).sum() == 0:
                    index_list.append(idx)
            elif self.args.subselect == 3:
                example_num = sum([len(self.memory_data[t]) for t in self.observed_tasks[:-1]])
                a_g = self.grads.index_select(1, indx).sum(-1) / example_num
                cos_G[idx] = get_G(self.grads[:, t].unsqueeze(0), a_g.unsqueeze(0)).item()
                index_list.append(idx)
            # cosine = get_G(self.grads[:, t].unsqueeze(0), a_g.unsqueeze(0).t())

            # print (self.grads[:, t], a_g)
            # cosine_list.append(cosine.sum().item())
            # print(dotp)
            idx += 1

        # index_list = [i[0] for i in sorted(enumerate(cosine_list), key=lambda x: x[1])]

        print("Original dataset length is {0}".format(len(train_set.examples)), file=sys.stderr)
        pruned_train_set = Dataset([train_set.examples[i] for i in index_list])
        print("Pruned dataset length is {0}".format(len(pruned_train_set.examples)), file=sys.stderr)

        return pruned_train_set

    def get_complete_G(self, model, train_examples, t_offset1, t_offset2, nt_offset):

        train_set = Dataset(train_examples)

        G = torch.cuda.FloatTensor(len(train_set), len(train_set)).fill_(0)

        # get gradients from the ring buffer
        i = out_beg = out_end = 0
        for out_examples in train_set.batch_iter(batch_size=self.args.num_memory_buffer, shuffle=False, sort=False):
            j = in_beg = in_end = 0

            out_end += len(out_examples)

            out_mem_grads = torch.cuda.FloatTensor(len(out_examples), sum(self.grad_dims))


            # out_mask = torch.zeros(len(out_examples), device='cuda')

            # out_weight = 1 / len(out_examples)
            out_idx = 0
            for example in out_examples:
                model.zero_grad()

                # out_mask[out_idx] = out_weight

                _, _, ptloss, non_avg_loss = self.train_iter_process(model, [example], 0, 0, t_offset1,
                                                                     t_offset2,
                                                                     nt_offset)

                ptloss.backward()
                # add the new grad to the memory grads and add it is cosine similarity
                # print (self.grad_dims)
                add_memory_grad(model.parameters, out_mem_grads, self.grad_dims, out_idx)
                out_idx+=1

            for inner_examples in train_set.batch_iter(batch_size=self.args.num_memory_buffer, shuffle=False, sort=False):
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
                            _, _, ptloss, non_avg_loss = self.train_iter_process(model, [example], 0, 0, t_offset1,
                                                                                 t_offset2,
                                                                                 nt_offset)

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
                        torch.cuda.empty_cache()

                    G[out_beg:out_end, in_beg: in_end].copy_(sub_G)
                    G[in_beg: in_end, out_beg:out_end].copy_(sub_G.T)

                in_beg += len(inner_examples)
                j += 1
            out_beg += len(out_examples)
            i += 1
            del out_mem_grads
            torch.cuda.empty_cache()
        model.zero_grad()
        # print (G)
        # print (torch.nonzero((G==0), as_tuple=False).size(0))
        assert torch.nonzero((G==0), as_tuple=False).size(0) == 0

        return G


    def select_samples_per_group(self, model, task, train_set, nb_selected, cos_G = None, smooth = 0):
        """
        Assuming a ring buffer, backup constraints and constrains,
        re-estimate the backup constrains and constrains
        """

        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                module.p = 0

        begin_time = time.time()
        t_offset1, t_offset2 = self.compute_offsets(task)

        nt_offset = self.nt_nc_per_task[task]

        self.reset_label_smoothing(model, t_offset1, t_offset2, nt_offset)

        print("constraints selector")

        G = self.get_complete_G(model, train_set.examples, t_offset1=t_offset1, t_offset2=t_offset2,nt_offset=nt_offset)
        torch.cuda.empty_cache()
        added_inds = cosine_similarity_selector_IQP_Exact(G,
                                                          nb_selected=nb_selected,
                                                          solver=self.solver, sample_method=self.args.sample_method, cos_G=cos_G, smooth = smooth)

        added_inds_list = added_inds.squeeze().tolist()

        selected_examples = [train_set.examples[indx] for indx in added_inds_list]

        print("Individual selection cost time {0}".format(time.time() - begin_time), file=sys.stderr)

        for name, module in model.named_modules():
            # print (name)
            if name == 'dropout':
                module.p = self.args.dropout


        return selected_examples, added_inds_list

    def IQP_uniform_sample(self, examples, nb_selected):

        actions = []

        for e in examples:
            for action in e.tgt_actions:
                actions.append(action)

        action_to_id = {token: idx for idx, token in enumerate(set(actions))}

        # convert token lists to token-id lists, e.g. [[1, 2], [2, 2]] here
        # action_ids = [[action_to_id[action] for action in e.tgt_actions] for e in examples]

        data = [[0 for i in range(len(action_to_id))] for j in range(len(examples))]
        # data = [[0]*len(action_to_id)]*len(examples)
        # print(len(data))
        # print(len(data[0]))
        for idx, e in enumerate(examples):
            # print("==================================================")
            # print (data[idx])
            for action in e.tgt_actions:
                # print (action_to_id[action])
                data[idx][action_to_id[action]] = 1
            # print(idx)
            # print(data[idx])
        # break
        # print (data[idx])
        # print (data)
        # print (type(train_set))
        print (len(data[0]))
        added_inds_list = select_uniform_data(nb_selected, data)
        selected_examples = [examples[indx] for indx in added_inds_list]
        return selected_examples

    def greedy_uniform_sample(self, train_set, nb_selected):


        examples = train_set.examples
        # index_arr = np.arange(len(examples))
        # print (index_arr)

        # np.random.shuffle(index_arr)

        # index_list = index_arr[:nb_selected].tolist()

        index_list = set()
        for l, idx_list in train_set.class_idx.items():
            random_idx = random.randint(0, len(idx_list)-1)
            index_list.add(idx_list[random_idx])
        index_list = list(index_list)

        if len(index_list) > nb_selected:
            index_arr = np.array(index_list)
            np.random.shuffle(index_arr)
            index_list = index_arr[:nb_selected].tolist()
        else:
            for i in range(nb_selected-len(index_list)):
                random_idx = random.randint(0, len(examples) - 1)
                while random_idx in index_list:
                    random_idx = random.randint(0, len(examples) - 1)
                index_list.append(random_idx)

        assert len(set(index_list)) == nb_selected

        actions = []

        for e in examples:
            for action in e.tgt_actions:
                actions.append(action)

        action_to_id = {token: idx for idx, token in enumerate(set(actions))}

        # convert token lists to token-id lists, e.g. [[1, 2], [2, 2]] here
        # action_ids = [[action_to_id[action] for action in e.tgt_actions] for e in examples]

        data = torch.zeros(len(examples), len(action_to_id))
        # [[0 for i in range(len(action_to_id))] for j in range(len(examples))]
        # data = [[0]*len(action_to_id)]*len(examples)
        # print(len(data))
        # print(len(data[0]))
        for idx, e in enumerate(examples):
            # print("==================================================")
            # print (data[idx])
            for action in e.tgt_actions:
                # print (action_to_id[action])
                data[idx][action_to_id[action]] = 1
            # print(idx)
            # print(data[idx])

        freq_m = data[index_list].clone().detach()
        print (freq_m.sum(dim=0))
        print (freq_m.sum(dim=0).size())
        freq_sum = freq_m.sum(dim=0)
        freq_prob = freq_sum / freq_m.sum()
        current_entropy = Categorical(probs=freq_prob).entropy()
        print (current_entropy)
        print ("===========")

        added_inds_list = index_list
        entropy_sum = 1000
        while entropy_sum > 0:
            idx = 0
            entropy_sum = 0
            for e in examples:
                freq_sum = freq_m.sum(dim=0)
                freq_prob = freq_sum/freq_m.sum()
                current_entropy = Categorical(probs=freq_prob).entropy()
                example_freq = data[idx]

                entropy_tensor = torch.zeros(nb_selected)

                # entropy_per_action = torch.zeros(nb_selected)

                for i in range(nb_selected):
                    temp_freq_sum = freq_sum - freq_m[i] + example_freq
                    temp_freq_prob = temp_freq_sum/temp_freq_sum.sum()
                    temp_entropy = Categorical(probs=temp_freq_prob).entropy()
                    entropy_tensor[i] = (temp_entropy - current_entropy)#/len(set(examples[added_inds_list[i]].tgt_actions))
                    # len(set(examples[added_inds_list[max_entropy_ind.item()]].tgt_actions))
                    # /len(set(examples[added_inds_list[i]].tgt_actions))
                    # entropy_per_action[i] = (len(set(e.tgt_actions))/ len(set(examples[added_inds_list[i]].tgt_actions)))*temp_entropy
                #print (entropy_tensor)
                #print (entropy_tensor.size())
                max_entropy, max_entropy_ind = torch.max(entropy_tensor,dim=-1)
                #print (max_entropy_ind.item())
                if entropy_tensor[max_entropy_ind.item()].item() > 0 and not (idx in added_inds_list):
                    added_inds_list[max_entropy_ind.item()] = idx
                    freq_m[max_entropy_ind.item()] = example_freq
                    entropy_sum += entropy_tensor[max_entropy_ind.item()].item()

                idx += 1

            print(entropy_sum)

        # break
        # print (data[idx])
        # print (data)
        # print (type(train_set))
        # print (len(data[0]))
        #added_inds_list = select_uniform_data(nb_selected, data)
        print (added_inds_list)
        freq_m = data[added_inds_list].clone().detach()
        print(freq_m.sum(dim=0))
        print (freq_m.sum(dim=0).size())
        freq_sum = freq_m.sum(dim=0)
        freq_prob = freq_sum / freq_m.sum()
        current_entropy = Categorical(probs=freq_prob).entropy()
        print (current_entropy)
        selected_examples = [examples[indx] for indx in added_inds_list]
        return selected_examples
    # check whether this is the last minibatch of the current task
    # We assume only 1 epoch!
    # if self.examples_seen == self.samples_per_task:
    #    self.examples_seen = 0
    # get labels from previous task; we assume labels are consecutive

    # Reduce exemplar set by updating value of num. exemplars per class
    # self.num_exemplars = int(self.n_memories / (num_classes + len(self.mem_class_x.keys())))

    def forward_with_offset(self, model, t_offset1, t_offset2, nt_offset, batch):
        if self.args.base_model == 'irnet':
            query_vectors = model(batch, t_offset1, t_offset2)
            self.query_vector = query_vectors
            # if model.training_state == 'recon_train':
            #    nt_scores, t_scores = model.action_weight_readout(query_vectors, self.model.nt_stable_embed.weight, self.model.t_stable_embed.weight)
            # else:
            nt_scores, t_scores = model.action_readout(query_vectors)
            t_action_scores = t_scores

            nt_scores = nt_scores[:, :, :nt_offset]
        else:
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