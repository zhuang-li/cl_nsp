# coding=utf-8
from __future__ import print_function

import argparse
import uuid
from datetime import datetime
from itertools import chain

import six.moves.cPickle as pickle
from six.moves import xrange as range
from six.moves import input
import traceback

import numpy as np
import time
import os
import sys
import collections
import torch
from torch.autograd import Variable

from continual_model.utils import confusion_matrix
from model.utils import GloveHelper, merge_vocab_entry
from common.registerable import Registrable
from components.dataset import Dataset, Example, ContinuumDataset
from common.utils import update_args, init_arg_parser
from datasets import *
from model import nn_utils
import evaluation
from components.grammar_validation.asdl import LambdaCalculusTransitionSystem
from components.evaluator import DefaultEvaluator, ActionEvaluator
from continual_model.seq2seq_topdown import Seq2SeqModel


# from model.seq2seq_align import Seq2SeqModel

def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args


def read_domain_data(domains, file_prefix, suffix):
    path_list = []
    for domain in domains:
        path = os.path.join(file_prefix, domain + suffix)
        path_list.append(path)
    return path_list


def read_domain_vocab(domains, file_prefix, suffix):
    assert len(domains) > 0
    vocab = pickle.load(open(os.path.join(file_prefix, domains[0] + suffix), 'rb'))
    for domain in domains[1:]:
        vocab_after = pickle.load(open(os.path.join(file_prefix, domain + suffix), 'rb'))
        vocab = merge_vocab_entry(vocab_after, vocab)

    return vocab


def eval_tasks(model, tasks, args):
    model.eval()
    result = []
    for i, task in enumerate(tasks):
        decode_results = model.decode(task.test.examples, i)
        eval_results = model.evaluator.evaluate_dataset(task.test.examples, decode_results,
                                                       fast_mode=args.eval_top_pred_only)
        test_score = eval_results[model.evaluator.default_metric]

        result.append(test_score)

    return result

def life_experience(net, continuum_dataset, args):
    result_a = []
    result_t = []

    current_task = 0
    time_start = time.time()

    for task_indx, task_data in enumerate(continuum_dataset):
        if((task_indx != current_task)):
            result_a.append(eval_tasks(net, continuum_dataset, args))
            result_t.append(current_task)
            current_task = task_indx
        net.train()
        net.observe(task_data, task_indx)

    result_a.append(eval_tasks(net, continuum_dataset, args))
    result_t.append(current_task)

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_t), torch.Tensor(result_a), time_spent

if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    print(args, file=sys.stderr)
    domains = args.domains
    print("training started ...")

    train_path_list = read_domain_data(domains, args.train_file, "_train.bin")

    test_path_list = read_domain_data(domains, args.train_file, "_test.bin")

    if args.dev_file:
        dev_path_list = read_domain_data(domains, args.dev_file, "_dev.bin")
    else:
        dev_path_list = None

    vocab_path_list = read_domain_data(domains, args.vocab, ".vocab.freq.bin")

    continuum_dataset = ContinuumDataset.read_continuum_data(train_path_list, test_path_list, dev_path_list, vocab_path_list)

    # unique identifier
    uid = uuid.uuid4().hex

    print("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    net = parser_cls(args, continuum_dataset)
    result_t, result_a, spent_time = life_experience(net, continuum_dataset, args)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    fname = args.parser + '_' + args.train_file + '_'
    fname += datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_' + uid
    fname = os.path.join(args.save_decode_to, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_t, result_a, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))