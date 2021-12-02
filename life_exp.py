# coding=utf-8
from __future__ import print_function

import argparse
import uuid
import datetime
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

from continual_model.utils import confusion_matrix, write_unforget_examples, write_action_prob_diff
from model.utils import GloveHelper, merge_vocab_entry
from common.registerable import Registrable
from components.dataset import ContinuumDataset, Batch
from common.utils import update_args, init_arg_parser
from datasets import *
from model import nn_utils
import evaluation
from components.grammar_validation.asdl import LambdaCalculusTransitionSystem
from components.evaluator import DefaultEvaluator, ActionEvaluator, SmatchEvaluator
from continual_model.gem import Net
from continual_model.icarl import Net
from continual_model.emr import Net
from continual_model.ea_emr import Net
from continual_model.origin import Net
from continual_model.independent import Net
from continual_model.gem_emr import Net
from continual_model.loss_emr import Net
from continual_model.loss_gem import Net
from continual_model.a_gem import Net
from continual_model.gss_emr import Net
from continual_model.emr_wo_task_boundary import Net
from continual_model.adap_emr import Net
from continual_model.emar import Net
from continual_model.hat import Net
from continual_model.ewc import Net
from continual_model.adap_emr_bi import Net
# from model.seq2seq_align import Seq2SeqModel

def init_config():
    args = arg_parser.parse_args()

    return args

def init_parameter_seed():
    # seed the RNG
    torch.manual_seed(args.p_seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.p_seed)
    np.random.seed(int(args.p_seed * 13 / 7))


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

def eval_action_prob(model, tasks, args, current_task):
    was_training = model.training
    model.eval()
    result = []

    for i, task in enumerate(tasks):
        if i <= current_task:
            t_offset1, t_offset2 = model.compute_offsets(i)
            nt_offset = model.nt_nc_per_task[i]
            action_prob_dict = {}
            for example in task.train.examples:
                batch = Batch([example], model.vocab, use_cuda=model.use_cuda)
                if args.parser == 'adap_emr':
                    nt_scores, t_scores, gate_vecs = model.forward_with_offset(model.model, t_offset1, t_offset2, nt_offset,
                                                                    batch, task_id = i)
                else:
                    nt_scores, t_scores = model.forward_with_offset(model.model, t_offset1, t_offset2, nt_offset,
                                                                   batch)

                batch.t_action_idx_matrix[batch.t_action_idx_matrix.nonzero(as_tuple=True)] = batch.t_action_idx_matrix[
                                                                                                  batch.t_action_idx_matrix.nonzero(
                                                                                                      as_tuple=True)] - t_offset1 + 1

                t_action_prob = torch.softmax(t_scores, dim=-1)
                # (tgt_action_len, batch_size)
                tgt_t_action_prob = torch.gather(t_action_prob, dim=2,
                                                     index=batch.t_action_idx_matrix.unsqueeze(2)).squeeze(2)

                tgt_t_action_prob = tgt_t_action_prob * batch.t_action_mask

                # print (t_action_prob.squeeze().size())

                nt_action_prob = torch.softmax(nt_scores, dim=-1)

                tgt_nt_action_prob = torch.gather(nt_action_prob, dim=2,
                                                      index=batch.nt_action_idx_matrix.unsqueeze(2)).squeeze(2)

                tgt_nt_action_prob = tgt_nt_action_prob * batch.nt_action_mask

                # print (nt_action_prob.squeeze().size())

                action_cat_prob = torch.cat([t_action_prob.squeeze(), nt_action_prob.squeeze()], dim=1)

                att_weights = model.model.att_weights

                #print (action_cat_prob.size())
                #print (len(model.model.t_action_vocab) + len(model.model.nt_action_vocab))
                #print(att_weights.size())
                max_indx = torch.argmax(action_cat_prob, dim=1)
                max_att_indx = torch.argmax(att_weights, dim=1)
                #print (max_indx.size())
                action_prob = tgt_t_action_prob + tgt_nt_action_prob
                for aciton_id, action in enumerate(example.tgt_actions):
                    action_indice = max_indx[aciton_id].item()

                    utterance_token_indx = max_att_indx[aciton_id].item()

                    #print (action_indice)
                    if action_indice < t_action_prob.squeeze().size(1):
                        action_indice = action_indice + t_offset1 - 1
                        pred_action = model.model.t_action_vocab.id2token[action_indice]
                    else:
                        action_indice = (action_indice - t_action_prob.squeeze().size(1)) % nt_offset
                        pred_action = model.model.nt_action_vocab.id2token[action_indice]

                    if action in action_prob_dict:
                        action_prob_dict[action]['action_prob'].append(action_prob[aciton_id].item())
                        action_prob_dict[action]['action_prob_all'].append(action_prob.detach().cpu())
                        action_prob_dict[action]['pred_action'].append(pred_action)
                        utterance = " ".join(example.src_sent)
                        expand_utterance = ["<bos>"] + example.src_sent + ["<eos>"]
                        action_prob_dict[action]['utterance_att_tokens'].add(expand_utterance[utterance_token_indx])
                        if not utterance in action_prob_dict[action]:
                            action_prob_dict[action]['utterance'].append(utterance)
                            action_prob_dict[action]['utterance_embed'].append(None)
                    else:
                        action_prob_dict[action] = {}
                        action_prob_dict[action]['action_prob'] = []
                        action_prob_dict[action]['action_prob'].append(action_prob[aciton_id].item())
                        action_prob_dict[action]['action_prob_all'] = []
                        action_prob_dict[action]['action_prob_all'].append(action_prob.detach().cpu())

                        action_prob_dict[action]['pred_action'] = []
                        action_prob_dict[action]['pred_action'].append(pred_action)
                        action_prob_dict[action]['utterance'] = []
                        utterance = " ".join(example.src_sent)
                        expand_utterance = ["<bos>"] + example.src_sent + ["<eos>"]
                        action_prob_dict[action]['utterance_att_tokens'] = set()
                        action_prob_dict[action]['utterance_att_tokens'].add(expand_utterance[utterance_token_indx])
                        action_prob_dict[action]['utterance'].append(utterance)
                        action_prob_dict[action]['utterance_embed'] = []
                        action_prob_dict[action]['utterance_embed'].append(None)


            result.append(action_prob_dict)
    if was_training: model.train()
    return result

def eval_task_forget_examples(model, tasks, args, current_task):
    was_training = model.training
    model.eval()
    result = []

    for i, task in enumerate(tasks):
        if i <= current_task:
            decode_results = model.decode(task.train.examples, i)
            eval_results = model.evaluator.evaluate_dataset(task.train.examples, decode_results,
                                                               fast_mode=args.eval_top_pred_only)

            result.append(eval_results[model.evaluator.correct_array])



    if was_training: model.train()
    return result

def eval_tasks(model, tasks, args, current_task):
    was_training = model.training
    model.eval()
    result = []
    correct_num = 0
    total_num = 0
    for i, task in enumerate(tasks):
        if args.num_known_domains == len(args.domains):
            decode_results = model.decode(task.test.examples, 0)
        else:
            decode_results = model.decode(task.test.examples, i)

        if args.evaluator == "denotation_evaluator":
            eval_results = model.evaluator.evaluate_dataset(task.test.examples, decode_results,
                                                           fast_mode=args.eval_top_pred_only, test_data = task.test)
        else:
            eval_results = model.evaluator.evaluate_dataset(task.test.examples, decode_results,
                                                           fast_mode=args.eval_top_pred_only)
        print("[accuracy: " + str(eval_results[model.evaluator.default_metric]) + " , correct_num: " + str(eval_results[model.evaluator.correct_num]) + "]", file=sys.stderr)
        test_score = eval_results[model.evaluator.default_metric]

        result.append(test_score)
        if i <= current_task:
            total_num += len(task.test.examples)
            correct_num += eval_results[model.evaluator.correct_num]
    if total_num == 0:
        final_acc = 0
    else:
        final_acc = correct_num/total_num
    result.append(final_acc)
    if was_training: model.train()
    return result

def life_experience(net, continuum_dataset, args, test_continuum_dataset):
    result_a = []

    forget_result = []
    action_prob = []
    time_start = time.time()

    for task_indx, task_data in enumerate(continuum_dataset):
        last_task_indx = task_indx - 1
        result_a.append(eval_tasks(net, continuum_dataset, args, last_task_indx))
        net.train()
        net.observe(task_data, task_indx)
        if args.forget_evaluate:
            forget_result.append(eval_task_forget_examples(net, continuum_dataset, args, task_indx))
        if args.action_forget_evaluate:
            action_prob.append(eval_action_prob(net, continuum_dataset, args, task_indx))

    last_task_indx = len(continuum_dataset) - 1
    result_a.append(eval_tasks(net, continuum_dataset, args, last_task_indx))
    if test_continuum_dataset is not None:
        sep_result = eval_tasks(net, test_continuum_dataset, args, last_task_indx)
        tensor_sep_result = torch.Tensor(sep_result)
        #print (sep_result)
        print (tensor_sep_result.mean().item())

    time_end = time.time()
    time_spent = time_end - time_start

    return torch.Tensor(result_a), time_spent, forget_result, action_prob

if __name__ == '__main__':

    arg_parser = init_arg_parser()
    args = init_config()

    if torch.cuda.is_available():
        args.use_cuda = True
    else:
        args.use_cuda = False

    print(args, file=sys.stderr)
    domains = args.domains
    print("training started ...")

    train_path_list = read_domain_data(domains, args.train_file, "_train.bin")
    if args.dev:
        test_path_list = read_domain_data(domains, args.test_file, "_dev.bin")
    else:
        test_path_list = read_domain_data(domains, args.test_file, "_test.bin")

    if args.dev_file:
        dev_path_list = read_domain_data(domains, args.dev_file, "_dev.bin")
    else:
        dev_path_list = None

    vocab_path_list = read_domain_data(domains, args.vocab, ".vocab.freq.bin")

    continuum_dataset = ContinuumDataset.read_continuum_data(args, train_path_list, test_path_list, dev_path_list, vocab_path_list, domains)

    test_continuum_dataset = None
    if len(domains) == int(args.num_known_domains):
        args.num_known_domains = 1
        test_continuum_dataset = ContinuumDataset.read_continuum_data(args, train_path_list, test_path_list, dev_path_list, vocab_path_list, domains)
        args.num_known_domains = len(args.domains)

    init_parameter_seed()
    # unique identifier
    uid = uuid.uuid4().hex

    print("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg
    net = parser_cls(args, continuum_dataset)
    result_a, spent_time, forget_results, action_prob = life_experience(net, continuum_dataset, args, test_continuum_dataset)
    save_dir = os.path.join(args.save_decode_to, args.parser)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)


    fname = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    fname += '_parser.' + str(args.parser) + \
             '_max_ep.' + str(args.max_epoch) + \
             '_samp.' + str(args.sample_method) + \
             '_lr.' + str(args.lr) + \
             '_perm_sed.' + str(args.seed) + \
             '_para_sed.' + str(args.p_seed) + \
             '_bat.' + str(args.batch_size) + \
             '_nd.' + str(args.num_known_domains) + \
             '_e_num.' + str(args.num_exemplars_per_task) + \
             '_sm.' + str(args.reg) + \
             '_ewc.' + str(args.ewc) + \
             '_b_model.' + str(args.base_model) + \
             '_init_ep.' + str(args.initial_epoch) + \
             '_sec_iter.' + str(args.second_iter) + \
             '_cluster_fil.' + str(args.clustering_filter) + \
             '_act_num.' + str(args.action_num) + \
             '_gate.' + str(args.gate_function) + \
             '_proto.' + str(args.init_proto) + \
             '_warm.' + str(args.warm) + \
             '_dev.' + str(args.dev) + \
             '_emb.' + str(args.embed_type)

    fname = os.path.join(save_dir, fname)

    # save confusion matrix and print one line of stats
    stats = confusion_matrix(result_a, fname + '.txt')
    one_liner = str(vars(args)) + ' # '
    one_liner += ' '.join(["%.3f" % stat for stat in stats])
    print(fname + ': ' + one_liner + ' # ' + str(spent_time))

    forget_dir_prefix = 'parser.' + str(args.parser) + \
                        '_max_ep.' + str(args.max_epoch) + \
                        '_samp.' + str(args.sample_method) + \
                        '_lr.' + str(args.lr) + \
                        '_para_sed.' + str(args.p_seed) + \
                        '_bat.' + str(args.batch_size) + \
                        '_nd.' + str(args.num_known_domains) + \
                        '_e_num.' + str(args.num_exemplars_per_task) + \
                        '_sm.' + str(args.reg) + \
                        '_ewc.' + str(args.ewc) + \
                        '_b_model.' + str(args.base_model) + \
                        '_init_ep.' + str(args.initial_epoch) + \
                        '_sec_iter.' + str(args.second_iter) + \
                        '_cluster_fil.' + str(args.clustering_filter) + \
                        '_act_num.' + str(args.action_num) + \
                        '_gate.' + str(args.gate_function) + \
                        '_proto.' + str(args.init_proto) + \
                        '_warm.' + str(args.warm) + \
                        '_emb.' + str(args.embed_type)

    if args.forget_evaluate:

        forget_save_dir = os.path.join(os.path.join(save_dir, forget_dir_prefix), 'forget_examples')
        if not os.path.exists(forget_save_dir):
            os.makedirs(forget_save_dir)
        permute_name = '_perm_sed.' + str(args.seed)
        forget_fname = os.path.join(forget_save_dir, permute_name)
        write_unforget_examples(continuum_dataset, forget_results, forget_fname + '.txt')
    if args.action_forget_evaluate:
        forget_save_dir = os.path.join(os.path.join(save_dir, forget_dir_prefix), 'action_prob')
        if not os.path.exists(forget_save_dir):
            os.makedirs(forget_save_dir)

        permute_name = '_perm_sed.' + str(args.seed)

        action_forget_fname = os.path.join(forget_save_dir, permute_name)

        pickle.dump(action_prob, open(action_forget_fname+ ".stat.bin", 'wb'))
        write_action_prob_diff(action_prob, action_forget_fname + '.txt')

    if args.record_error:
        error_save_dir = os.path.join(os.path.join(save_dir, forget_dir_prefix), 'error')
        if not os.path.exists(error_save_dir):
            os.makedirs(error_save_dir)

        permute_name = '_perm_sed.' + str(args.seed)

        error_fname = os.path.join(error_save_dir, permute_name)

        print (net.train_error)
        print (net.test_error)
        pickle.dump(net.train_error, open(error_fname+ ".train.bin", 'wb'))
        pickle.dump(net.test_error, open(error_fname+ ".test.bin", 'wb'))