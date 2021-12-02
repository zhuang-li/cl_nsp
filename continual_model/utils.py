import os
import pickle
import re
from operator import itemgetter

import numpy as np
import ot
from scipy.stats import wilcoxon
from zss import simple_distance, Node
import torch
import scipy.stats as st

from components.vocab import TokenVocabEntry
from grammar.action import GenNTAction, GenTAction
from grammar.vertex import RuleVertex

def ori_tree2edit_tree(ori_tree):
    ori_queue = []
    ori_queue.append(ori_tree)
    new_queue = []
    new_node = Node(ori_tree.head)
    new_queue.append(new_node)
    while len(ori_queue) > 0:
        current_ori_node = ori_queue.pop(0)
        current_new_node = new_queue.pop(0)
        for child in current_ori_node.children:
            ori_queue.append(child)
            new_child_node = Node(child.head)
            current_new_node.addkid(new_child_node)
            new_queue.append(new_child_node)
    return (new_node)

def read_domain_data(domains, file_prefix, suffix):
    path_list = []
    for domain in domains:
        path = os.path.join(file_prefix, domain + suffix)
        path_list.append(path)
    return path_list


def read_vocab_list(domains, file_prefix, suffix, num_init_domains = 5):
    assert len(domains) > 0

    vocab_known = pickle.load(open(os.path.join(file_prefix, domains[0] + suffix), 'rb'))

    for i in range(1, num_init_domains):
        domain = domains[i]
        vocab_after = pickle.load(open(os.path.join(file_prefix, domain + suffix), 'rb'))
        vocab_known = merge_vocab_entry(vocab_after, vocab_known)

    vocab_dict = dict()
    vocab_dict['known'] = vocab_known

    for domain in domains[num_init_domains:]:

        vocab = pickle.load(open(os.path.join(file_prefix, domain + suffix), 'rb'))
        vocab_dict[domain] = vocab
        #vocab_after = pickle.load(open(os.path.join(file_prefix, domain + suffix), 'rb'))
        #vocab = merge_vocab_entry(vocab_after, vocab)

    return vocab_dict


def read_domain_vocab(domains, file_prefix, suffix):
    assert len(domains) > 0
    vocab = pickle.load(open(os.path.join(file_prefix, domains[0] + suffix), 'rb'))
    for domain in domains[1:]:
        vocab_after = pickle.load(open(os.path.join(file_prefix, domain + suffix), 'rb'))
        vocab = merge_vocab_entry(vocab_after, vocab)

    return vocab


def merge_vocab(pre_train_vocab_entry, new_vocab_entry, type):
    merged_token2id = dict()
    for token, id in pre_train_vocab_entry.token2id.items():
        if not token in merged_token2id:
            merged_token2id[token] = len(merged_token2id)

    for token, id in new_vocab_entry.token2id.items():
        if not token in merged_token2id:
            merged_token2id[token] = len(merged_token2id)


    merged_id2token = {v: k for k, v in merged_token2id.items()}
    new_vocab_entry.token2id = merged_token2id
    new_vocab_entry.id2token = merged_id2token

    if type == "t_action" or type == "nt_action":

        merged_vertex2action = dict()
        for vertex, action in pre_train_vocab_entry.vertex2action.items():
            if not vertex in merged_vertex2action:
                merged_vertex2action[vertex] = action

        for vertex, action in new_vocab_entry.vertex2action.items():
            if not vertex in merged_vertex2action:
                merged_vertex2action[vertex] = action
            else:
                if type == "t_action":
                    assert str(vertex) == '<pad>', str(vertex)

        new_vocab_entry.vertex2action = merged_vertex2action

        merged_lhs2rhs = dict()
        for lhs, rhs in pre_train_vocab_entry.lhs2rhs.items():
            if not lhs in merged_lhs2rhs:
                merged_lhs2rhs[lhs] = rhs
            else:
                merged_lhs2rhs[lhs].extend(rhs)

        for lhs, rhs in new_vocab_entry.lhs2rhs.items():
            if not lhs in merged_lhs2rhs:
                merged_lhs2rhs[lhs] = rhs
            else:
                merged_lhs2rhs[lhs].extend(rhs)

        merged_rhs2lhs = dict()

        for lhs, rhs_list in merged_lhs2rhs.items():
            for rhs in rhs_list:
                merged_rhs2lhs[rhs] = lhs

        new_vocab_entry.lhs2rhs = merged_lhs2rhs
        new_vocab_entry.rhs2lhs = merged_rhs2lhs


        merged_lhs2rhsid = dict()
        for lhs, rhs_list in merged_lhs2rhs.items():
            rhsid = []
            for rhs in rhs_list:
                action = merged_vertex2action[rhs]
                id = merged_token2id[action]
                rhsid.append(id)
            merged_lhs2rhsid[lhs] = rhsid
        new_vocab_entry.lhs2rhsid = merged_lhs2rhsid

        merged_entype2action = dict()

        for entype, action in pre_train_vocab_entry.entype2action.items():
            if not entype in merged_entype2action:
                merged_entype2action[entype] = action
            else:
                merged_entype2action[entype].extend(action)

        for entype, action in new_vocab_entry.entype2action.items():
            if not entype in merged_entype2action:
                merged_entype2action[entype] = action
            else:
                merged_entype2action[entype].extend(action)


        new_vocab_entry.entype2action = merged_entype2action


        merged_action2nl = dict()

        for action, nl in pre_train_vocab_entry.action2nl.items():
            if not action in merged_action2nl:
                merged_action2nl[action] = nl
            else:
                merged_action2nl[action].extend(nl)

        for action, nl in new_vocab_entry.action2nl.items():
            if not action in merged_action2nl:
                merged_action2nl[action] = nl
            else:
                merged_action2nl[action].extend(nl)

        new_vocab_entry.action2nl = merged_action2nl



def merge_vocab_entry(pretrain_vocab, new_vocab):
    merge_vocab(pretrain_vocab.source, new_vocab.source, "source")
    merge_vocab(pretrain_vocab.nt_action, new_vocab.nt_action, "nt_action")
    merge_vocab(pretrain_vocab.t_action, new_vocab.t_action, "t_action")
    return new_vocab


def merge_token_vocab(old_vocab, new_vocab):
    vocab_entry = TokenVocabEntry()
    merged_token2id = {k: v for k, v in old_vocab.token2id.items()}
    vocab_entry.pad_id = old_vocab.pad_id
    vocab_entry.unk_id = old_vocab.unk_id

    #print(vocab_entry.pad_id)
    #print(vocab_entry.unk_id)

    #for token, id in old_vocab.token2id.items():
    #    if not token in merged_token2id:
    #        merged_token2id[token] = len(merged_token2id)

    for token, id in new_vocab.token2id.items():
        if not token in merged_token2id:
            merged_token2id[token] = len(merged_token2id)


    merged_id2token = {v: k for k, v in merged_token2id.items()}
    vocab_entry.token2id = merged_token2id
    vocab_entry.id2token = merged_id2token

    return vocab_entry




def get_init_nc(vocab, vocabs, dataset_permutation_indx):
    main_nt_action_set = set(vocab.nt_action.token2id.keys())

    new_vocab_list = [vocabs[idx] for idx in dataset_permutation_indx]

    nt_action_set_list = [set(new_vocab.nt_action.token2id.keys()) for new_vocab in new_vocab_list]
    nt_action_set_list.append(main_nt_action_set)
    intersect_actions = set.intersection(*nt_action_set_list)

    merged_token2id = dict()

    merged_token2id[GenNTAction(RuleVertex('<pad>'))] = 0

    old_nt_action_entry = vocab.nt_action.copy()
    for action in list(intersect_actions):
        if not action in merged_token2id:
            merged_token2id[action] = len(merged_token2id)


    merged_id2token = {v: k for k, v in merged_token2id.items()}

    vocab.nt_action.token2id = merged_token2id
    vocab.nt_action.id2token = merged_id2token

    merge_vocab(old_nt_action_entry, vocab.nt_action, "nt_action")

    return len(intersect_actions)

def intersect(*d):
    sets = iter(map(set, d))
    result = sets.__next__()
    for s in sets:
        result = result.intersection(s)
    return result

def interval(data):
    """
    data: 1-dim np array
    """
    interv = st.t.interval(0.95, len(data) - 1, loc=np.mean(data), scale=st.sem(data))
    mean = np.mean(data)
    interv = interv - mean
    return mean, interv

def jaccard(a, b):
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

#1 #
#2 # #
#3 # # #
#4 # # # #

def write_action_prob_diff(action_results, action_forget_fname, previous_transition_dict=None, transition_dict=None):

    action_prob_dict = {}

    if transition_dict is None:
    #print (action_results)
        transition_dict = {}
        for out_task_id, action_dict_list in enumerate(action_results):

            # out means the model trained on task id

            #print (out_task_id)
            #print (action_dict_list)
            for in_task_id, action_dict in enumerate(action_dict_list):

                # in means the data evaluated

                #if out_task_id - 1 > 0:
                #    print (len(action_results[out_task_id - 1]))
                #    print (in_task_id)
                if out_task_id - 1 >= 0 and len(action_results[out_task_id - 1]) > in_task_id:
                    #print (out_task_id - 1)
                    last_action_dict = action_results[out_task_id - 1][in_task_id]
                    current_action_dict = action_results[out_task_id][in_task_id]
                    current_data_action_dict = action_results[out_task_id][out_task_id]
                    prob_diff = {}

                    for action, action_record_dict in current_action_dict.items():
                        prob_list = action_record_dict['action_prob']

                        prob_all_list = action_record_dict['action_prob_all']

                        pred_action_list = action_record_dict['pred_action']

                        last_pred_action_list = last_action_dict[action]['pred_action']

                        if in_task_id == 0 and out_task_id - 1 == 0:
                            action_prob_dict[str(action).replace("nonterminal", "NT")] = sum(last_action_dict[action]['action_prob'])/len(last_action_dict[action]['action_prob'])

                        #print (action)
                        #print (pred_action_list)

                        #print (sum([1 for pred_action in pred_action_list if pred_action == action]))

                        #print ([1 for pred_action in pred_action_list if pred_action == action])

                        #print (sum([1 for pred_action in last_pred_action_list if pred_action == action]))

                        #print ([1 for pred_action in last_pred_action_list if pred_action == action])

                        current_acc = sum([1 for pred_action in pred_action_list if pred_action == action])/len(pred_action_list)
                        last_acc =  sum([1 for pred_action in last_pred_action_list if pred_action == action])/len(last_pred_action_list)

                        #print (last_acc)

                        prob_diff[action] = {}
                        if action in current_data_action_dict:

                            previous_current_action_dict = action_results[in_task_id][in_task_id]

                            prob_diff[action]['is_overlap'] = True
                            prob_diff[action]['overlap_freq'] = len(current_data_action_dict[action]['action_prob'])

                            previous_embed_list = previous_current_action_dict[action]['utterance_embed']

                            current_embed_list = current_data_action_dict[action]['utterance_embed']

                            previous_action_tokens = previous_current_action_dict[action]['utterance_att_tokens']
                            current_action_tokens = current_data_action_dict[action]['utterance_att_tokens']

                            jaccard_dist = jaccard(previous_action_tokens, current_action_tokens)
                            prob_diff[action]['jaccard_dist'] = jaccard_dist
                            #a = np.ndarray((len(previous_embed_list),)).fill(1/(len(previous_embed_list)))
                            #b = np.ndarray((len(current_embed_list),)).fill(1/(len(current_embed_list)))
                            #previous_embed = np.concatenate(previous_embed_list, axis=0)
                            #current_embed = np.concatenate(current_embed_list, axis=0)
                            #print (current_embed.shape)
                            #print (previous_embed.shape)
                            #M = ot.dist(previous_embed, current_embed, metric='cosine')
                            #print (M.shape)

                            #print (M)
                            #print (ot.emd([], [], M).sum())


                            #W = ot.emd2([], [], M)
                            #W_weight = ot.emd([], [], M)
                            #flatten_index = np.argmax(W_weight)
                            #x_len, y_len = W_weight.shape

                            #previous_current_utterance_list = action_record_dict['utterance']
                            #current_utterance_list = current_data_action_dict[action]['utterance']

                            #x_indx = int(flatten_index/y_len)
                            #y_indx = flatten_index%y_len

                            #previous_utterance = previous_current_utterance_list[x_indx]

                            #current_utterance = current_utterance_list[y_indx]
                            #matched_utterance.append((previous_utterance, current_utterance))
                            #actions.append(action)
                            #print (W)
                            prob_diff[action]['OT_dist'] = 0.0
                        else:
                            prob_diff[action]['is_overlap'] = False
                            prob_diff[action]['overlap_freq'] = 0
                            prob_diff[action]['OT_dist'] = 10000
                            prob_diff[action]['jaccard_dist'] = 0
                        action_prob_diff_list = []
                        for prob_id, prob in enumerate(prob_list):
                            action_prob_diff_list.append(prob - last_action_dict[action]['action_prob'][prob_id])

                        action_prob_all_diff_list = []
                        for prob_all_id, prob_all in enumerate(prob_all_list):
                            #print (prob_all.size())
                            w, p = wilcoxon((prob_all.squeeze() - last_action_dict[action]['action_prob_all'][prob_all_id].squeeze()).tolist())
                            action_prob_all_diff_list.append(p)

                        prob_diff[action]['prob_diff'] = action_prob_diff_list

                        prob_diff[action]['prob_all_diff'] = action_prob_all_diff_list

                        prob_diff[action]['acc_diff'] = current_acc - last_acc
                    transition_states = str(out_task_id - 1) + " -> " + str(out_task_id)
                    if transition_states in transition_dict:
                        transition_dict[transition_states][in_task_id] = prob_diff
                    else:
                        transition_dict[transition_states] = [None] * len(action_results)
                        transition_dict[transition_states][in_task_id] = prob_diff

        #print (transition_dict)

        pickle.dump(transition_dict, open(action_forget_fname[:-3]+"transition.bin", 'wb'))

    f = open(action_forget_fname, 'w')
    pos_W_all = []
    neg_W_all = []

    top_NT_actions = []

    top_T_actions = []


    top_NT_actions_dict = {}

    top_T_actions_dict = {}

    pos_jacc_dist = []

    neg_jacc_dist = []

    wiscon_stat = []

    wiscon_all = []

    pos_wiscon_stat = []
    neg_wiscon_stat = []

    pos_task_specific_action = []
    neg_task_specific_action = []

    positive_action_num = []
    negative_action_num = []

    pos_cross_actions = []
    pos_specific_actions = []

    neg_cross_actions = []
    neg_specific_actions = []

    cross_actions = []
    specific_actions = []



    for i in range(len(action_results) - 1):
        transition_state = str(i) + " -> " + str(i+1)
        print("Transition State : ", transition_state, file=f)
        pos_num = 0
        neg_num = 0
        for task_id, task_instances in enumerate(transition_dict[transition_state]):
            if task_instances is not None:
                print("Task Id : ", str(task_id), file=f)
                print("", file=f)
                NT_action_prob_mean = {}
                NT_action_prob_interval = {}
                NT_action_overlap = {}
                NT_action_overlap_freq = {}
                NT_action_acc_diff = {}

                NT_action_prob_stat_mean = {}
                NT_action_prob_stat_interval = {}

                NT_action_OT_dist = {}

                NT_action_jaccard_dist = {}

                T_action_prob_mean = {}
                T_action_prob_interval = {}
                T_action_overlap = {}
                T_action_overlap_freq = {}
                T_action_acc_diff = {}

                T_action_prob_stat_mean = {}
                T_action_prob_stat_interval = {}

                T_action_OT_dist = {}

                T_action_jaccard_dist = {}

                pos_W = []
                neg_W = []

                for action, prob_diff_dict in task_instances.items():
                    prob_diff = prob_diff_dict['prob_diff']
                    is_overlap = prob_diff_dict['is_overlap']
                    overlap_freq = prob_diff_dict['overlap_freq']
                    acc_diff = prob_diff_dict['acc_diff']
                    jaccard_dist = prob_diff_dict['jaccard_dist']
                    """
                    if float(acc_diff) >0:
                        pos_num += 1
                    else:
                        neg_num += 1
                    """

                    prob_diff = np.array(prob_diff)
                    mean_diff, diff_interval = interval(prob_diff)


                    prob_all_diff = prob_diff_dict['prob_all_diff']
                    wiscon_all.extend(prob_all_diff)
                    prob_all_diff = np.array(prob_all_diff)
                    mean_all_diff, diff_all_interval = interval(prob_all_diff)

                    wiscon_stat.append(mean_all_diff)

                    OT_dist = prob_diff_dict['OT_dist']

                    if not (int(OT_dist) == 10000):
                        if previous_transition_dict is not None:
                            origin_task_instance = previous_transition_dict[transition_state][task_id]
                            origin_prob_diff = origin_task_instance[action]['prob_diff']
                            origin_prob_diff = np.array(origin_prob_diff)
                            origin_mean_diff, origin_diff_interval = interval(origin_prob_diff)
                            if float(origin_mean_diff)>=0:
                                pos_W.append(float(OT_dist))
                                pos_W_all.append(float(OT_dist))

                                pos_wiscon_stat.extend(prob_all_diff)
                                #pos_jacc_dist.append(float(jaccard_dist))
                                #pos_num+=1
                            else:
                                neg_W.append(float(OT_dist))
                                neg_W_all.append(float(OT_dist))
                                neg_wiscon_stat.extend(prob_all_diff)
                                #neg_jacc_dist.append(float(jaccard_dist))
                                #neg_num+=1
                        else:
                            if float(mean_diff) >= 0:
                                pos_W.append(float(OT_dist))
                                pos_W_all.append(float(OT_dist))
                                pos_wiscon_stat.extend(prob_all_diff)
                                #pos_jacc_dist.append(float(jaccard_dist))
                                pos_num+=1
                            else:
                                neg_W.append(float(OT_dist))
                                neg_W_all.append(float(OT_dist))
                                neg_wiscon_stat.extend(prob_all_diff)
                                #neg_jacc_dist.append(float(jaccard_dist))
                                neg_num+=1

                    if isinstance(action, GenNTAction):
                        NT_action_prob_mean[action] = float(mean_diff)
                        NT_action_prob_interval[action] = str(diff_interval)
                        NT_action_overlap[action] = is_overlap
                        NT_action_overlap_freq[action] = overlap_freq
                        NT_action_acc_diff[action] = acc_diff

                        NT_action_prob_stat_mean[action] = float(mean_all_diff)
                        NT_action_prob_stat_interval[action] = str(diff_all_interval)

                        NT_action_OT_dist[action] = str(OT_dist)

                        NT_action_jaccard_dist[action] = str(jaccard_dist)

                        cross_actions.extend(prob_all_diff)
                        if float(mean_diff) >=0:
                            pos_cross_actions.extend(prob_all_diff)
                            pos_jacc_dist.append(float(jaccard_dist))
                        else:
                            neg_cross_actions.extend(prob_all_diff)
                            neg_jacc_dist.append(float(jaccard_dist))

                    elif isinstance(action, GenTAction):
                        T_action_prob_mean[action] = float(mean_diff)
                        T_action_prob_interval[action] = str(diff_interval)
                        T_action_overlap[action] = is_overlap
                        T_action_overlap_freq[action] = overlap_freq
                        T_action_acc_diff[action] = acc_diff

                        T_action_prob_stat_mean[action] = float(mean_all_diff)
                        T_action_prob_stat_interval[action] = str(diff_all_interval)

                        T_action_OT_dist[action] = str(OT_dist)

                        T_action_jaccard_dist[action] = str(jaccard_dist)

                        specific_actions.extend(prob_all_diff)
                        if float(mean_diff) >=0:
                            pos_specific_actions.extend(prob_all_diff)
                        else:
                            neg_specific_actions.extend(prob_all_diff)

                        if previous_transition_dict is not None:
                            origin_task_instance = previous_transition_dict[transition_state][task_id]
                            origin_prob_diff = origin_task_instance[action]['prob_diff']
                            origin_prob_diff = np.array(origin_prob_diff)
                            origin_mean_diff, origin_diff_interval = interval(origin_prob_diff)
                            if float(origin_mean_diff) >= 0:
                                pos_task_specific_action.extend(prob_all_diff)
                            else:
                                neg_task_specific_action.extend(prob_all_diff)
                        else:
                            if float(mean_diff) >= 0:
                                pos_task_specific_action.extend(prob_all_diff)
                            else:
                                neg_task_specific_action.extend(prob_all_diff)

                print("NT Action, mean diff, diff interval, is_overlap, overlap_freq, acc_diff, score rank stat, score rank interval, OT dist, Jaccard dist", file=f)
                print("", file=f)

                NT_actions = []
                T_actions = []
                NT_actions_dict = {}
                T_actions_dict = {}
                #sorted(NT_action_prob_mean.items(), key=itemgetter(1))
                for action, mean_diff in sorted(NT_action_prob_mean.items(), key=itemgetter(1)):
                    NT_actions.append(str(action).replace("nonterminal", "NT"))
                    NT_actions_dict[str(action).replace("nonterminal", "NT")] = mean_diff
                    print(str(action) + ": " + str(mean_diff) + ", " + NT_action_prob_interval[action] + ", " + str(NT_action_overlap[action]) + ", " + str(NT_action_overlap_freq[action]) + ", " + str(NT_action_acc_diff[action]) + ", " + str(NT_action_prob_stat_mean[action]) + ", " + str(NT_action_prob_stat_interval[action]) + ", " + str(NT_action_OT_dist[action]) + ", " + str(NT_action_jaccard_dist[action]), file=f)
                print("", file=f)
                print("T Action, mean diff, diff interval", file=f)
                print("", file=f)
                for action, mean_diff in sorted(T_action_prob_mean.items(), key=itemgetter(1)):
                    T_actions.append(str(action).replace("nonterminal", "NT"))
                    T_actions_dict[str(action).replace("nonterminal", "NT")] = mean_diff
                    print(str(action) + ": " + str(mean_diff) + ", " + T_action_prob_interval[action] + ", " +str(T_action_overlap[action]) + ", " + str(T_action_overlap_freq[action]) + ", " + str(T_action_acc_diff[action]) + ", " + str(T_action_prob_stat_mean[action]) + ", " + str(T_action_prob_stat_interval[action]) + ", " + str(T_action_OT_dist[action]) + ", " + str(T_action_jaccard_dist[action]), file=f)
                print("", file=f)

                if i == 0 and task_id == 0:
                    top_NT_actions = NT_actions[:3] + NT_actions[-3:]
                    top_T_actions = T_actions[:3] + T_actions[-3:]

                if task_id == 0:
                    for action in top_NT_actions:
                        if action in top_NT_actions_dict:
                            top_NT_actions_dict[action].append(top_NT_actions_dict[action][-1] + NT_actions_dict[str(action).replace("nonterminal", "NT")])
                        else:
                            top_NT_actions_dict[action] = []
                            top_NT_actions_dict[action].append(action_prob_dict[action])
                            top_NT_actions_dict[action].append(action_prob_dict[action] + NT_actions_dict[str(action).replace("nonterminal", "NT")])

                    for action in top_T_actions:
                        if action in top_T_actions_dict:
                            top_T_actions_dict[action].append(top_T_actions_dict[action][-1] + T_actions_dict[
                                str(action).replace("nonterminal", "NT")])
                        else:
                            top_T_actions_dict[action] = []
                            top_T_actions_dict[action].append(action_prob_dict[action])
                            top_T_actions_dict[action].append(action_prob_dict[action] + T_actions_dict[str(action).replace("nonterminal", "NT")])

                print("Positive Action OT dist", file=f)

                pos_W = np.array(pos_W)
                pos_W_mean, pos_W_interval = interval(pos_W)

                print (pos_W_mean,file=f)
                print("", file=f)
                print (str(pos_W_interval), file=f)
                print("Negative Action OT dist", file=f)

                neg_W = np.array(neg_W)
                neg_W_mean, neg_W_interval = interval(neg_W)
                print(neg_W_mean, file=f)
                print("", file=f)
                print (str(neg_W_interval), file=f)
                print("", file=f)
        positive_action_num.append(pos_num)
        negative_action_num.append(neg_num)
    #for indx, (previous_utterance, current_utterance) in enumerate(matched_utterance):
    #    print(str(actions[indx]) + " : " + previous_utterance + "   :----:  " + current_utterance, file=f)


    record_result = []
    record_result.append(pos_W_all) #0
    record_result.append(neg_W_all) #1
    record_result.append(wiscon_stat) #2
    record_result.append(wiscon_all) #3
    record_result.append(pos_wiscon_stat) #4
    record_result.append(neg_wiscon_stat) #5
    record_result.append(pos_task_specific_action) #6
    record_result.append(neg_task_specific_action) #7
    record_result.append(positive_action_num) #8
    record_result.append(negative_action_num) #9
    record_result.append(pos_cross_actions) #10
    record_result.append(pos_specific_actions) #11
    record_result.append(neg_cross_actions) #12
    record_result.append(neg_specific_actions) #13
    record_result.append(cross_actions) #14
    record_result.append(specific_actions) #15
    record_result.append(top_NT_actions_dict)  # 16
    record_result.append(top_T_actions_dict)  # 17



    pickle.dump(record_result, open(action_forget_fname[:-3] + "record.bin", 'wb'))


    print("Positive Action OT dist", file=f)

    pos_W_all = np.array(pos_W_all)
    pos_W_all = remove_nan(pos_W_all)

    pos_W_all_mean, pos_W_all_interval = interval(pos_W_all)

    print(pos_W_all_mean, file=f)
    print("", file=f)
    print(str(pos_W_all_interval), file=f)
    print("Negative Action OT dist", file=f)

    neg_W_all = np.array(neg_W_all)
    neg_W_all = remove_nan(neg_W_all)

    neg_W_all_mean, neg_W_all_interval = interval(neg_W_all)
    print(neg_W_all_mean, file=f)
    print("", file=f)
    print(str(neg_W_all_interval), file=f)
    print("", file=f)

    print("Positive Action Jacc dist", file=f)

    pos_jacc_dist = np.array(pos_jacc_dist)
    pos_jacc_dist = remove_nan(pos_jacc_dist)

    pos_jacc_dist_all_mean, pos_jacc_dist_all_interval = interval(pos_jacc_dist)

    print(pos_jacc_dist_all_mean, file=f)
    print("", file=f)
    print(str(pos_jacc_dist_all_interval), file=f)
    print("Negative Action Jacc dist", file=f)

    neg_jacc_dist = np.array(neg_jacc_dist)
    neg_jacc_dist = remove_nan(neg_jacc_dist)

    neg_jacc_dist_all_mean, neg_jacc_dist_all_interval = interval(neg_jacc_dist)
    print(neg_jacc_dist_all_mean, file=f)
    print("", file=f)
    print(str(neg_jacc_dist_all_interval), file=f)
    print("", file=f)


    print("Positive Action Wiscon dist", file=f)

    pos_wiscon_stat = np.array(pos_wiscon_stat)
    pos_wiscon_stat = remove_nan(pos_wiscon_stat)

    pos_wiscon_stat_mean, pos_wiscon_stat_interval = interval(pos_wiscon_stat)

    print(pos_wiscon_stat_mean, file=f)
    print("", file=f)
    print(str(pos_wiscon_stat_interval), file=f)

    print("Negative Action Wiscon dist", file=f)

    neg_wiscon_stat = np.array(neg_wiscon_stat)
    neg_wiscon_stat = remove_nan(neg_wiscon_stat)

    neg_wiscon_stat_mean, neg_wiscon_stat_interval = interval(neg_wiscon_stat)
    print(neg_wiscon_stat_mean, file=f)
    print("", file=f)
    print(str(neg_wiscon_stat_interval), file=f)
    print("", file=f)

    print("Wiscon Stat", file=f)

    wiscon_stat = np.array(wiscon_stat)

    wiscon_stat = remove_nan(wiscon_stat)

    wiscon_stat_mean, wiscon_stat_interval = interval(wiscon_stat)
    print(wiscon_stat_mean, file=f)
    print("", file=f)
    print(str(wiscon_stat_interval), file=f)
    print("", file=f)

    print("Wiscon All Stat", file=f)

    wiscon_all = np.array(wiscon_all)

    wiscon_all = remove_nan(wiscon_all)

    wiscon_all_mean, wiscon_all_interval = interval(wiscon_all)
    print(wiscon_all_mean, file=f)
    print("", file=f)
    print(str(wiscon_all_interval), file=f)
    print("", file=f)

    f.close()

def remove_nan(out_vec):
    if np.isnan(np.sum(out_vec)):
        out_vec = out_vec[~np.isnan(out_vec)]
    return out_vec

def plot_stat(data):
    import matplotlib.pyplot as plt
    import numpy as np

    ticks = ['Overnight', 'NLMap(qt)', 'NLMap(city)']

    def set_box_color(bp, color, id):
        plt.setp(bp['boxes'][int(id/2)], color=color)
        plt.setp(bp['whiskers'][id], color=color)
        plt.setp(bp['whiskers'][id+1], color=color)
        plt.setp(bp['caps'][id], color=color)
        plt.setp(bp['caps'][id+1], color=color)
        #plt.setp(bp['medians'][0], color=color)

    plt.figure()

    bpl = plt.boxplot(data, positions=np.array(range(len(data))), sym='', widths=0.6,showfliers=True)
    set_box_color(bpl, '#D7191C', 0)  # colors are from http://colorbrewer2.org/
    set_box_color(bpl, '#9ebcda', 2)
    set_box_color(bpl, '#2ca25f', 4)


    plt.xticks(range(0, len(ticks)), ticks)
    plt.xlim(-1, len(ticks))
    plt.ylim(0, 0.9)
    plt.tight_layout()
    plt.savefig("record_dir/stat_compare_origin.png")
    plt.show()

def plot_OT(positive_action, negtive_action, new_positive_actions, new_negative_actions):
    import matplotlib.pyplot as plt
    import numpy as np

    data_a = positive_action
    data_b = negtive_action

    ticks = ['Overnight', 'NLMap(qt)', 'NLMap(city)']

    def set_box_color(bp, color):
        plt.setp(bp['boxes'], color=color)
        plt.setp(bp['whiskers'], color=color)
        plt.setp(bp['caps'], color=color)
        plt.setp(bp['medians'], color=color)

    plt.figure()

    fig, (ax1, ax2) = plt.subplots(1, 2)
    #fig.suptitle('Horizontally stacked subplots')

    bpl1 = ax1.boxplot(positive_action, positions=np.array(range(len(data_a))) * 2.0 - 0.4, sym='', widths=0.6,showfliers=True)
    bpr1 = ax1.boxplot(negtive_action, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6,showfliers=True)
    set_box_color(bpl1, '#D7191C')  # colors are from http://colorbrewer2.org/
    set_box_color(bpr1, '#2C7BB6')

    # draw temporary red and blue lines and use them to create a legend
    ax1.plot([], c='#D7191C', label='Cross-task Actions')
    ax1.plot([], c='#2C7BB6', label='Task-specific Actions')
    ax1.legend()

    bpl2 = ax2.boxplot(new_positive_actions, positions=np.array(range(len(data_a))) * 2.0 - 0.4, sym='', widths=0.6,showfliers=True)
    bpr2 = ax2.boxplot(new_negative_actions, positions=np.array(range(len(data_b))) * 2.0 + 0.4, sym='', widths=0.6,showfliers=True)
    set_box_color(bpl2, '#D7191C')  # colors are from http://colorbrewer2.org/
    set_box_color(bpr2, '#2C7BB6')

    plt.setp((ax1, ax2), xticks=range(0, len(ticks) * 2, 2), xticklabels=ticks,yticks=[0,0.25, 0.5, 0.75,1, 1.25])

    #plt.xticks(range(0, len(ticks) * 2, 2), ticks)
    #plt.xlim(-2, len(ticks) * 2)
    #plt.ylim(0, 0.4)
    plt.tight_layout()
    plt.savefig("record_dir/OT_dist_compare.png")
    plt.show()

def write_unforget_examples(tasks, forget_results, forget_fname):
    f = open(forget_fname, 'w')
    print (len(forget_results))
    for i, task_data in enumerate(tasks):
        if i < len(tasks) - 1:
            acc_list = []
            for f_r in forget_results:
                if i < len(f_r):
                    acc_list.append(f_r[i])
            #print (acc_list)
            unforget_acc = len(acc_list)
            acc_matrix = torch.stack(acc_list, dim=0)
            print (acc_matrix.size())
            acc_values = acc_matrix.sum(dim=0).tolist()
            for train_id, value in enumerate(acc_values):
                if value == unforget_acc:
                    example = task_data.train.examples[train_id]
                    print(str(example) + '\t' + example.to_logic_form, file=f)
    f.close()

    dir_name = os.path.dirname(forget_fname)
    current_forget_fname_without_permu_id = os.path.basename(re.sub(r'_perm_sed\..', r'_perm_sed.__', forget_fname))
    #print (current_forget_fname_without_permu_id)
    line_list = []
    for filename in os.listdir(dir_name):
        forget_fname_without_permu_id = re.sub(r'_perm_sed\..', r'_perm_sed.__', filename)
        #print(forget_fname_without_permu_id)
        if filename.endswith('.txt') and forget_fname_without_permu_id == current_forget_fname_without_permu_id:
            with open(os.path.join(dir_name, filename)) as f:
                instance_str_list = f.readlines()
                line_list.append(instance_str_list)
    intersect_list = set.intersection(*map(set,line_list))
    sum_forget_fname = current_forget_fname_without_permu_id.replace("_perm_sed.__", "_perm_sed_sum")
    f = open(os.path.join(dir_name, sum_forget_fname), 'w')
    for line in intersect_list:
        print(line.strip(), file=f)
    f.close()



def confusion_matrix(result_a, fname=None):

    baseline = result_a[0]

    result_a = result_a[1:]

    nt = result_a.size(0)


    result = result_a[:,:nt]

    assert nt == result.size(1)
    # acc[t] equals result[t,t]
    acc = result.diag()
    fin = result[nt - 1]
    # bwt[t] equals result[T,t] - acc[t]
    bwt = result[nt - 1] - acc

    # fwt[t] equals result[t-1,t] - baseline[t]
    fwt = torch.zeros(nt)
    for t in range(1, nt):
        fwt[t] = result[t - 1, t] - baseline[t]

    if fname is not None:
        f = open(fname, 'w')

        print(' '.join(['%.4f' % r for r in baseline]), file=f)
        print('|', file=f)
        for row in range(result_a.size(0)):
            print(' '.join(['%.4f' % r for r in result_a[row]]), file=f)
        print('', file=f)
        # print('Diagonal Accuracy: %.4f' % acc.mean(), file=f)
        print('Final Accuracy: %.4f' % fin.mean(), file=f)
        print('Backward: %.4f' % bwt.mean(), file=f)
        print('Forward:  %.4f' % fwt.mean(), file=f)
        f.close()

    stats = []
    # stats.append(acc.mean())
    stats.append(fin.mean())
    stats.append(bwt.mean())
    stats.append(fwt.mean())

    return stats