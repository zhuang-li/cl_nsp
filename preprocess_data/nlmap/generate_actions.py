import json
from nltk.corpus import stopwords
from grammar.vertex import RuleVertex
from grammar.consts import *
from grammar.db import normalize_trees
from grammar.utils import is_var, is_lit
from grammar.action import ReduceAction, GenAction
from grammar.db import create_template_to_leaves
from grammar.rule import product_rules_to_actions
from grammar.rule import Action
from components.dataset import Example
from components.vocab import Vocab
from components.vocab import TokenVocabEntry
from components.vocab import ActionVocabEntry, GeneralActionVocabEntry, GenVocabEntry, ReVocabEntry, VertexVocabEntry
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import os
import sys
from itertools import chain
import re

from preprocess_data.utils import prune_test_set

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from grammar.rule import get_reduce_action_length_set

path_prefix = "../../datasets/nlmap/question_split/"
dump_path_prefix = "../../datasets/nlmap/question_split/stack_lstm/"
variable_phrases_path = "../../datasets/nlmap/variable_phrases.txt"
key_phrases_path = "../../datasets/nlmap/key_phrases.txt"

stemmer = SnowballStemmer("english")
lemmer = WordNetLemmatizer()

def parse_prolog_query_helper(elem_list, var_name = 'A', separator_set = set([',','(',')',';'])):
    with open(variable_phrases_path) as f:
        phrase_content = f.readlines()
    phrase_content = [phrase.strip() for phrase in phrase_content]
    phrase_content = set(phrase_content)

    with open(key_phrases_path) as f:
        key_content = f.readlines()
    key_content = [phrase.strip() for phrase in key_content]
    key_content = set(key_content)

    root = RuleVertex(ROOT)
    root.is_auto_nt = True
    root.position = 0
    depth = 0
    i = 0
    current = root
    node_pos = 1
    for elem in elem_list:
        #print("{} : {} ".format(elem, current.head))
        if elem == '(':
            depth += 1
            if i > 0:
                last_elem = elem_list[i-1]
                if last_elem in separator_set:
                    child = RuleVertex(IMPLICIT_HEAD)
                    child.parent = current
                    current.add(child)
                    child.is_auto_nt = True
                    current = child
                    child.position = node_pos
                    node_pos += 1
                else:
                    current = current.children[-1]

        elif elem == ')':
            current = current.parent
            depth -= 1
        elif not elem == ',':
            is_literal = is_lit(elem, dataset='nlmap',elem_set=phrase_content)
            is_variable = is_var(elem, dataset='nlmap',elem_set=key_content)
            if is_literal:
                #print (elem)
                norm_elem = SLOT_PREFIX
            elif is_variable:
                norm_elem = VAR_NAME
            else:
                norm_elem = elem
            child = RuleVertex(norm_elem)
            if is_variable:
                child.original_var = elem
            elif is_literal:
                child.original_entity = elem

            child.parent = current
            current.add(child)
            child.position = node_pos
            node_pos += 1
        i+=1

    return root

def parse_prolog_query(bracket_query):
    root = parse_prolog_query_helper(bracket_query)
    return root


def produce_data(data_filepath):
    example = []
    tgt_code_list = []
    src_list = []
    with open(data_filepath) as json_file:
        for line in json_file:
            src, tgt = line.split('\t')
            # make the jobs data format same with the geoquery
            tgt = tgt.strip()
            src = src.strip().lower()
            src_list.append([stemmer.stem(token) for token in src.split(' ')])
            tgt_code_list.append(tgt.split(' '))

    tgt_asts = [parse_prolog_query(t) for t in tgt_code_list]

    temp_db = normalize_trees(tgt_asts)
    leaves_list = create_template_to_leaves(tgt_asts, temp_db)
    tid2config_seq = product_rules_to_actions(tgt_asts, leaves_list, temp_db, True, "ProductionRuleBL",turn_v_back=True)
    reduce_action_length_set = get_reduce_action_length_set(tid2config_seq)

    assert len(src_list) == len(tgt_code_list), "instance numbers should be consistent"
    assert isinstance(list(temp_db.action2id.keys())[0], Action), "action2id must contain actions"

    for i, src_sent in enumerate(src_list):
        # todo change it back
        # temp_list = [type(action).__name__ if isinstance(action, ReduceAction) else action for action in tid2config_seq[i]]
        example.append(
            Example(src_sent=src_sent, tgt_code=tgt_code_list[i], tgt_ast=tgt_asts[i], tgt_actions=tid2config_seq[i],
                    idx=i, meta=None))

        assert len(tid2config_seq[i]) == len(example[i].tgt_ast_seq), "the node head length must be equal to the action length"
    example.sort(key=lambda x : len(x.tgt_actions))
    return example, temp_db.action2id


def prepare_nlmap_prolog(train_file, test_file):
    vocab_freq_cutoff = 0
    train_set, train_action2id = produce_data(train_file)
    test_set, test_action2id = produce_data(test_file)

    print ("==============================================================")
    print("Diff Examples :", prune_test_set(train_set, test_set))

    src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)
    # generate vocabulary for the code tokens!
    code_tokens = [e.tgt_code for e in train_set]
    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)
    action_vocab = ActionVocabEntry.from_action2id(action2id=train_action2id)
    general_action_vocab = GeneralActionVocabEntry.from_action_vocab(action_vocab.token2id)
    gen_vocab = GenVocabEntry.from_action2id(action2id=train_action2id)
    reduce_vocab = ReVocabEntry.from_action2id(action2id=train_action2id)
    vertex_vocab = VertexVocabEntry.from_example_list(train_set)
    vocab = Vocab(source=src_vocab, code=code_vocab, action=action_vocab, general_action=general_action_vocab,
                  gen_action=gen_vocab, re_action=reduce_vocab, vertex = vertex_vocab)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, test_set)]
    print('Train set len: %d' % len(train_set))
    print('Test set len: %d' % len(test_set))
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

    pickle.dump(train_set, open(os.path.join(dump_path_prefix, 'train.bin'), 'wb'))
    pickle.dump(test_set, open(os.path.join(dump_path_prefix, 'test.bin'), 'wb'))
    pickle.dump(vocab, open(os.path.join(dump_path_prefix, 'vocab.freq2.bin'), 'wb'))


if __name__ == '__main__':
    train_file = os.path.join(path_prefix, 'train.txt')
    test_file = os.path.join(path_prefix, 'test.txt')
    prepare_nlmap_prolog(train_file, test_file)
    pass
