import json
from nltk.corpus import stopwords

from grammar.vertex import RuleVertex
from grammar.consts import *
from grammar.db import normalize_trees
from grammar.utils import is_var, is_lit, is_predicate
from grammar.action import ReduceAction, GenAction
from grammar.db import create_template_to_leaves
from grammar.rule import product_rules_to_actions_bottomup,product_rules_to_actions_topdown
from grammar.rule import Action
from components.dataset import Example, data_augmentation, generate_augment_samples
from components.vocab import Vocab
from components.vocab import TokenVocabEntry

from components.vocab import *
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
import os
import sys
from itertools import chain
import re

try:
    import cPickle as pickle
except:
    import pickle
import numpy as np
from grammar.rule import get_reduce_action_length_set
from preprocess_data.utils import diff, produce_data, prune_test_set

lang = 'overnight'

path_prefix = "../../datasets/overnight/continual_split"
dump_path_prefix = "../../datasets/overnight/continual_split"
grammar_path_prefix = "../../datasets/overnight/grammar"



def prepare_overnight_lambda(train_file, test_file, test_length = 0, diff_length = 0):
    vocab_freq_cutoff = 0
    train_set, train_temp_db = produce_data(train_file, data_format, lang, turn_v_back=False,parse_mode='topdown', frequent=500)
    test_set, test_temp_db = produce_data(test_file, data_format, lang, turn_v_back=False,parse_mode='topdown', frequent=500)

    #train_len = int(len(train_set) - len(train_set)/7)

    #dev_set = train_set[train_len:]

    #train_set = train_set[:train_len]

    print("==============================================================")
    unit_diff_length = prune_test_set(train_set, test_set)
    diff_length += unit_diff_length
    print("Diff Examples :", unit_diff_length)

    test_length += len(test_set)

    # print (train_set[0].tgt_ast_t_seq)
    src_vocab = TokenVocabEntry.from_corpus([e.src_sent for e in train_set], size=5000, freq_cutoff=vocab_freq_cutoff)
    # generate vocabulary for the code tokens!
    code_tokens = [e.tgt_code for e in train_set]
    code_vocab = TokenVocabEntry.from_corpus(code_tokens, size=5000, freq_cutoff=0)

    nt_action_vocab = GenNTVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)

    test_nt_action_vocab = GenNTVocabEntry.from_corpus([e.tgt_actions for e in test_set], size=5000, freq_cutoff=0)

    print ("length of different actions", diff(list(nt_action_vocab.token2id.keys()),list(test_nt_action_vocab.token2id.keys())))

    t_action_vocab = GenTVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)
    t_action_vocab.read_grammar(grammar_path_prefix, data_format)
    # action_vocab = ActionVocabEntry.from_corpus([e.tgt_actions for e in train_set], size=5000, freq_cutoff=0)
    vocab = Vocab(source=src_vocab, code=code_vocab, nt_action=nt_action_vocab, t_action=t_action_vocab)

    augment_memory_dict = data_augmentation(train_set, t_action_vocab)

    generate_augment_samples(train_set, augment_memory_dict)

    print('generated vocabulary %s' % repr(vocab), file=sys.stderr)

    action_len = [len(e.tgt_actions) for e in chain(train_set, test_set)]
    print('Train set len: %d' % len(train_set))
    print('Test set len: %d' % len(test_set))
    print('Max action len: %d' % max(action_len), file=sys.stderr)
    print('Avg action len: %d' % np.average(action_len), file=sys.stderr)
    print('Actions larger than 100: %d' % len(list(filter(lambda x: x > 100, action_len))), file=sys.stderr)

    train_dump_path = data_format + "_train.bin"
    test_dump_path = data_format + "_test.bin"

    dev_dump_path = data_format + "_dev.bin"

    vocab_dump_path = data_format + ".vocab.freq.bin"

    pickle.dump(train_set, open(os.path.join(dump_path_prefix, train_dump_path), 'wb'))

    #pickle.dump(dev_set, open(os.path.join(dump_path_prefix, dev_dump_path), 'wb'))

    pickle.dump(test_set, open(os.path.join(dump_path_prefix, test_dump_path), 'wb'))
    pickle.dump(vocab, open(os.path.join(dump_path_prefix, vocab_dump_path), 'wb'))
    return test_length, diff_length

if __name__ == '__main__':
    # 'basketball'
    data_formats = ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']
    test_length = diff_length = 0
    for data_format in data_formats:
        train_path = data_format + "_train.tsv"
        test_path = data_format + "_test.tsv"
        train_file = os.path.join(path_prefix, train_path)
        test_file = os.path.join(path_prefix, test_path)
        test_length, diff_length = prepare_overnight_lambda(train_file,test_file, test_length, diff_length)
    print (diff_length/test_length)