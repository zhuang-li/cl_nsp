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

lang = 'nlmap'

path_prefix = "../../datasets/nlmap_city/continual_split"
dump_path_prefix = "../../datasets/nlmap_city/continual_split"
grammar_path_prefix = "../../datasets/nlmap_city/grammar"



def prepare_nlmap(train_file, test_file, test_length, diff_length):
    inputfile = open(file=test_file, mode='r')
    line_list = []
    for line in inputfile.readlines():
        line_list.append(line)

    test_len = int(len(line_list) - len(line_list) / 3)

    test_set = line_list[:test_len]

    dev_set = line_list[test_len:]

    dev_dump_path = data_format + "_dev.tsv"
    test_dump_path = data_format + "_test.tsv"

    dev_file = open(os.path.join(dump_path_prefix, dev_dump_path), 'w')
    test_file = open(os.path.join(dump_path_prefix, test_dump_path), 'w')

    for line in test_set:
        test_file.write(line)

    for line in dev_set:
        dev_file.write(line)




if __name__ == '__main__':
    data_formats = ['task0', 'task1', 'task2', 'task3', 'task4', 'task5', 'task6', 'task7']
    test_length = diff_length = 0
    for data_format in data_formats:
        train_path = data_format + "_train.tsv"
        test_path = data_format + "_test.tsv"
        train_file = os.path.join(path_prefix, train_path)
        test_file = os.path.join(path_prefix, test_path)
        prepare_nlmap(train_file,test_file, test_length, diff_length)
