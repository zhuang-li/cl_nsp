
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
from preprocess_data.utils import parse_prolog_query_helper, produce_data
# print(os.path.basename(your_path))

# overnight_entities = ["en.player", "en.player.kobe_bryant", "en.player.lebron_james" , "en.team.lakers", "en.team.cavaliers","en.position","en.position.point_guard","en.position.forward"]

path_prefix = "../../datasets/overnight/original_data"
dump_path_prefix = "../../datasets/overnight/question_split"

entities_to_process = ["2004 -1 -1", "2010 -1 -1"]

def absolute_file_paths(directory):
   for dirpath,_,filenames in os.walk(directory):
       for f in filenames:
           yield os.path.abspath(os.path.join(dirpath, f))

def read_process_write(file_path,dump_file_path):
    src_list = []
    tgt_list = []
    with open(file_path) as json_file:
        for line in json_file:
            line_list = line.split('\t')
            src = line_list[0]
            tgt = line_list[1]
            # make the jobs data format same with the geoquery
            tgt = tgt.strip()
            # src = re.sub(r"((?:c|s|m|r|co|n)\d)", VAR_NAME, src)
            tgt_split = tgt.split(' ')
            new_tgt = []
            flag = False
            for idx in range(len(tgt_split)):
                if flag:
                    flag = False
                    continue
                if tgt_split[idx] == 'call':
                    new_tgt.append(tgt_split[idx] + '-' + tgt_split[idx + 1])
                    flag = True
                else:
                    new_tgt.append(tgt_split[idx])
            src_list.append(src)
            tgt_list.append(' '.join(new_tgt))
    with open(dump_file_path, "w") as dump_file:
        for idx in range(len(src_list)):
            src_str = src_list[idx]
            tgt_str = tgt_list[idx]
            dump_file.write(src_str + '\t' + tgt_str + '\n')
    dump_file.close()



def process_overnight(train_file,test_file,dump_train_file,dump_test_file):
    read_process_write(train_file,dump_train_file)
    read_process_write(test_file,dump_test_file)

if __name__ == '__main__':
    data_formats = ['basketball', 'blocks', 'calendar', 'housing', 'publications', 'recipes', 'restaurants', 'socialnetwork']

    for data_format in data_formats:
        train_path = data_format + "_train.tsv"
        test_path = data_format + "_test.tsv"

        train_file = os.path.join(path_prefix, train_path)
        test_file = os.path.join(path_prefix, test_path)
        dump_train_file = os.path.join(dump_path_prefix, train_path)
        dump_test_file = os.path.join(dump_path_prefix, test_path)
        process_overnight(train_file,test_file,dump_train_file,dump_test_file)
    pass