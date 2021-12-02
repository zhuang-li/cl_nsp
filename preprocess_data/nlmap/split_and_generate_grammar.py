import io

from preprocess_data import utils
import os
import numpy as np
import random

def traverse_both_trees(tree, template_tree, grammar_dict, loc_dict, loc_w_list, poi_w_list, q_type_list):
    queue = []
    queue.append((tree, template_tree))
    loc_list = []
    poi_list = []
    q_list = []
    try:
        while queue:
            tree_tuple = queue.pop()
            node, node_template = tree_tuple
            if node_template.head == '\'_LOCATION\'':
                grammar_dict['_LOCATION'].add(node.head)
                loc_list.append(node.head)


            if node_template.head == '\'_POI\'':
                grammar_dict['_POI'].add(node.head)
                poi_list.append(node.head)
            #print (node)
            #print (node_template)
            if node.head == 'qtype':
                #assert len(node.children) == 1
                q_list.extend([child for child in node.children])
            for child_idx, child in enumerate(node.children):
                #print (child_idx)
                template_child = node_template.children[child_idx]
                queue.append((child, template_child))
    except:
        print (tree)
    loc_w_list.append(list(set(loc_list)))
    poi_w_list.append(list(set(poi_list)))
    #assert len(q_list) == 1, "q list is {0}".format(q_list)
    q_type_list.append(q_list)
    for loc in loc_list:
        if not loc in loc_dict:
            loc_dict[loc] = set()
        for poi in poi_list:
            loc_dict[loc].add(poi)
    return loc_list, poi_list, q_list

def read_src_tgt(path):
    data_list = []
    with open(path, "r") as lines:
        nlmap_tree_list = []
        nlmap_template_tree_list = []
        for line in lines:
            split_items = line.split('\t')
            src = split_items[0]
            tgt_list = split_items[1].split(' ')
            tgt_template_list = split_items[2].split(' ')
            nlmap_tree = utils.parse_nlmap_query_helper(tgt_list)
            nlmap_template_tree = utils.parse_nlmap_query_helper(tgt_template_list)
            nlmap_tree_list.append(nlmap_tree)
            nlmap_template_tree_list.append(nlmap_template_tree)
            data_str = src + '\t' + split_items[1]
            data_list.append(data_str)
        return nlmap_tree_list, nlmap_template_tree_list, data_list


def batch_iter(loc_list, batch_size, seed):
    index_arr = np.arange(len(loc_list))
    np.random.seed(seed)
    np.random.shuffle(index_arr)

    batch_num = int(np.ceil(len(loc_list) / float(batch_size)))
    for batch_id in range(batch_num):
        #print (batch_id)
        batch_ids = index_arr[batch_size * batch_id: batch_size * (batch_id + 1)]
        batch_examples = [loc_list[i] for i in batch_ids]

        yield batch_examples

def write_grammar(location_grammar_list, poi_grammar_list, write_dir_path):
    for task_index, loc_grammar_tokens in enumerate(location_grammar_list):
        sub_path = "task" + str(task_index) + ".grammar"
        path = os.path.join(write_dir_path, sub_path)
        txt_fp = io.open(path, "w")
        for loc_grammar_token in loc_grammar_tokens:
            txt_fp.write("$LOC" + '\t' + loc_grammar_token + '\t' +  ' '.join(' '.join(loc_grammar_token.split('_')).split('/'))[1:-1] + '\n')

        poi_grammar_tokens = poi_grammar_list[task_index]
        for poi_grammar_token in poi_grammar_tokens:
            txt_fp.write("$POI" + '\t' + poi_grammar_token + '\t' + ' '.join(' '.join(poi_grammar_token.split('_')).split('/'))[1:-1] + '\n')

        txt_fp.close()

def write_q_type_LOC_POI_grammar(grammar_dict_loc_list, grammar_dict_poi_list, write_dir_path):
    sub_path = "general.grammar"

    path = os.path.join(write_dir_path, sub_path)
    txt_fp = io.open(path, "w")

    for loc_grammar_token in grammar_dict_loc_list:
        txt_fp.write(
            "#LOC" + '\t' + loc_grammar_token + '\t' + ' '.join(' '.join(loc_grammar_token.split('_')).split('/'))[
                                                       1:-1] + '\n')

    for poi_grammar_token in grammar_dict_poi_list:
        txt_fp.write(
            "#POI" + '\t' + poi_grammar_token + '\t' + ' '.join(' '.join(poi_grammar_token.split('_')).split('/'))[
                                                       1:-1] + '\n')

    txt_fp.close()


def write_task_file(instance_ids, data_type, write_dir_path):
    for task_index, task_instance_ids in enumerate(instance_ids):
        sub_path = "task" + str(task_index) + "_" + data_type + ".tsv"
        path = os.path.join(write_dir_path, sub_path)
        txt_fp = io.open(path, "w")
        for instance_id in task_instance_ids:
            txt_fp.write(data_list[instance_id] + '\n')
        txt_fp.close()


def write_q_type_grammar(qtype_grammar, write_dir_path):
    for q_type, qtype_tokens in qtype_grammar.items():
        sub_path = q_type + ".grammar"
        path = os.path.join(write_dir_path, sub_path)
        txt_fp = io.open(path, "w")
        for qtype_token in qtype_tokens:
            txt_fp.write("$QT" + '\t' + str(qtype_token) + '\n')

        txt_fp.close()


def write_q_type_task_file(q_type_ids, data_type, write_dir_path, data_list):
    for q_type, task_instance_ids in q_type_ids.items():
        sub_path = q_type + "_" + data_type + ".tsv"
        path = os.path.join(write_dir_path, sub_path)
        txt_fp = io.open(path, "w")
        for instance_id in task_instance_ids:
            txt_fp.write(data_list[instance_id] + '\n')
        txt_fp.close()


def lists_overlap3(a, b):
    return bool(set(a) & set(b))

def lists_overlap(a, b, poi_id_first, poi_id_sec):
    print (poi_id_first, poi_id_sec)
    print (set(a) & set(b))
    return set(a) & set(b)


dir_path = "../../datasets/nlmap/nlmaps_v2/question_split/"
train_file_path = os.path.join(dir_path, "train.txt")
test_file_path = os.path.join(dir_path, "test.txt")
train_nlmap_tree_list, train_nlmap_template_tree_list, train_data_list = read_src_tgt(train_file_path)
test_nlmap_tree_list, test_nlmap_template_tree_list, test_data_list = read_src_tgt(test_file_path)
nlmap_tree_list = train_nlmap_tree_list + test_nlmap_tree_list
nlmap_template_tree_list = train_nlmap_template_tree_list + test_nlmap_template_tree_list
data_list = train_data_list + test_data_list

grammar_dict = {}
grammar_dict['_LOCATION'] = set()
grammar_dict['_POI'] = set()
grammar_dict['_QT'] = {}
loc_dict = {}

loc_w_list = []
poi_w_list = []
q_type_list = []
loc_id_dict = {}

for idx, tree in enumerate(nlmap_tree_list):
    loc_list, poi_list, q_list = traverse_both_trees(tree, nlmap_template_tree_list[idx], grammar_dict, loc_dict, loc_w_list, poi_w_list, q_type_list)
    loc_list = list(set(loc_list))
    poi_list = list(set(poi_list))
    assert len(loc_list) <= 1, " length of locations is {0}, the loc list is {1}".format(len(loc_list), loc_list)
    if len(loc_list) == 1:
        if loc_list[0] in loc_id_dict:
            loc_id_dict[loc_list[0]].append(idx)
        else:
            loc_id_dict[loc_list[0]] = []
            loc_id_dict[loc_list[0]].append(idx)

count = 0
q_type_id_dict = {}
for index, q_list in enumerate(q_type_list):
    if len(q_list) == 1:
        if q_list[0].head in q_type_id_dict:
            q_type_id_dict[q_list[0].head].append(index)
        else:
            q_type_id_dict[q_list[0].head] = []
            q_type_id_dict[q_list[0].head].append(index)
        if q_list[0].head in grammar_dict['_QT']:
            grammar_dict['_QT'][q_list[0].head].add(q_list[0])
        else:
            grammar_dict['_QT'][q_list[0].head] = set()
            grammar_dict['_QT'][q_list[0].head].add(q_list[0])
    else:
        count += 1
print ("q type is greater than 1 ", count)

task_num = 10
total_loc_list = list(grammar_dict['_LOCATION'])
total_loc_list.sort()
data_split_ids = []
instance_sum = 0
seed = 11
train_ids = []
test_ids = []
random.seed(seed)
split_ratio = 0.7
for loc_items in batch_iter(total_loc_list, task_num, seed):
    task_specific_ids = []
    train_task_specific_ids = []
    test_task_specific_ids = []
    for loc_item in loc_items:
        loc_id_dict[loc_item].sort()
        random.shuffle(loc_id_dict[loc_item])
        data_length = len(loc_id_dict[loc_item])
        train_size = int(data_length*split_ratio)
        test_size = data_length - train_size
        train_task_specific_ids.extend(loc_id_dict[loc_item][:train_size])
        test_task_specific_ids.extend(loc_id_dict[loc_item][train_size:])
        task_specific_ids.extend(loc_id_dict[loc_item])
    train_ids.append(train_task_specific_ids)
    test_ids.append(test_task_specific_ids)
    instance_sum += len(task_specific_ids)
    print (len(task_specific_ids))
print ("number of instance is : ",instance_sum)
#print (grammar_dict)
#print (loc_dict)
location_grammar_list = []
poi_grammar_list = []
for domain_index, train_domain_data_ids in enumerate(train_ids):
    location_grammar = []
    poi_grammar = []
    test_domain_data_ids = test_ids[domain_index]
    for train_data_id in train_domain_data_ids:
        location_grammar.extend(list(set(loc_w_list[train_data_id])))
        poi_grammar.extend(list(set(poi_w_list[train_data_id])))
    location_grammar_list.append(list(set(location_grammar)))
    poi_grammar_list.append(list(set(poi_grammar)))

init_loc_list = location_grammar_list[0]

init_poi_list = poi_grammar_list[0]

for loc_grammar in location_grammar_list[1:]:
    if lists_overlap3(init_loc_list, loc_grammar):
        raise ValueError
    init_loc_list = loc_grammar
print ("======================================================================================")
for poi_id_first, poi_grammar_first in enumerate(poi_grammar_list):
    for poi_id_sec, poi_grammar_second in enumerate(poi_grammar_list):
        if poi_id_sec > poi_id_first:
            overlap_pois = lists_overlap(poi_grammar_first, poi_grammar_second, poi_id_first, poi_id_sec)
            overlap_pois_list = list(overlap_pois)
            for poi in overlap_pois_list:
                poi_grammar_first.remove(poi)
                #poi_grammar_second.remove(poi)

for loc_id_first, loc_grammar_first in enumerate(location_grammar_list):
    for poi_id_sec, poi_grammar_second in enumerate(poi_grammar_list):
        overlap_pois = lists_overlap(loc_grammar_first, poi_grammar_second, loc_id_first, poi_id_sec)
        overlap_pois_list = list(overlap_pois)
        for poi in overlap_pois_list:
            poi_grammar_second.remove(poi)

print ("======================================================================================")
            #print ("")
for poi_id_first, poi_grammar_first in enumerate(poi_grammar_list):
    for poi_id_sec, poi_grammar_second in enumerate(poi_grammar_list):
        if poi_id_sec > poi_id_first:
            overlap_pois = lists_overlap(poi_grammar_first, poi_grammar_second, poi_id_first, poi_id_sec)
print ("======================================================================================")

for train_id_first, train_first in enumerate(train_ids):
    for train_id_sec, train_second in enumerate(train_ids):
        if train_id_sec > train_id_first:
            if lists_overlap3(train_first, train_second):
                raise ValueError

for test_id_first, test_first in enumerate(test_ids):
    for test_id_sec, test_second in enumerate(test_ids):
        if test_id_sec > test_id_first:
            if lists_overlap3(test_first, test_second):
                raise ValueError


print (location_grammar_list)
print (poi_grammar_list)

write_dir_path = "../../datasets/nlmap/continual_split/"
write_task_file(train_ids, "train", write_dir_path)
write_task_file(test_ids, "test", write_dir_path)

grammar_write_dir_path = "../../datasets/nlmap/grammar/"
write_grammar(location_grammar_list, poi_grammar_list, grammar_write_dir_path)


qtype_train_ids = {}
qtype_test_ids = {}
# merge findkey and nodeup
q_type_id_dict['findkey'].extend(list(q_type_id_dict['nodup']))
del q_type_id_dict['nodup']
grammar_dict['_QT']['findkey'].union(grammar_dict['_QT']['nodup'])
del grammar_dict['_QT']['nodup']

for q_type, id_list in q_type_id_dict.items():
    id_list.sort()
    random.shuffle(id_list)
    data_length = len(id_list)
    train_size = int(data_length * split_ratio)
    q_type_task_train_ids = id_list[:train_size]

    qtype_train_ids[q_type] = q_type_task_train_ids

    q_type_task_test_ids = id_list[train_size:]

    qtype_test_ids[q_type] = q_type_task_test_ids

write_q_type_dir_path = "../../datasets/nlmap_qtype/continual_split/"
q_type_grammar_write_dir_path = "../../datasets/nlmap_qtype/grammar/"
write_q_type_task_file(qtype_train_ids, "train", write_q_type_dir_path, data_list)
write_q_type_task_file(qtype_test_ids, "test", write_q_type_dir_path, data_list)
write_q_type_grammar(grammar_dict['_QT'], q_type_grammar_write_dir_path)
write_q_type_LOC_POI_grammar(grammar_dict['_LOCATION'], grammar_dict['_POI'], q_type_grammar_write_dir_path)