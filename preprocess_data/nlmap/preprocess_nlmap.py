from nltk.tokenize import TweetTokenizer
import io


def read_en_lines(lines):
    tknzr = TweetTokenizer()
    result = []
    for line in lines:
        result.append(tknzr.tokenize(line))
    return result


def read_mrl_lines(lines):
    result = []
    for line in lines:
        tgt = ''
        flag = True
        for i, ch in enumerate(line.strip()):
            #if ch == '\'':
            #    flag = not flag
            if (ch == '(' or ch == ')' or ch == ','):
                if tgt[-1] == ' ':
                    tgt = tgt + ch + ' '
                else:
                    tgt = tgt + ' ' + ch + ' '
            elif ch == ' ':
                tgt = tgt + "_"
            else:
                tgt = tgt + ch
        tgt_list = tgt.strip().split(' ')
        result.append(tgt_list)
    return result


def read_nlmap_data(en_path, mrl_path, loc_path):
    with open(en_path, "r") as lines:
        en_result = read_en_lines(lines)
    with open(mrl_path, "r") as lines:
        mrl_result = read_mrl_lines(lines)
    with open(loc_path, "r") as lines:
        loc_result = read_mrl_lines(lines)
    return en_result, mrl_result, loc_result


def write_to_txt_file(src_list, tgt_list, loc_list, fp):
    fp.write(' '.join(src_list) + '\t' + ' '.join(tgt_list) + '\t' + ' '.join(loc_list) + '\n')


def process_results(src_result, tgt_result, loc_result, path):
    txt_fp = io.open(path, "w")
    for i, src_list in enumerate(src_result):
        tgt_list = tgt_result[i]
        loc_list = loc_result[i]
        write_to_txt_file(src_list, tgt_list, loc_list, txt_fp)


dir_path = "../../datasets/nlmap/nlmaps_v2/split_1_train_dev_test/"

train_en_path = dir_path + "nlmaps.v2.train.en"
train_mrl_path = dir_path + "nlmaps.v2.train.mrl"
train_loc_path = dir_path + "nlmaps.v2.train.loc.mrl"

dev_en_path = dir_path + "nlmaps.v2.dev.en"
dev_mrl_path = dir_path + "nlmaps.v2.dev.mrl"
dev_loc_path = dir_path + "nlmaps.v2.dev.loc.mrl"

test_en_path = dir_path + "nlmaps.v2.test.en"
test_mrl_path = dir_path + "nlmaps.v2.test.mrl"
test_loc_path = dir_path + "nlmaps.v2.test.loc.mrl"

train_txt = dir_path + "train.txt"
test_txt = dir_path + "test.txt"

train_en_result, train_mrl_result, train_loc_result = read_nlmap_data(train_en_path, train_mrl_path, train_loc_path)
dev_en_result, dev_mrl_result, dev_loc_result = read_nlmap_data(dev_en_path, dev_mrl_path, dev_loc_path)
test_en_result, test_mrl_result, test_loc_result = read_nlmap_data(test_en_path, test_mrl_path, test_loc_path)

train_en_result = train_en_result + dev_en_result
train_mrl_result = train_mrl_result + dev_mrl_result
train_loc_result = train_loc_result + dev_loc_result


process_results(train_en_result, train_mrl_result, train_loc_result, train_txt)
process_results(test_en_result, test_mrl_result, test_loc_result, test_txt)