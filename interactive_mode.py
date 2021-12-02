import torch
from nltk import WordNetLemmatizer, TweetTokenizer

import evaluation
from common.registerable import Registrable
from common.utils import init_arg_parser
import numpy as np
import six.moves.cPickle as pickle
import os
from continual_model.domain_overnight import OvernightDomain
from continual_model.seq2seq_topdown import Seq2SeqModel
from dropout_few_shot_exp import get_model_loss, update_action_freq, init_vertex_embedding_with_glove
from model.seq2seq_final_few_shot import Seq2SeqModel
from components.evaluator import DefaultEvaluator, ActionEvaluator
from components.dataset import Example, Dataset
from model.utils import GloveHelper
from preprocess_data.utils import lemmatize, pre_few_shot_vocab, produce_data


def init_config():
    args = arg_parser.parse_args()

    # seed the RNG
    torch.manual_seed(args.seed)
    if args.use_cuda:
        torch.cuda.manual_seed(args.seed)
    np.random.seed(int(args.seed * 13 / 7))

    return args

def prepare_data(train_set,train_vocab, support_file, data_type='atis_lambda', lang='lambda'):
    train_src_list = [e.src_sent for e in train_set]
    train_action_list = [e.tgt_actions for e in train_set]

    support_set, support_db = produce_data(support_file, data_type, lang,rule_type='ProductionRuleBLB',normlize_tree=True,previous_src_list=train_src_list, previous_action_seq=train_action_list)

    #print([list(train_vocab.entity.token2id.keys())] )
    #print(support_db.entity_list)

    support_vocab = pre_few_shot_vocab(data_set=train_set + support_set,
                                       entity_list=[list(train_vocab.entity.token2id.keys())] + support_db.entity_list,
                                       variable_list=[list(train_vocab.variable.token2id.keys())] + support_db.variable_list,
                                       disjoint_set=support_set)



    #pickle.dump(support_set, open(os.path.join(dump_path_prefix, 'support.bin'), 'wb'))

    #pickle.dump(support_vocab, open(os.path.join(dump_path_prefix, 'support_vocab.bin'), 'wb'))

    return support_set, support_vocab

def fine_tune(dataset_name, args):
    """Maximum Likelihood Estimation"""
    if dataset_name == 'atis':
        dump_path_prefix = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/atis/temp_support/'
        train_set = Dataset.from_bin_file("/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/atis/query_split/few_shot_split_random_5_predi/shuffle_0_shot_1/train.bin")
        train_vocab = pickle.load(open("/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/atis/query_split/few_shot_split_random_5_predi/shuffle_0_shot_1/train_vocab.bin", 'rb'))
        support_file = "/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/atis/temp_support/support.txt"
        support_set, support_vocab = prepare_data(train_set.examples, train_vocab, support_file, data_type='atis_lambda', lang='lambda')
        args.dropout = 0.6
        args.use_cuda = True
        args.dropout_i = 0
        args.hidden_size = 256
        args.embed_size = 200
        args.att_reg = 1
        args.att_filter = 0
        args.action_embed_size = 128
        args.lr_decay = 0.985
        args.proto_dropout = 0.5
        args.lr_decay_after_epoch = 40
        args.max_epoch = 40
        args.patience = 1000  # disable patience since we don't have dev set
        args.batch_size = 2
        args.beam_size = 1
        args.n_way = 30
        args.k_shot = 1
        args.query_num = 3
        args.lr = 0.0005
        args.alpha = 0.95
        args.label_smoothing = 0.2
        args.relax_factor = 10
        args.lstm = 'lstm'
        args.optimizer = 'Adam'
        args.clip_grad_mode = 'norm'
        args.parser = 'seq2seq_few_shot'
        args.sup_attention = True
        args.glorot_init = True
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/saved_models/atis/query_split/few_shot_split_random_5_predi/shuffle_0_shot_1/model.atis.pre_train.lstm.hid256.embed200.act128.att_reg1.drop0.5.dropout_i0.lr_decay0.985.lr_dec_aft20.beam1.sup_turn2.train_vocab.bin.train.bin.pat1000.max_ep80.batch64.lr0.0025.p_dropout0.5.parserseq2seq_few_shot.suffixpre_no_copy.bin'
        args.save_to = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/atis/temp_support/temp_atis_model.bin'
        args.glove_embed_path = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/embedding/glove/glove.6B.200d.txt'
    elif dataset_name == 'geo':
        train_set = Dataset.from_bin_file(
            "/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/geo/lambda/query_split/few_shot_split_random_5_predi/shuffle_0_shot_1/train.bin")
        train_vocab = pickle.load(open(
            "/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/geo/lambda/query_split/few_shot_split_random_5_predi/shuffle_0_shot_1/train_vocab.bin",
            'rb'))
        support_file = "/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/geo/temp_support/support.txt"
        support_set, support_vocab = prepare_data(train_set.examples, train_vocab, support_file,
                                                  data_type='geo_lambda', lang='lambda')
        args.dropout = 0.6
        args.use_cuda = True
        args.dropout_i = 0
        args.hidden_size = 256
        args.embed_size = 200
        args.att_reg = 0.1
        args.att_filter = 0
        args.action_embed_size = 128
        args.lr_decay = 0.985
        args.proto_dropout = 0.5
        args.lr_decay_after_epoch = 60
        args.max_epoch = 80
        args.patience = 1000  # disable patience since we don't have dev set
        args.batch_size = 2
        args.beam_size = 1
        args.n_way = 30
        args.k_shot = 1
        args.query_num = 3
        args.lr = 0.0005
        args.alpha = 0.95
        args.label_smoothing = 0.2
        args.relax_factor = 10
        args.lstm = 'lstm'
        args.optimizer = 'Adam'
        args.clip_grad_mode = 'norm'
        args.parser = 'seq2seq_few_shot'
        args.sup_attention = True
        args.glorot_init = True
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/saved_models/geo/lambda/query_split/few_shot_split_random_5_predi/shuffle_0_shot_1/model.geo.pre_train.lstm.hid256.embed200.act128.att_reg1.drop0.5.dropout_i0.lr_decay0.985.lr_dec_aft20.beam1.sup_turn2.train_vocab.bin.train.bin.max_ep200.batch64.lr0.0025.p_drop0.5.metricdot.parserseq2seq_few_shot.suffixpre_point_fix.bin'
        args.save_to = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/geo/temp_support/temp_geo_model.bin'
        args.glove_embed_path = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/embedding/glove/glove.6B.200d.txt'
    elif dataset_name == 'jobs':
        train_set = Dataset.from_bin_file(
            "/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/jobs/query_split/few_shot_split_random_3_predishuffle_0_shot_1/train.bin")
        train_vocab = pickle.load(open(
            "/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/jobs/query_split/few_shot_split_random_3_predishuffle_0_shot_1/train_vocab.bin",
            'rb'))
        support_file = "/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/jobs/temp_support/support.txt"
        support_set, support_vocab = prepare_data(train_set.examples, train_vocab, support_file,
                                                  data_type='job_prolog', lang='prolog')
        args.dropout = 0.6
        args.use_cuda = True
        args.dropout_i = 0
        args.hidden_size = 256
        args.embed_size = 200
        args.att_reg = 0.001
        args.att_filter = 0
        args.action_embed_size = 128
        args.lr_decay = 0.985
        args.proto_dropout = 0.5
        args.lr_decay_after_epoch = 60
        args.max_epoch = 60
        args.patience = 1000  # disable patience since we don't have dev set
        args.batch_size = 2
        args.beam_size = 1
        args.n_way = 30
        args.k_shot = 1
        args.query_num = 3
        args.lr = 0.0005
        args.alpha = 0.95
        args.label_smoothing = 0.15
        args.relax_factor = 10
        args.lstm = 'lstm'
        args.optimizer = 'Adam'
        args.clip_grad_mode = 'norm'
        args.parser = 'seq2seq_few_shot'
        args.sup_attention = False
        args.glorot_init = True
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/saved_models/jobs/query_split/few_shot_split_random_3_predishuffle_0_shot_1/model.jobs.sup.pre_train.lstm.hid256.embed200.act128.att_reg0.01.drop0.5.dropout_i0.lr_decay0.985.lr_dec_aft20.beam1.sup_turn2.pat1000.max_ep200.batch64.lr0.0025.p_dropout0.5.parserseq2seq_few_shot.suffixpre_sup_no_copy.bin'
        args.save_to = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/jobs/temp_support/temp_jobs_model.bin'
        args.glove_embed_path = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/embedding/glove/glove.6B.200d.txt'

        # load in train/dev set
    train_set = Dataset(support_set)

    if args.dev_file:
        dev_set = Dataset.from_bin_file(args.dev_file)
    else:
        dev_set = Dataset(examples=[])

    vocab = support_vocab

    #print("register parser ...")
    parser_cls = Registrable.by_name(args.parser)  # TODO: add arg

    #print("fine-tuning started ...")
    if args.load_model:
        #print('load model from [%s]' % args.load_model, file=sys.stderr)
        #print(
        #    "=============================================================loading model===============================================================")
        model, src_vocab, vertex_vocab = parser_cls.load(model_path=args.load_model, use_cuda=args.use_cuda,
                                                         loaded_vocab=vocab, args=args)
        #print("load pre-trained word embedding (optional)")
        #print(args.glove_embed_path)
        if args.glove_embed_path and (src_vocab is not None):
            # print (args.embed_size)
            #print('load glove embedding from: %s' % args.glove_embed_path, file=sys.stderr)
            glove_embedding = GloveHelper(args.glove_embed_path, args.embed_size)
            glove_embedding.load_pre_train_to(model.src_embed, src_vocab)

        if args.init_vertex and args.glove_embed_path and vertex_vocab is not None:
            #print(
            #    "================================================= init vertex ============================================")
            #print(vertex_vocab)
            word_ids, non_init_vertex = init_vertex_embedding_with_glove(model.ast_embed, vertex_vocab,
                                                                         vocab.predicate_tokens, args)
            #print("non init vertex")
            #print(non_init_vertex)
    else:
        model = parser_cls(vocab, args)
    #print("setting model to fintuning mode")
    model.few_shot_mode = 'fine_tune'
    model.train()

    # init_vertex_embedding_with_glove(model.ast_embed, model.vertex_vocab.token2id, vocab.predicate_tokens, args)

    evaluator = Registrable.by_name(args.evaluator)(args=args)
    if args.use_cuda and torch.cuda.is_available():
        model.cuda()
    optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
    if args.optimizer == 'RMSprop':
        optimizer = optimizer_cls(model.parameters(), lr=args.lr, alpha=args.alpha)
    else:
        optimizer = optimizer_cls(model.parameters(), lr=args.lr)

    #print('begin training, %d training examples, %d dev examples' % (len(train_set), len(dev_set)), file=sys.stderr)
    #print('vocab: %s' % repr(vocab), file=sys.stderr)

    epoch = train_iter = 0
    report_loss = report_examples = report_sup_att_loss = report_att_reg_loss = report_ent_loss = 0.
    history_dev_scores = []
    num_trial = patience = 0
    train_set.examples.sort(key=lambda e: -len(e.src_sent))
    update_action_freq(model.action_freq, [e.tgt_actions for e in train_set.examples])

    # random_init_action_embedding = model.action_embedding.weight.data[-7:]

    #rint("init action embedding")
    # print (args.metric)
    if args.metric == 'matching':
        model.proto_train_action_set = dict()
        # print ("=================================")
        # print (model.proto_train_action_set)
    model.init_action_embedding(train_set.examples, few_shot_mode='fine_tune')

    # if args.num_exemplars_per_task:
    # subselect_examples = train_set.random_sample_batch_iter(args.num_exemplars_per_task)
    # replay_dataset = Dataset(subselect_examples)
    # print (model.proto_train_action_set.keys())


    while True:
        if not args.max_epoch == 0:
            epoch += 1
            #epoch_begin = time.time()

            for batch_examples in train_set.batch_iter(batch_size=args.batch_size, shuffle=True):
                train_iter += 1
                optimizer.zero_grad()

                loss, report_loss, report_examples, report_sup_att_loss, report_att_reg_loss, report_ent_loss = \
                    get_model_loss(model, batch_examples, args, report_loss, report_examples, report_sup_att_loss,
                                   report_att_reg_loss, report_ent_loss)

                loss.backward()

                # clip gradient
                if args.clip_grad > 0.:
                    if args.clip_grad_mode == 'norm':
                        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                    elif args.clip_grad_mode == 'value':
                        grad_norm = torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_grad)

                optimizer.step()

                if train_iter % args.log_every == 0:
                    log_str = '[Iter %d] encoder loss=%.5f' % (train_iter, report_loss / report_examples)
                    if args.sup_attention:
                        log_str += ' supervised attention loss=%.5f' % (report_sup_att_loss / report_examples)
                        report_sup_att_loss = 0.
                    if args.att_reg:
                        log_str += ' attention regularization loss=%.5f' % (report_att_reg_loss / report_examples)
                        report_att_reg_loss = 0.
                    log_str += ' entity attention loss=%.5f' % (report_ent_loss / report_examples)
                    #print(log_str, file=sys.stderr)
                    report_loss = report_examples = report_ent_loss = 0.

            #print('[Epoch %d] epoch elapsed %ds' % (epoch, time.time() - epoch_begin), file=sys.stderr)

        if args.save_all_models:
            # model.init_action_embedding(train_set.examples)
            model_file = args.save_to + '.iter%d.bin' % train_iter
            #print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)

        # perform validation
        is_better = False
        if args.dev_file:
            if epoch % args.valid_every_epoch == 0:
                model.eval()
                #print('[Epoch %d] begin validation' % epoch, file=sys.stderr)
                #eval_start = time.time()
                eval_results = evaluation.evaluate(dev_set.examples, model, evaluator, args,
                                                   verbose=True, eval_top_pred_only=args.eval_top_pred_only)
                dev_score = eval_results[evaluator.default_metric]

                #print('[Epoch %d] evaluate details: %s, dev %s: %.5f (took %ds)' % (
                #    epoch, eval_results,
                #    evaluator.default_metric,
                #    dev_score,
                #    time.time() - eval_start), file=sys.stderr)

                is_better = history_dev_scores == [] or dev_score > max(history_dev_scores)
                history_dev_scores.append(dev_score)
                model.train()
        else:
            is_better = True

        if args.decay_lr_every_epoch and epoch > args.lr_decay_after_epoch and epoch % args.valid_every_epoch == 0:
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            #print('decay learning rate to %f' % lr, file=sys.stderr)

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        if is_better:
            # model.init_action_embedding(train_set.examples)
            patience = 0
            model_file = args.save_to + '.bin'
            #print('save the current model ..', file=sys.stderr)
            #print('save model to [%s]' % model_file, file=sys.stderr)
            model.save(model_file)
            # also save the optimizers' state
            torch.save(optimizer.state_dict(), args.save_to + '.optim.bin')

        elif patience < args.patience and epoch >= args.lr_decay_after_epoch:
            patience += 1
            #print('hit patience %d' % patience, file=sys.stderr)

        if epoch == args.max_epoch:
            #print('reached max epoch, stop!', file=sys.stderr)
            break

        if patience >= args.patience and epoch >= args.lr_decay_after_epoch:
            num_trial += 1
            #print('hit #%d trial' % num_trial, file=sys.stderr)
            if num_trial == args.max_num_trial:
                #print('early stop!', file=sys.stderr)
                break

            # decay lr, and restore from previously best checkpoint
            lr = optimizer.param_groups[0]['lr'] * args.lr_decay
            #print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

            # load model
            params = torch.load(args.save_to + '.bin', map_location=lambda storage, loc: storage)
            model.load_state_dict(params['state_dict'])
            if args.use_cuda: model = model.cuda()

            # load optimizers
            if args.reset_optimizer:
                #print('reset optimizer', file=sys.stderr)
                reset_optimizer_cls = eval('torch.optim.%s' % args.optimizer)  # FIXME: this is evil!
                if args.optimizer == 'RMSprop':
                    optimizer = reset_optimizer_cls(model.parameters(), lr=args.lr, alpha=args.alpha)
                else:
                    optimizer = reset_optimizer_cls(model.parameters(), lr=args.lr)
                # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            else:
                #print('restore parameters of the optimizers', file=sys.stderr)
                optimizer.load_state_dict(torch.load(args.save_to + '.optim.bin'))

            # set new lr
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # reset patience
            patience = 0


def decode(dataset_name, args, domain=None):

    txt = input("Type the text you want to parse: ")
    nltk_lemmer = WordNetLemmatizer()
    tknzr = TweetTokenizer()
    src = txt.lower()
    src = src.strip()
    src_split = tknzr.tokenize(src)
    src_text = lemmatize(src_split, nltk_lemmer)
    txt_example = Example(src_sent=src_text, tgt_code=['ci0'], tgt_ast=[],
            tgt_actions=[],
            idx=0, meta=None)
    #assert args.load_model
    #print (args.lang)
    #print('load model from [%s]' % args.load_model, file=sys.stderr)
    if dataset_name == 'atis':
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/saved_models/atis/query_split/few_shot_split_random_5_predi/shuffle_0_shot_1/model.atis.pre_train.lstm.hid256.embed200.act128.att_reg1.drop0.5.dropout_i0.lr_decay0.985.lr_dec_aft20.beam1.sup_turn2.train_vocab.bin.train.bin.pat1000.max_ep80.batch64.lr0.0025.p_dropout0.5.parserseq2seq_few_shot.suffixpre_no_copy.bin'
        args.lang = 'atis_lambda'
        args.parser = 'seq2seq_few_shot'
    elif dataset_name == 'atis-finetuned':
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/atis/temp_support/temp_atis_model.bin.bin'
        args.lang = 'atis_lambda'
        args.parser = 'seq2seq_few_shot'
    elif dataset_name == 'jobs':
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/saved_models/jobs/query_split/few_shot_split_random_3_predishuffle_0_shot_1/model.jobs.sup.pre_train.lstm.hid256.embed200.act128.att_reg0.01.drop0.5.dropout_i0.lr_decay0.985.lr_dec_aft20.beam1.sup_turn2.pat1000.max_ep200.batch64.lr0.0025.p_dropout0.5.parserseq2seq_few_shot.suffixpre_sup_no_copy.bin'
        args.lang = 'job_prolog'
        args.parser = 'seq2seq_few_shot'
    elif dataset_name == 'jobs-finetuned':
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/jobs/temp_support/temp_jobs_model.bin.bin'
        args.lang = 'job_prolog'
        args.parser = 'seq2seq_few_shot'
    elif dataset_name == 'geo':
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/saved_models/geo/lambda/query_split/few_shot_split_random_5_predi/shuffle_0_shot_1/model.geo.pre_train.lstm.hid256.embed200.act128.att_reg1.drop0.5.dropout_i0.lr_decay0.985.lr_dec_aft20.beam1.sup_turn2.train_vocab.bin.train.bin.max_ep200.batch64.lr0.0025.p_drop0.5.metricdot.parserseq2seq_few_shot.suffixpre_point_fix.bin'
        args.lang = 'geo_lambda'
        args.parser = 'seq2seq_few_shot'
    elif dataset_name == 'geo-finetuned':
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/datasets/geo/temp_support/temp_geo_model.bin.bin'
        args.lang = 'geo_lambda'
        args.parser = 'seq2seq_few_shot'
    elif dataset_name == 'overnight':
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/saved_models/overnight/continual_split/model.overn.lstm.hid256.embed256.act128.drop0.lr_decay0.985.lr_dec_aft200.beam1.pat1000.max_ep50.batch64.lr0.0025.glo.ls0.1.seed.cgmnorm.porigin.sa_merandom.bin'
        args.lang = 'overnight_lambda'
        args.parser = 'seq2seq_topdown'
    elif dataset_name == 'nlmap':
        args.load_model = '/home/zhuang-li/Workshop/cross_template_semantic_parsing/saved_models/nlmap/continual_split/model.nlmap.lstm.hid256.embed200.act128.drop0.lr_decay0.985.lr_dec_aft20.beam1.pat1000.max_ep50.batch64.lr0.0025.glo.ls0.1.seed.cgmnorm.porigin.sa_merandom.bin'
        args.lang = 'nlmap'
        args.parser = 'seq2seq_topdown'

    args.use_cuda = True
    args.mode = 'test'
    args.beam_size = 1
    args.clip_grad_mode = 'norm'
    args.evaluator = 'default_evaluator'
    args.decode_max_time_step = 50
    args.att_reg = 1
    params = torch.load(args.load_model, map_location=lambda storage, loc: storage)
    saved_args = params['args']
    saved_args.use_cuda = args.use_cuda
    #print(saved_args)

    parser_cls = Registrable.by_name(args.parser)
    if dataset.startswith('atis') or dataset.startswith('jobs') or dataset.startswith('geo'):
        parser, src_vocab, vertex_vocab = parser_cls.load(model_path=args.load_model, use_cuda=args.use_cuda, args=args)
    else:
        parser = parser_cls.load(model_path=args.load_model, use_cuda=args.use_cuda)
    parser.few_shot_mode = 'fine_tune'
    parser.eval()
    vocab = parser.vocab
    evaluator = Registrable.by_name(args.evaluator)(args=args)
    eval_results, decode_results = evaluation.evaluate([txt_example], parser, evaluator, args,
                                                       verbose=args.verbose, return_decode_result=True)
    if dataset_name.startswith('atis') or dataset_name.startswith('geo'):
        res_lf = decode_results[0][0].to_lambda_template
    elif dataset_name.startswith('jobs'):
        res_lf = decode_results[0][0].to_prolog_template
    elif dataset_name == 'overnight':
        res_lf = decode_results[0][0].to_lambda_template
        odomain = OvernightDomain(domain)
        hyp_lf_normalized = odomain.normalize([res_lf])
        hyp_denotatations = odomain.obtain_denotations(hyp_lf_normalized)
        print("The query result is :")
        print(hyp_denotatations)


    elif dataset_name == 'nlmap':
        res_lf = decode_results[0][0].to_prolog_template
    print("The parse result :")
    print(res_lf)
    del parser

if __name__ == '__main__':
    arg_parser = init_arg_parser()
    args = init_config()
    while True:
        dataset = input("Type the dataset on which you trained your model [atis, jobs, geo, overnight, nlmap, atis-finetuned, geo-finetuned, jobs-finetuned] :")
        # Note that in version 3, the print() function
        # requires the use of parenthesis.
        print("Is this the dataset? ", dataset)
        if dataset == 'atis' or dataset == 'jobs' or dataset == 'geo':
            do_finetune = input("Few-shot Fine-tune or not [y, n] :")
            if do_finetune == 'y':
                fine_tune(dataset, args)
            else:
                decode(dataset, args)
        elif dataset == 'overnight':
            domain = input("Type the domain on which you trained your model [basketball, blocks, calendar, housing, publications, recipes, restaurants, socialnetwork] :")
            # Note that in version 3, the print() function
            # requires the use of parenthesis.
            print("Is this the domain? ", domain)
            decode(dataset, args, domain=domain)
        elif dataset == 'nlmap':
            decode(dataset, args)
        elif dataset == 'atis-finetuned' or dataset == 'geo-finetuned' or dataset == 'jobs-finetuned':
            decode(dataset, args)
        else:
            print ("wrong name of the dataset")