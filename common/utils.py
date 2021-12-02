# coding=utf-8
import argparse

import os
import logging
import time


def config_logger(log_prefix):
    logger = logging.getLogger()
    logging.basicConfig(format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s',
        level=logging.INFO)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.getcwd() + '/logs/' + log_prefix + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + rq + '.log'
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    return logger

class cached_property(object):
    """ A property that is only computed once per instance and then replaces
        itself with an ordinary attribute. Deleting the attribute resets the
        property.

        Source: https://github.com/bottlepy/bottle/commit/fa7733e075da0d790d809aa3d2f53071897e6f76
        """

    def __init__(self, func):
        self.__doc__ = getattr(func, '__doc__')
        self.func = func

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = obj.__dict__[self.func.__name__] = self.func(obj)
        return value


def init_arg_parser():
    arg_parser = argparse.ArgumentParser()

    #### General configuration ####
    arg_parser.add_argument('--seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--p_seed', default=0, type=int, help='Random seed')
    arg_parser.add_argument('--use_cuda', action='store_true', default=False, help='Use gpu')
    arg_parser.add_argument('--hierarchy_label', choices=['default','base'], required=False, help='hierarchy label loss')
    arg_parser.add_argument('--lang', choices=['geo_prolog', 'geo_lambda', 'atis_lambda', "job_prolog", "nlmap", "overnight_lambda", "question_lambda", "top"], default='geo_lambda',
                            help='language to parse. Deprecated, use --transition_system and --parser instead')
    arg_parser.add_argument('--mode', choices=['train', 'pre_train', 'fine_tune', 'test', 'continual_train', 'valid'], default='train', help='Run mode')
    arg_parser.add_argument('--few_shot_mode', choices=['pre_train', 'fine_tune', 'test'], default='pre_train', help='Few shot running mode')
    # few shot learning mode
    #arg_parser.add_argument('--few_shot_mode', choices=['pre_train', 'fine_tune', 'test'], default='pre_train', help='Few shot running mode')
    arg_parser.add_argument('--metric', choices=['prototype', 'relation', 'dot', 'matching'], default='dot', help='Metrics')
    arg_parser.add_argument('--k_shot', default=1, type=int, help='number of eposide shots')
    arg_parser.add_argument('--n_way', default=5, type=int, help='number of eposide types')
    arg_parser.add_argument('--query_num', default=1, type=int, help='number of query instances in one eposide')
    arg_parser.add_argument('--task_num', default=4, type=int, help='number of tasks in one batch')

    #### Modularized configuration ####
    arg_parser.add_argument('--parser', type=str, default='seq2seq', required=False, help='name of parser class to load')
    arg_parser.add_argument('--evaluator', type=str, default='default_evaluator', required=False, help='name of evaluator class to use')

    #### Model configuration ####
    arg_parser.add_argument('--lstm', choices=['lstm','parent'], default='lstm', help='Type of LSTM used, currently only standard LSTM cell is supported')

    arg_parser.add_argument('--rnn_zero_state', choices=['average', 'cls'], default='average',
                            help='RNN zero state')

    arg_parser.add_argument('--attention', choices=['dot','general', 'concat', 'bahdanau','mhd'], default='dot', help='Type of LSTM used, currently only standard LSTM cell is supported')

    arg_parser.add_argument('--sup_attention', action='store_true', default=False, help='Use supervised attention')
    arg_parser.add_argument('--hinge_loss', action='store_true', default=False, help='Use supervised attention')
    arg_parser.add_argument('--embed_fixed', action='store_true', default=False, help='Fix the embed')

    arg_parser.add_argument('--att_reg',  default=1., type=float, help='Use attention regularization')

    arg_parser.add_argument('--reg',  default=1., type=float, help='Use ewc regularization')

    arg_parser.add_argument('--bin_reg',  default=0.001, type=float, help='Use binary regularization')

    arg_parser.add_argument('--att_reg_freq', action='store_true', default=False, help='Use mask during training')

    arg_parser.add_argument('--ewc', type=int, default=0,
                        help='use ewc or not')

    arg_parser.add_argument('--copy', action='store_true', default=False, help='Use copy or not')

    arg_parser.add_argument('--train_mask', action='store_true', default=False, help='Use mask during training')

    arg_parser.add_argument('--mem_net', action='store_true', default=False, help='Use memory network')

    arg_parser.add_argument('--encoder_embed_size', default=768, type=int, help='Size of encoder embedding size')
    arg_parser.add_argument('--decoder_embed_size', default=128, type=int, help='Size of decoder embedding size')

    arg_parser.add_argument('--decoder_layer_size', default=3, type=int, help='Size of decoder layer')
    arg_parser.add_argument('--multi_att_size', default=4, type=int, help='Size of multi-head')
    # Embedding sizes
    arg_parser.add_argument('--embed_size', default=128, type=int, help='Size of word embeddings')
    arg_parser.add_argument('--action_embed_size', default=128, type=int, help='Size of ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--general_action_embed_size', default=32, type=int,help='Size of general ApplyRule/GenToken action embeddings')
    arg_parser.add_argument('--use_general_action_embed', default=False, action='store_true', help='Use general action embed')
    arg_parser.add_argument('--use_children_lstm_encode', default=False, action='store_true',
                            help='Use children lstm encode')
    arg_parser.add_argument('--use_input_lstm_encode', default=False, action='store_true',
                            help='Use input lstm encode')
    arg_parser.add_argument('--use_att', default=True, action='store_true',
                            help='Use att encoding in the next time step')

    arg_parser.add_argument('--init_vertex', default=False, action='store_true',
                            help='init vertex with glove')

    arg_parser.add_argument('--use_coverage', default=False, action='store_true',
                            help='Use coverage attention in the next time step')
    # Hidden sizes
    arg_parser.add_argument('--hidden_size', default=256, type=int, help='Size of LSTM hidden states')

    #### Training ####
    arg_parser.add_argument('--vocab', type=str, help='Path of the serialized vocabulary')
    arg_parser.add_argument('--glove_embed_path', default=None, type=str, help='Path to pretrained Glove mebedding')

    arg_parser.add_argument('--train_file', type=str, help='path to the training target file')
    arg_parser.add_argument('--dev_file', type=str, help='path to the dev source file')
    arg_parser.add_argument('--support_file', type=str, help='path to the support source file')

    arg_parser.add_argument('--english_file', type=str, help='path to the english file')

    arg_parser.add_argument('--supplement_file', type=str, help='path to the supplement file')

    arg_parser.add_argument('--query_file', type=str, help='path to the query source file')

    arg_parser.add_argument('--warmup', default=1, type=int, help='warmup size')

    arg_parser.add_argument('--accumulate_step_size', default=20, type=int, help='accumulate step size')

    arg_parser.add_argument('--batch_size', default=10, type=int, help='Batch size')
    arg_parser.add_argument('--train_iter', default=5, type=int, help='Train iteration size')
    arg_parser.add_argument('--dropout_i', default=0., type=float, help='Input Dropout rate')
    arg_parser.add_argument('--dropout', default=0., type=float, help='Dropout rate')
    arg_parser.add_argument('--word_dropout', default=0., type=float, help='Word dropout rate')
    arg_parser.add_argument('--decoder_word_dropout', default=0.3, type=float, help='Word dropout rate on decoder')
    arg_parser.add_argument('--label_smoothing', default=0.0, type=float,
                            help='Apply label smoothing when predicting labels')

    # training schedule details
    arg_parser.add_argument('--valid_metric', default='acc', choices=['acc'],
                            help='Metric used for validation')
    arg_parser.add_argument('--valid_every_epoch', default=1, type=int, help='Perform validation every x epoch')
    arg_parser.add_argument('--sup_proto_turnover', default=2, type=int, help='supervised proto turn over rate')
    arg_parser.add_argument('--log_every', default=10, type=int, help='Log training statistics every n iterations')
    arg_parser.add_argument('--forward_pass', default=10, type=int, help='forward pass n iterations in maml')
    arg_parser.add_argument('--save_to', default='model', type=str, help='Save trained model to')
    arg_parser.add_argument('--save_all_models', default=False, action='store_true', help='Save all intermediate checkpoints')
    arg_parser.add_argument('--patience', default=5, type=int, help='Training patience')
    arg_parser.add_argument('--max_num_trial', default=10, type=int, help='Stop training after x number of trials')
    arg_parser.add_argument('--uniform_init', default=None, type=float,
                            help='If specified, use uniform initialization for all parameters')
    arg_parser.add_argument('--glorot_init', default=False, action='store_true', help='Use glorot initialization')
    arg_parser.add_argument('--clip_grad', default=5., type=float, help='Clip gradients')
    arg_parser.add_argument('--clip_grad_mode', choices=['value', 'norm'], required=False, help='clip gradients type')
    arg_parser.add_argument('--max_epoch', default=-1, type=int, help='Maximum number of training epoches')
    arg_parser.add_argument('--optimizer', default='Adam', type=str, help='optimizer')
    arg_parser.add_argument('--lr', default=0.001, type=float, help='Learning rate')

    arg_parser.add_argument('--weight_decay', default=0.0, type=float, help='weight decay')

    arg_parser.add_argument('--step_size', default=0.001, type=float, help='Inner Learning rate')
    arg_parser.add_argument('--lr_decay', default=0.5, type=float,
                            help='decay learning rate if the validation performance drops')
    arg_parser.add_argument('--proto_dropout', default=0.5, type=float,
                            help='drop out for the proto probability')
    arg_parser.add_argument('--lr_decay_after_epoch', default=0, type=int, help='Decay learning rate after x epoch')
    arg_parser.add_argument('--lr_decay_every_n_epoch', default=1, type=int, help='Decay learning rate every n epoch')
    arg_parser.add_argument('--decay_lr_every_epoch', action='store_true', default=False, help='force to decay learning rate after each epoch')
    arg_parser.add_argument('--alpha', default=0.95, type=float, help='alpha rate for the rmsprop')
    arg_parser.add_argument('--reset_optimizer', action='store_true', default=False, help='Whether to reset optimizer when loading the best checkpoint')
    arg_parser.add_argument('--verbose', action='store_true', default=False, help='Verbose mode')
    arg_parser.add_argument('--eval_top_pred_only', action='store_true', default=True,
                            help='Only evaluate the top prediction in validation')

    #### decoding/validation/testing ####
    arg_parser.add_argument('--load_model', default=None, type=str, help='Load a pre-trained model')
    arg_parser.add_argument('--beam_size', default=5, type=int, help='Beam size for beam search')
    # arg_parser.add_argument('--place_holder', default=0, type=int, help='Place holder size')

    arg_parser.add_argument('--decode_max_time_step', default=50, type=int, help='Maximum number of time steps used '
                                                                                  'in decoding and sampling')
    arg_parser.add_argument('--relax_factor', default=10, type=int, help='Relax factor for the forward search of reduce in the beam search')
    arg_parser.add_argument('--test_file', type=str, help='Path to the test file')
    arg_parser.add_argument('--save_decode_to', default=None, type=str, help='Save decoding results to file')
    # validate grammar
    arg_parser.add_argument('--validate_grammar', default=False, action='store_true', help='Use grammar check')

    arg_parser.add_argument('--share_parameter', default=False, action='store_true',
                            help='Use shared parameter or not')
    arg_parser.add_argument('--use_adapter', default=False, action='store_true',
                            help='Use adapter or not')

    arg_parser.add_argument('--mask_tgt', default=False, action='store_true',
                            help='Mask target lang or not')

    arg_parser.add_argument('--hyp_embed', default=True, action='store_true',
                            help='Use hybrid shared parameter or not')

    arg_parser.add_argument('--proto_embed', default=False, action='store_true',
                            help='Use proto embedding or not')

    arg_parser.add_argument('--domains', nargs='+')

    arg_parser.add_argument('--src_langs', nargs='+')
    arg_parser.add_argument('--tgt_langs', nargs='+')

    arg_parser.add_argument('--num_exemplars', default=50, type=int, help='Number of exemplars')

    arg_parser.add_argument('--num_exemplars_per_class', default=5, type=int, help='Number of exemplars per class')

    arg_parser.add_argument('--num_exemplars_per_task', default=0, type=int, help='Number of exemplars per task')

    arg_parser.add_argument('--num_exemplars_ratio', default=0.01, type=float, help='Number of exemplars ratio to the number of data instances')

    arg_parser.add_argument('--num_known_domains', default=2, type=int, help='Number of known domains')

    arg_parser.add_argument('--sample', choices=['task_level', 'sample_level','normal_level'], default='task_level', help='Experience replay sample level')

    arg_parser.add_argument('--sample_mode', choices=['template', 'random'], default='template', help='Experience replay sample mode')
    arg_parser.add_argument('--plmm_model_name', choices=['xlm-roberta-base','roberta-base', 'facebook/bart-large'], default='xlm-roberta-base', help='Pre-trained model names')

    arg_parser.add_argument('--shuffle_tasks', default=True, action='store_true', help='Shuffle task or not')

    arg_parser.add_argument('--memory_strength', default=1, type=float, help='memory strength (meaning depends on memory)')

    arg_parser.add_argument('--align_lr', default=0.0025, type=float, help='Align Learning rate')

    arg_parser.add_argument('--max_align_epoch', default=200, type=int, help='Maximum number of training align epoches')

    arg_parser.add_argument('--initial_epoch', default=5, type=int, help='Maximum number of initial training align epoches')

    arg_parser.add_argument('--initial_iter', default=1000, type=int, help='Maximum number of initial training align iters')

    arg_parser.add_argument('--second_iter', default=0, type=int, help='Maximum number of second iter')

    arg_parser.add_argument('--replay_dropout', default=0.3, type=float, help='Dropout rate for replay')

    arg_parser.add_argument('--sample_method', choices=['loss', 'random','f_cluster', 'gss', 'gem', 'max', 'normal', 'sum', 'ne_max', 'ne_sum', 'topk_sum', 'topk_max','init_max', 'init_sum', 'ne_topk_max', 'ne_topk_sum', 'init_min', 'ne_topk_min', 'entropy', 'ne_entropy', 'balance', 'label_uni', 'pro_max', 'pro_sum', 'pro_max_gem', 'pro_max_gem_sub', 'pro_max_gem_reverse','pro_max_gem_reverse_v1', 'pro_max_gem_reverse_v2', 'pro_max_gem_rescale', 'pro_max_gem_turn', 'IQP_uniform','gss_cluster','greedy_uniform', 'graph_clustering', 'IQP_graph',
                                                        'kmedoids_graph', 'ce_slot',
                                                        'lf_tfidf','length'], default='random', help='Sample Method')

    arg_parser.add_argument('--num_memory_buffer', default=60, type=int, help='Memory buffer size')

    arg_parser.add_argument('--gradient_buffer_size', default=150, type=int, help='Gradient buffer size')

    arg_parser.add_argument('--subselect', type=int, default=0,
                        help='first subsample from recent memories')

    arg_parser.add_argument('--mask_action', type=float, default=0., help='use masked action')

    arg_parser.add_argument('--base_model', choices=['seq2seq', 'irnet'],
                            default='seq2seq', help='Base models selection')

    arg_parser.add_argument('--rebalance', type=int, default=0,
                        help='rebalance samples from recent memories')

    arg_parser.add_argument('--augment', type=int, default=0,
                        help='augment the data')

    arg_parser.add_argument('--clustering_filter', type=int, default=0,
                        help='filter clustering action')

    arg_parser.add_argument('--ada_emr',  default=0, type=int, help='Adaptive emr')

    arg_parser.add_argument('--ada_ewc',  default=0, type=float, help='Adaptive ewc')

    arg_parser.add_argument('--discriminate_loss',  default=0, type=float, help='discriminate loss')

    arg_parser.add_argument('--align_att',  default=0, type=int, help='align att')

    arg_parser.add_argument('--att_filter',  default=0, type=int, help='att filter')

    arg_parser.add_argument('--remove_id',  default=-1, type=int, help='att filter')

    arg_parser.add_argument('--action_num',  default=300, type=int, help='number of actions')

    arg_parser.add_argument('--action_dropout',  default=0.5, type=float, help='action selection rate')

    arg_parser.add_argument('--forget_evaluate',  default=0, type=int, help='evaluate forget instances')

    arg_parser.add_argument('--action_forget_evaluate',  default=0, type=int, help='evaluate action forget instances')

    arg_parser.add_argument('--gate_function',  default=0, type=int, help='use gating function or not')
    arg_parser.add_argument('--init_proto',  default=0, type=int, help='init with proto type')
    arg_parser.add_argument('--smax',  default=400, type=float, help='smax value')
    arg_parser.add_argument('--warm',  default=0, type=int, help='warm up type')
    arg_parser.add_argument('--record_error',  default=0, type=int, help='record error')
    arg_parser.add_argument('--dev',  default=0, type=int, help='record error')

    arg_parser.add_argument('--embed_type', choices=['bert','mbert', 'mbert-fix', 'glove','bert-mini', 'bert-mini-fix', 'bert-fix'], default='glove', help='word embedding type')

    return arg_parser


def update_args(args, arg_parser):
    for action in arg_parser._actions:
        if isinstance(action, argparse._StoreAction) or isinstance(action, argparse._StoreTrueAction) \
                or isinstance(action, argparse._StoreFalseAction):
            if not hasattr(args, action.dest):
                setattr(args, action.dest, action.default)
