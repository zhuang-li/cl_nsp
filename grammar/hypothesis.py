# coding=utf-8
from grammar.rule import ReduceAction
from grammar.vertex import CompositeTreeVertex, RuleVertex, TreeVertex
from grammar.rule import GenNTAction, GenTAction
from grammar.rule import ProductionRuleBLB
from model.attention_util import dot_prod_attention
import torch
from grammar.consts import IMPLICIT_HEAD, ROOT, FULL_STACK_LENGTH, NT, TYPE_SIGN, NT_TYPE_SIGN, STRING_FIELD, END_TOKEN


class Hypothesis(object):
    def __init__(self):
        self.tree = None
        # action
        self.actions = []
        self.action_id = []
        self.general_action_id = []
        # tgt code
        self.tgt_code_tokens = []
        self.tgt_code_tokens_id = []

        self.var_id = []
        self.ent_id = []

        self.heads_stack = []
        self.tgt_ids_stack = []
        self.heads_embedding_stack = []
        self.embedding_stack = []
        self.hidden_embedding_stack = []
        self.v_hidden_embedding = None

        self.current_gen_emb = None
        self.current_re_emb = None
        self.current_att = None
        self.current_att_cov = None

        self.score = 0.
        self.is_correct = False
        self.is_parsable = True
        # record the current time step
        self.reduce_action_count = 0
        self.t = 0

        self.frontier_node = None
        self.frontier_field = None


    @property
    def to_prolog_template(self):
        if self.tree:
            return self.tree.to_prolog_expr
        else:
            return ""

    @property
    def to_lambda_template(self):
        if self.tree:
            return self.tree.to_lambda_expr
        else:
            return ""

    @property
    def to_logic_form(self):
        return " ".join([str(code) for code in self.tgt_code_tokens])

    def is_reduceable(self, rule, vertex_stack):
        # assert len(vertex_stack) == len(embedding_stack), "length of vertex stack should be equal to the length of embedding stack"
        head = rule.head.copy_no_link()
        if head.head.startswith(ROOT) and rule.body_length < len(vertex_stack):
            return False
        temp_stack = [node.copy_no_link() for node in vertex_stack[-rule.body_length:]]
        if head.head.startswith(IMPLICIT_HEAD) or head.head.startswith(ROOT):
            head.is_auto_nt = True

        if isinstance(rule, ProductionRuleBLB):
            if head.head.startswith(ROOT):
                for i in range(len(temp_stack)):
                    head.children.append(vertex_stack.pop().copy())
            else:
                for child in rule.body[::-1]:
                    if not (str(child) == NT):
                        head.children.append(child.copy())
                    else:
                        head.children.append(vertex_stack.pop().copy())

        else:
            for i in range(len(temp_stack)):
                head.children.append(vertex_stack.pop().copy())

        head.children.reverse()
        vertex_stack.append(head)
        return True


    def reduce_actions(self, reduce_action):
        stack = self.heads_stack
        if self.is_reduceable(reduce_action.rule, stack):
            self.is_parsable = True
            if len(stack) == 1 and stack[0].is_answer_root():
                self.tree = stack[0]
        else:
            self.is_parsable = False
        return stack

    def reduce_embedding(self, reduce_embedding, length_of_rule, mem_net):
        assert length_of_rule <= len(self.embedding_stack), "embedding stack length must be longer than or equal to the rule body length"
        if mem_net:
            partial_stack_embedding = torch.stack(self.embedding_stack[-length_of_rule:])
            reduce_embedding = reduce_embedding.unsqueeze(0)
            partial_stack_embedding = partial_stack_embedding.unsqueeze(0)
            cxt_vec, cxt_weight = dot_prod_attention(reduce_embedding, partial_stack_embedding, partial_stack_embedding)
            reduce_embedding = reduce_embedding.squeeze() + cxt_vec.squeeze()
        for i in range(length_of_rule):
            self.embedding_stack.pop()
        self.embedding_stack.append(reduce_embedding)
        return self.embedding_stack

    def reduce_action_ids(self, reduce_id, length_of_rule):
        if length_of_rule == FULL_STACK_LENGTH:
            length_of_rule = len(self.tgt_ids_stack)
        assert length_of_rule <= len(self.tgt_ids_stack), "action id length must be longer than or equal to the rule body length"
        for i in range(length_of_rule):
            self.tgt_ids_stack.pop()
        self.tgt_ids_stack.append(reduce_id)
        return self.tgt_ids_stack
    """
    def clone_and_apply_action(self, actions):
        new_hyp = self.copy()
        new_hyp.apply_action(actions)

        return new_hyp
    """

    def copy(self):
        new_hyp = Hypothesis()
        if self.tree:
            new_hyp.tree = self.tree.copy()
        new_hyp.action_id = list(self.action_id)
        new_hyp.actions = list(self.actions)
        new_hyp.general_action_id = list(self.general_action_id)
        new_hyp.var_id = list(self.var_id)
        new_hyp.ent_id = list(self.ent_id)
        new_hyp.heads_stack = list(self.heads_stack)
        new_hyp.tgt_ids_stack = list(self.tgt_ids_stack)
        new_hyp.embedding_stack = [embedding.clone() for embedding in self.embedding_stack]
        new_hyp.heads_embedding_stack = [embedding.clone() for embedding in self.heads_embedding_stack]
        new_hyp.hidden_embedding_stack = [(state.clone(), cell.clone()) for state, cell in self.hidden_embedding_stack]
        new_hyp.tgt_code_tokens_id = list(self.tgt_code_tokens_id)
        new_hyp.tgt_code_tokens = list(self.tgt_code_tokens)

        new_hyp.score = self.score
        new_hyp.t = self.t
        new_hyp.is_correct = self.is_correct
        new_hyp.is_parsable = self.is_parsable
        new_hyp.reduce_action_count = self.reduce_action_count

        if self.current_gen_emb is not None:
            new_hyp.current_gen_emb = self.current_gen_emb.clone()

        if self.current_re_emb is not None:
            new_hyp.current_re_emb = self.current_re_emb.clone()

        if self.current_att is not None:
            new_hyp.current_att = self.current_att.clone()

        if self.current_att_cov is not None:
            new_hyp.current_att_cov = self.current_att_cov.clone()

        if self.v_hidden_embedding is not None:
            new_hyp.v_hidden_embedding = (self.v_hidden_embedding[0].clone(), self.v_hidden_embedding[1].clone())

        new_hyp.update_frontier_info()
        return new_hyp

    def completed(self):
        return self.tree

    def rule_completed(self):
        return self.tree and self.frontier_field == None

    def update_frontier_info(self):
        def _find_frontier_node_and_field(tree_node):
            if tree_node:
                for field in tree_node.children:
                    # if it's an intermediate node, check its children
                    if field.created_time == -1:
                        field.created_time = tree_node.created_time
                    result = _find_frontier_node_and_field(field)
                    if result: return result
                    #print (field.finished)
                    # now all its possible children are checked
                    if not field.finished:
                        return tree_node, field

                return None
            else: return None

        frontier_info = _find_frontier_node_and_field(self.tree)
        if frontier_info:
            self.frontier_node, self.frontier_field = frontier_info
        else:
            self.frontier_node, self.frontier_field = None, None

    def init_created_time(self, tree_node, t):
        #print (tree_node)
        tree_node.created_time = t
        for child in tree_node.children:
            self.init_created_time(child, t)

    def apply_action(self, action):
        if self.tree is None:
            assert isinstance(action, GenNTAction), 'Invalid action [%s], only ApplyRule action is valid ' \
                                                        'at the beginning of decoding'

            self.tree = action.vertex.copy()
            #print ("========tree=========")
            #print (self.tree)
            self.update_frontier_info()
            if self.frontier_node:
                self.init_created_time(self.frontier_node, self.t)
        elif self.frontier_node:

            field_node = action.vertex

            self.init_created_time(field_node, self.t)
            #print (self.t)
            if self.frontier_field.head.endswith(STRING_FIELD):
                if str(field_node.copy()) == END_TOKEN:
                    field_head = self.frontier_node.children[self.frontier_field.position].head
                    self.frontier_node.children[self.frontier_field.position].head = " ".join(field_head.split(' ')[1:])
                    self.frontier_node.children[self.frontier_field.position].finished = True
                else:
                    self.frontier_node.children[self.frontier_field.position].add_head_token(field_node.copy())
            else:
                self.frontier_node.children[self.frontier_field.position] = field_node.copy()
            #print (self.frontier_node)
            if isinstance(action, GenNTAction):
                assert self.frontier_field.head == NT or self.frontier_field.head.startswith(NT_TYPE_SIGN)
            elif isinstance(action, GenTAction):
                assert self.frontier_field.head.startswith(TYPE_SIGN)
            else:
                raise ValueError
            #print("before",self.frontier_node)
            #print("before",self.frontier_field)
            #print("before",self.tree)
            self.update_frontier_info()
            #print("after",self.frontier_node)
            #print("after",self.frontier_field)
        else:  # fill in a primitive field
            raise ValueError

        self.t += 1
        self.actions.append(action)
        #print (self.actions)


    def clone_and_apply_action(self, action):
        #print ("=============")
        new_hyp = self.copy()
        #print(new_hyp.tree)
        #print(new_hyp.frontier_node)
        #print(new_hyp.frontier_field)
        #if new_hyp.frontier_field:
            #print(new_hyp.frontier_field.position)
        #print (action)
        new_hyp.apply_action(action)
        #print ("=============")
        #print (new_hyp.frontier_field.head)
        return new_hyp