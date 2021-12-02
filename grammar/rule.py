from grammar.vertex import *
from grammar.action import *
from grammar.consts import *
from common.registerable import Registrable

class Rule(object):
    def __init__(self, head):
        assert isinstance(head, TreeVertex), '{} is of type {}'.format(head, type(head))
        # The rule head is a RuleVertex object
        self.head = head
        self.body_length = 0

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{}".format(self.head)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()


@Registrable.register('ProductionRule')
class ProductionRule(Rule):
    def __init__(self, head, body_length):
        assert isinstance(head, TreeVertex), '{} is of type {}'.format(head, type(head))
        # The rule head is a RuleVertex object
        self.head = head
        # The rule body is a list of RuleVertex objects
        self.body_length = body_length
        self.body_length_set = set()


@Registrable.register('ProductionRuleB')
class ProductionRuleB(Rule):
    def __init__(self, head, body):
        assert isinstance(head, TreeVertex), '{} is of type {}'.format(head, type(head))
        # The rule head is a RuleVertex object
        self.head = head
        # The rule body is a list of RuleVertex objects
        self.body = body

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{} :- [{}]".format(self.head, self.body)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

@Registrable.register('ProductionRuleBL')
class ProductionRuleBL(Rule):
    def __init__(self, head, body_length):
        assert isinstance(head, TreeVertex), '{} is of type {}'.format(head, type(head))
        # The rule head is a RuleVertex object
        self.head = head
        # The rule body is a list of RuleVertex objects
        self.body_length = body_length

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{} :- [{}]".format(self.head, self.body_length)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

@Registrable.register('ProductionRuleBLB')
class ProductionRuleBLB(Rule):
    def __init__(self, head, body_length, body):
        assert isinstance(head, TreeVertex), '{} is of type {}'.format(head, type(head))
        # The rule head is a RuleVertex object
        self.head = head
        # The rule body is a list of RuleVertex objects
        self.body_length = body_length
        self.body = body

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return "{} :- {} {}".format(self.head, self.body_length, self.body)

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

def recursive_reduce(dfs_tree_root, config_seq, production_rule, body = None):
    if body:
        new_body = list(body)
        assert len(body) == len(dfs_tree_root.children), "body length must be equal to the children length"
    for id, child in enumerate(dfs_tree_root.children):
        if production_rule == ProductionRuleBLB:
            if not new_body[id] == NT:
                continue
        if not child.has_children():
            action = GenAction(child)
            config_seq.append(action)
            action.entities = extract_action_lit([[action]], 'entity')[0]
            action.variables = extract_action_lit([[action]], 'variable')[0]
            #if isinstance(child, CompositeTreeVertex):
                #vertex = child.vertex
            #elif isinstance(child, RuleVertex):
                #vertex = child
            #if vertex.head == 'salary_greater_than' or vertex.head == 'salary_less_than':
                #turn_var_back(vertex, turn_v_back=False)
                #action.entities = []
        else:
            head = RuleVertex(child.head)
            if str(head).startswith(IMPLICIT_HEAD):
                head.is_auto_nt = True
            #print (production_rule)
            if production_rule == ProductionRuleBLB:
                body = []
                body_len = 0
                for c in child.children:
                    if len(c.children) == 0 and (not isinstance(c, CompositeTreeVertex)):
                        body.append(c)
                        #config_seq = config_seq[:len(config_seq) - 1] + config_seq[len(config_seq):]
                    else:
                        body.append(NT)
                        body_len += 1
                rule = production_rule(head, body_len, body)
                recursive_reduce(child, config_seq, production_rule, body)
            else:
                rule = production_rule(head, len(child.children))
                recursive_reduce(child, config_seq, production_rule)

            reduce = ReduceAction(rule)
            if production_rule == ProductionRuleBLB:
                reduce.entities = extract_action_lit([[reduce]], 'entity')[0]
                reduce.variables = extract_action_lit([[reduce]], 'variable')[0]
            config_seq.append(reduce)


def turn_var_back(dfs_tree_root, turn_v_back = True, turn_e_back = True):
    list_nodes = [dfs_tree_root]
    while len(list_nodes) > 0:
        node = list_nodes.pop()
        if isinstance(node, RuleVertex):
            if node.original_var and turn_v_back:
                node.head = node.original_var
            if node.original_entity and turn_e_back:
                node.head = node.original_entity
        elif isinstance(node, CompositeTreeVertex):
            turn_var_back(node.vertex)
        for child in node.children:
            list_nodes.append(child)

def product_rules_to_actions_bottomup(template_trees, leaves_list, template_db, use_normalized_trees=True,
                             rule_type="ProductionRule",turn_v_back=False):
    tid2config_seq = []
    if use_normalized_trees:
        for tid, leaves in enumerate(leaves_list):
            convert_tree_to_composite_vertex(leaves)

    production_rule = Registrable.by_name(rule_type)
    composite_ast = []
    for tid, dfs_tree_root in enumerate(template_trees):
        config_seq = []
        if turn_v_back:
            turn_var_back(dfs_tree_root)
        head = RuleVertex(dfs_tree_root.head)
        head.is_auto_nt = True
        if production_rule == ProductionRuleBLB:
            body = []
            for c in dfs_tree_root.children:
                if len(c.children) == 0 and (not isinstance(c, CompositeTreeVertex)):
                    body.append(c)
                    c.parent = None
                    # config_seq = config_seq[:len(config_seq) - 1] + config_seq[len(config_seq):]
                else:
                    body.append(NT)
            rule = production_rule(head, FULL_STACK_LENGTH, [])
            recursive_reduce(dfs_tree_root, config_seq, production_rule, body)
        else:
            rule = production_rule(head, FULL_STACK_LENGTH)
            recursive_reduce(dfs_tree_root, config_seq, production_rule)

        reduce = ReduceAction(rule)
        config_seq.append(reduce)
        for c in config_seq:
            template_db.index_action(tid, c)
        tid2config_seq.append(config_seq)
        composite_ast.append(dfs_tree_root)
    return tid2config_seq

"""
def dfs_recursive(node,config_seq):

    node.created_time = len(config_seq)
    head = RuleVertex(node.head)
    head.is_grammar_vertex = node.is_grammar_vertex
    head.created_time = node.created_time
    if node.parent:
        head.parent = config_seq[node.parent.created_time].vertex
    if str(head).startswith(IMPLICIT_HEAD) or str(head) == ROOT:
        head.is_auto_nt = True
    if head.is_grammar_vertex:
        action = GenTAction(head)
    else:
        action = GenNTAction(head)
    if head.parent:
        action.parent_t = head.parent.created_time
    config_seq.append(action)
    body = []
    idx = 0

    for c in node.children:
        if c.is_terminal():
            if isinstance(c, CompositeTreeVertex):

                c_body = RuleVertex(NT)
                c_body.finished = False
                c_body.position = idx
                body.append(c_body)

                com_vertex = c.vertex

                com_vertex.parent = node

                com_vertex.created_time = len(config_seq)

                com_vertex.parent = config_seq[node.created_time].vertex
                if str(com_vertex.head).startswith(IMPLICIT_HEAD) or str(com_vertex.head) == ROOT:
                    com_vertex.is_auto_nt = True

                action = GenNTAction(com_vertex)
                if com_vertex.parent:
                    action.parent_t = com_vertex.parent.created_time
                config_seq.append(action)

                var_list = []
                get_vertex_variables(com_vertex, 'variable', var_list)
                for vertex in var_list:
                    vertex.parent = com_vertex
                    dfs_recursive(vertex, config_seq)


            elif str(c).startswith(TYPE_SIGN):
                c.finished = False
                var_list = []
                get_vertex_variables(c, 'variable', var_list)
                for vertex in var_list:
                    vertex.parent = head
                    dfs_recursive(vertex, config_seq)

                body.append(c)
                c.position = idx
            else:
                body.append(c)
                c.position = idx
        else:
            c_body = RuleVertex(NT)
            c_body.finished = False
            body.append(c_body)
            dfs_recursive(c, config_seq)
            c_body.position = idx
        idx += 1
    head.children = body
    return head

"""

def nlmap_dfs_recursive(node, config_seq, is_variable=False):
    if is_variable:
        node.created_time = len(config_seq)
        head = node.copy()
        head.is_grammar_vertex = node.is_grammar_vertex
        head.created_time = node.created_time
        if node.parent:
            head.parent = config_seq[node.parent.created_time].vertex
        if str(head.head).startswith(IMPLICIT_HEAD) or str(head.head) == ROOT:
            head.is_auto_nt = True
        if head.is_grammar_vertex:
            gen_t_action = GenTAction(head)
            if head.parent:
                gen_t_action.parent_t = head.parent.created_time
            config_seq.append(gen_t_action)
        else:
            gen_nt_action = GenNTAction(head)
            if head.parent:
                gen_nt_action.parent_t = head.parent.created_time
            config_seq.append(gen_nt_action)
    else:
        node.created_time = len(config_seq)
        head = RuleVertex(node.head)
        head.is_grammar_vertex = node.is_grammar_vertex
        head.created_time = node.created_time
        if node.parent:
            head.parent = config_seq[node.parent.created_time].vertex
        if str(head).startswith(IMPLICIT_HEAD) or str(head) == ROOT:
            head.is_auto_nt = True
        if head.is_grammar_vertex:
            gen_t_action = GenTAction(head)
            if head.parent:
                gen_t_action.parent_t = head.parent.created_time
            config_seq.append(gen_t_action)
        else:
            gen_nt_action = GenNTAction(head)
            if head.parent:
                gen_nt_action.parent_t = head.parent.created_time
            config_seq.append(gen_nt_action)
        body = []
        idx = 0
        for c in node.children:
            if c.is_terminal():
                if isinstance(c, CompositeTreeVertex):
                    c = c.vertex
                    c.created_time = head.created_time
                    var_list = []
                    get_vertex_variables(c, 'variable', var_list)
                    for vertex in var_list:
                        vertex.parent = head
                        nlmap_dfs_recursive(vertex, config_seq, is_variable=True)
                elif str(c).startswith(TYPE_SIGN) or str(c).startswith(NT_TYPE_SIGN):
                    c.created_time = head.created_time
                    c.finished = False
                    var_list = []
                    get_vertex_variables(c, 'variable', var_list)
                    for vertex in var_list:
                        vertex.parent = head
                        nlmap_dfs_recursive(vertex, config_seq, is_variable=True)
                body.append(c)
                c.position = idx
            else:
                c_body = RuleVertex(NT)
                c_body.finished = False
                body.append(c_body)
                nlmap_dfs_recursive(c, config_seq, is_variable=False)
                c_body.position = idx
            idx += 1
        head.children = body

def dfs_recursive(node, config_seq):
    node.created_time = len(config_seq)
    head = RuleVertex(node.head)
    head.is_grammar_vertex = node.is_grammar_vertex
    head.created_time = node.created_time
    if node.parent:
        head.parent = config_seq[node.parent.created_time].vertex
    if str(head).startswith(IMPLICIT_HEAD) or str(head) == ROOT:
        head.is_auto_nt = True
    if head.is_grammar_vertex:
        gen_t_action = GenTAction(head)
        if head.parent:
            gen_t_action.parent_t = head.parent.created_time
        config_seq.append(gen_t_action)
    else:
        gen_nt_action = GenNTAction(head)
        if head.parent:
            gen_nt_action.parent_t = head.parent.created_time
        config_seq.append(gen_nt_action)
    body = []
    idx = 0
    for c in node.children:
        if c.is_terminal():
            if isinstance(c, CompositeTreeVertex):
                c = c.vertex
                c.created_time = head.created_time
                var_list = []
                get_vertex_variables(c, 'variable', var_list)
                for vertex in var_list:
                    vertex.parent = head
                    dfs_recursive(vertex, config_seq)
            elif str(c).startswith(TYPE_SIGN) or str(c).startswith(NT_TYPE_SIGN):
                c.created_time = head.created_time
                c.finished = False
                var_list = []
                get_vertex_variables(c, 'variable', var_list)
                for vertex in var_list:
                    vertex.parent = head
                    dfs_recursive(vertex, config_seq)
            body.append(c)
            c.position = idx
        else:
            c_body = RuleVertex(NT)
            c_body.finished = False
            body.append(c_body)
            dfs_recursive(c, config_seq)
            c_body.position = idx
        idx += 1
    head.children = body


def init_position(node, idx):
    node.position = idx
    for c_idx, child in enumerate(node.children):
        init_position(child, c_idx)

def product_rules_to_actions_topdown(template_trees, leaves_list, template_db, use_normalized_trees=True,turn_v_back=False, lang='overnight'):
    tid2config_seq = []
    if use_normalized_trees:
        for tid, leaves in enumerate(leaves_list):
            convert_tree_to_composite_vertex(leaves)

    for tid, dfs_tree_root in enumerate(template_trees):
        config_seq = []
        if turn_v_back:
            turn_var_back(dfs_tree_root)

        #dfs_tree_root.children[0].parent = None
        if lang == 'nlmap_qtype':
            nlmap_dfs_recursive(dfs_tree_root, config_seq, is_variable=False)
        else:
            dfs_recursive(dfs_tree_root, config_seq)

        for c in config_seq:
            if isinstance(c, GenTAction):
                c.type = c.get_vertex_type()
            template_db.index_action(tid, c)
        tid2config_seq.append(config_seq)

    return tid2config_seq

def get_reduce_action_length_set(tid2config_seq):
    reduce_action_length_set = {}
    for action_seq in tid2config_seq:
        for action in action_seq:
            if isinstance(action, ReduceAction):
                rule_len = action.rule.body_length
                if action not in reduce_action_length_set:
                    reduce_action_length_set[action] = set()
                if rule_len not in reduce_action_length_set[action]:
                    reduce_action_length_set[action].add(rule_len)
    return reduce_action_length_set

def get_vertex_variables(vertex, type, seq_list):
    if vertex.has_children():
        for child in vertex.children:
            get_vertex_variables(child, type, seq_list)
    else:
        if type == 'variable':
            if vertex.original_var:
                vertex.finished = False
                seq_list.append(vertex.original_var)
        elif type == 'entity':
            if vertex.original_entity:
                vertex.finished = False
                seq_list.append(vertex.original_entity)

def extract_action_lit(action_seqs, type = 'variable'):
    seq = []
    for action_seq in action_seqs:
        seq.append([])
        for action in action_seq:
            if isinstance(action, GenAction):
                if isinstance(action.vertex, RuleVertex):
                    vertex = action.vertex
                else:
                    vertex =  action.vertex.vertex
                get_vertex_variables(vertex, type, seq[-1])
            elif isinstance(action, ReduceAction):
                for vertex in action.rule.body:
                    if isinstance(vertex, RuleVertex):
                        if type == 'variable':
                            if vertex.original_var:
                                seq[-1].append(vertex.original_var)
                        elif type == 'entity':
                            if vertex.original_entity:
                                seq[-1].append(vertex.original_entity)
                    elif isinstance(vertex, str):
                        continue
                    else:
                        raise ValueError
            else:
                raise ValueError
    return seq