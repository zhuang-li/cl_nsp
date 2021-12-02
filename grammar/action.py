from grammar.utils import rm_subbag

class Action(object):
    def __init__(self):
        self.neo_type = False
        self.entities = []
        self.variables = []
        self.prototype_tokens = []
        self.align_ids = []
        self.align_tokens = []
        self.string_sim_score = []
        self.cond_score = []
        self.entity_align = []

class ReduceAction(Action):

    def __init__(self, rule):
        Action.__init__(self)
        self.rule = rule

    def __repr__(self):
        return 'REDUCE {}'.format(self.rule).replace(' ', '_')

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()


class GenAction(Action):

    def __init__(self, vertex):
        Action.__init__(self)
        self.vertex = vertex
        self.parent_t = -1
        self.type = ""

    def __repr__(self):
        return 'GEN {}'.format(self.vertex.rep()).replace(' ', '_')

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

    def get_vertex_type(self):
        vertex_str = self.vertex.to_lambda_expr
        if vertex_str.startswith('('):
            vertex_str_list = vertex_str.split(' ')
            if len(vertex_str_list) > 4:
                return vertex_str_list[1] + "." + vertex_str_list[3]
            else:
                return vertex_str_list[1]
        else:
            vertex_str_list = vertex_str.split('.')
            return '.'.join(vertex_str_list[:-1])

class GenNTAction(GenAction):

    def __init__(self, vertex):
        GenAction.__init__(self, vertex)

    def __repr__(self):
        return 'GEN {}'.format(self.vertex.to_lambda_expr)

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

    def copy(self):
        # to do: add the other copy properties
        new_action = GenNTAction(self.vertex.copy())
        new_action.parent_t = self.parent_t

        return new_action

class GenTAction(GenAction):

    def __init__(self, vertex):
        GenAction.__init__(self, vertex)


    def __repr__(self):
        return 'GEN {}'.format(self.vertex.to_lambda_expr)

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()

    def copy(self):
        # to do: add the other copy properties
        new_action = GenTAction(self.vertex.copy())
        new_action.parent_t = self.parent_t
        new_action.type = self.type
        return new_action


class TerminalAction(Action):
    def __init__(self):
        Action.__init__(self)

    def __repr__(self):
        return 'Terminal'

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()


class NTAction(Action):

    def __init__(self, vertex):
        Action.__init__(self)
        self.vertex = vertex

    def __repr__(self):
        return 'NT [{}]'.format(self.vertex.rep())

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()


class TokenAction(Action):

    def __init__(self, vertex):
        Action.__init__(self)
        self.vertex = vertex

    def __repr__(self):
        return 'GENToken [{}]'.format(self.vertex.rep())

    def __str__(self):
        return self.__repr__()

    def __hash__(self):
        return hash(self.__repr__())

    def __eq__(self, other):
        return self.__class__ == other.__class__ and self.__repr__() == other.__repr__()


class ParserConfig(object):

    def __init__(self, action = None):
        # A queue stores the current level of vertices to process after applying the action.
        # Each vertex is not a reference to any vertex in the full parse tree.
        self.queue = []
        # Parse action
        self.action = action

    def transit(self, action):
        new_config = ParserConfig(action)
        new_config.queue.extend(self.queue)
        if isinstance(action, ReduceAction):
            rm_subbag(action.rule.body, new_config.queue)
            new_config.queue.append(action.rule.head.rep())
        else:
            new_config.queue.append(action.vertex.rep())
        return new_config


    def __repr__(self):
        return '({} {})'.format(self.action, self.queue)

    def __str__(self):
        return self.__repr__()
