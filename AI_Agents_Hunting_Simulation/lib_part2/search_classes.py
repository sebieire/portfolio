#import settings

from collections import deque
from . import settings # import module



# PROBLEM
class FindingHutProblem():
    """ Problem of searching the world (graph) from one node to the next. """    
    
    def __init__(self, initial_state, goal_state=None, graph=None): # !
        self.initial = initial_state
        self.goal = goal_state
        self.graph = graph
        
    def actions(self, the_state):
        """ all actions of that node (or that state) is its neighbors 
        the_state = current state of Node (call from Node.expand method) """
        
        if settings.searchProblemDebug:
            print("State:", the_state, " - Returned Dicts:", self.graph.get_inner_dict_from_state(the_state))
        
        # all keys --> list
        return list(self.graph.get_inner_dict_from_state(the_state).keys())
    
    def goal_test(self, the_goal):
        """ return True if the state is a goal
        compares state to self.goal or checks for state in self.goal if it is a list
        override if checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            #return is_in(state, self.goal) # original - requires aima utils import
            return any(x is the_goal for x in self.goal) # adapted -> direct implementation        
        else:
            return the_goal == self.goal
    
    def get_path_cost(self, cost_so_far, the_state_A, action, the_state_B):
        """ path cost so far """               
        return cost_so_far + (self.graph.get_distance_between_two_states(the_state_A,the_state_B) or settings.infinity)
    

# NODE
class Node:
    """ Node """

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""        
        listOfChildren = [self.child_node(problem, action) for action in problem.actions(self.state)]
        return listOfChildren


    def child_node(self, problem, action):
        """ New Child Node ( from Node.expand() ) """
        
        #next_state = problem.result(self.state, action) # not currently required
        next_state = action
        
        # new NODE
        next_node = Node(next_state, self, action, 
                         problem.get_path_cost(self.path_cost, self.state, action, next_state))
        
        # global expanded count        
        settings.nodes_expanded_count += 1
        
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        return hash(self.state)




# GRAPH
class CustomDirectedGraph():
    """
    Custom Implementation Of A Directed (two way) Graph
    Adaptation from original aima - search.py
    """
    
    def __init__(self, graph_dict=None):
        self.graph_dict = graph_dict or {}
        
        for a_key in list(self.graph_dict.keys()):
            for (b_key, value) in self.graph_dict[a_key].items():
                # b_key = inner dict keys
                # value = inner dict values (of that related key)
                self.connect_one(b_key, a_key, value) 
    
    def connect_one(self, A, B, distance):
        """Add a link from A to B of given distance, in one direction only."""
        self.graph_dict.setdefault(A, {})[B] = distance 

    def get_inner_dict_from_state(self, state):
        """
        Custom
        Returns dictionary of given state (=key)
        Basically returns the complete inner dictionary matching given state (coordinate or city or...)
        """
        #state = dict key / or else default is empty dict {}
        return self.graph_dict.setdefault(state, {})
    
    def get_distance_between_two_states(self, stateA, stateB):
        """
        Custom
        Returns value of matching stateB key from matching stateA key
        Basically returns value for B in inner dictionary
        """
        innerDictionary = self.get_inner_dict_from_state(stateA) # get inner dictionary where key = stateA
        
        # ++++++++++++++ position[0] holds actual distance
        return innerDictionary.get(stateB)[0] # return the first value where key = stateB    
    












