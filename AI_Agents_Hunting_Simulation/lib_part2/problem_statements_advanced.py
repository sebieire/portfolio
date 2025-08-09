#from . import settings # import module

import settings
from collections import deque
import pprint
import random


localDebug = False
problemDebug = False



# PROBLEM
class CritterHuntPerformanceProblem():
    """ Problem of searching the world (graph) from one node to the next. """    
    
    def __init__(self, initial_state, goal_state=None, graph=None): ## !
        self.initial = initial_state
        self.goal = goal_state
        self.graph = graph
        
    def actions(self, the_state):
        """ all actions of that node (or that state) is its neighbors 
        the_state = current state of Node (call from Node.expand method) """
        
        if problemDebug:
            print("State:", the_state, " - Returned Dicts:", self.graph.get_inner_dict_from_state(the_state))
        
        # all keys --> list
        return list(self.graph.get_inner_dict_from_state(the_state).keys())
    
    # currently not required
    """
    def result(self, the_state, action): 
        #going to neighbor is the result = neighbor 
        #called from Node.child_node (creating a new Node)
        return action # simple implementation
    """
    
    def goal_test(self, node, the_goal):
        """ return True if the state is a goal
        compares state to self.goal or checks for state in self.goal if it is a list
        override if checking against a single self.goal is not enough."""
        if isinstance(self.goal, list):
            #return is_in(state, self.goal) # original - requires aima utils import
            return any(x is the_goal for x in self.goal) # adapted -> direct implementation
        else:            
            #return the_goal == self.goal            
            print("Node Test:", node, "Current Performance Goal -------------------->", the_goal)
            return int(the_goal) >= self.goal
    
    def get_performance_score(self, performance_score_so_far, the_state_A, the_state_B ):
        """ performance score so far """        
        return performance_score_so_far + (self.graph.get_performance_score_between_two_states(the_state_A,the_state_B))    
    
    def get_path_cost(self, cost_so_far, the_state_A, the_state_B):
        """ path cost so far """               
        return cost_so_far + (self.graph.get_path_cost_between_two_states(the_state_A,the_state_B) or settings.infinity)
    

# NODE
class Node:
    """ Node """

    def __init__(self, state, parent=None, action=None, path_cost=0, performance_score=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.performance_score = performance_score
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
                         problem.get_path_cost(self.path_cost, self.state, next_state),
                         problem.get_performance_score(self.performance_score, self.state, next_state))
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

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

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
    
    def get_path_cost_between_two_states(self, stateA, stateB):
        """
        Custom Implementation
        Returns value of matching stateB key from matching stateA key
        Basically returns value for B in inner dictionary
        """
        innerDictionary = self.get_inner_dict_from_state(stateA) # get inner dictionary where key = stateA
        
        # ++++++++++++++ position[0] holds actual distance
        return innerDictionary.get(stateB)[0] # return the first value where key = stateB    
    
    def get_value_list_from_states(self, parent_state, child_state):
        """
        Custom Implementation (somehow a bit similar to get_path_cost_between_two_states)
        Returns entire value list of matching keys where parent key cascades to child key and then value list
        Example: {(0, 0): {(0, 1): [1, 2, 3, 4],
                          (0, 4): [1, 14124],
                          (1, 0): [1, 'hi'],
                            (4, 0): [1, 13]} }
        parent (0,0) and child (1,0) would return the list [1, 'hi']
        """
        return self.graph_dict.get(parent_state).get(child_state)
    
    def get_performance_score_between_two_states(self, stateA, stateB):
        """
        Custom Implementation (similar to get_path_cost_between_two_states)
        Returns performance score of matching stateB key from matching stateA key
        Basically returns value for B in inner dictionary
        """
        innerDictionary = self.get_inner_dict_from_state(stateA) # get inner dictionary where key = stateA
        
        # ++++++++++++++ position[1] holds actual performance score
        return innerDictionary.get(stateB)[1] # return the second value where key = stateB  


# BREADTH FIRST SEARCH
def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    # INITIAL PARENT NODE
    node = Node(problem.initial)
    if localDebug:
        print("Initial Node (state):", node.state)
    
    # if initial node is goal - return    
    # Initial GOAL TEST (can never test right - taken out in this scenario)    
    #if problem.goal_test(node.state): 
    #    return node
    
    # else add to frontier (deque)
    frontier = deque([node])
    explored = set()
    
    # in frontier
    while frontier:
        # take node out from left of deque (FIFO)
        node = frontier.popleft()
        if localDebug:
            print("--------> Investigating - Node taken from frontier:", node)
            print("Its action:", node.action, ", state:", node.state, "parent:", node.parent, "path_cost:", node.path_cost, "depth:", node.depth)        
       
        
        # and add to explored set (no duplicates in set!)
        explored.add(node.state)
        if localDebug:
            print("Explored Set so far:", explored)        
            print("\n----------------------")
            print("Going through each child in current Node (expand node):")
        # each in frontier -> node.expand
        for child in node.expand(problem):
            
            if localDebug:
                print("Current cycled child is:", child)
            
            # if not already explored and not in frontier (either solution or add to frontier)
            if child.state not in explored and child not in frontier:
                
                if localDebug: # DEBUG
                    print("CHILD:", child.state, "PARENT:", child.parent.state)
                    print("Value List (Parent -> Child -> Values):", problem.graph.get_value_list_from_states(child.parent.state,child.state)[1])
                    
                #if problem.goal_test(child.state): # GOAL TEST - - - > pass in list
                #if problem.goal_test(problem.graph.get_value_list_from_states(child.parent.state,child.state)[1]): # GOAL TEST PROPER- - - > pass in list item
                if problem.goal_test(child, child.performance_score):
                    return child
                frontier.append(child)
                
                if localDebug:
                    print("Node appended to frontier:", child)
    return None
    







# populate world size dict
# position[0] is the step_cost
# position[1] is scoreOnWorldGridTile (assignment through world map)

scoreOnWorldGridTile = [10,3,1] # mapped later depending on where deer, rabbits or empty fields are
startCoordinate = (0,0)
performanceScoreGoal = 30


# TEST WORLD GRID ONLY
worldx=6
worldy=6
coordinatesDict = {}
temporaryWorldPlaceholderDict = {}

# initialise world size empty dict
for x in range(0,worldx):
    for y in range(0, worldy):
        coordinatesDict[tuple([x,y])] = {}
        
        #TEMPORARY WORLD PLACEHOLDER - just pretending we have the world from Part 1 (only placeholder to have a "stable world" - substitute later!)
        temporaryWorldPlaceholderDict[tuple([x,y])] = [1,random.choice(scoreOnWorldGridTile)]

pprint.pprint(temporaryWorldPlaceholderDict)

for key_tuple in coordinatesDict:
    innerDict = {}
    for x in range(0,worldx):
        for y in range(0, worldy):
            # get rid of the same current coordinate from coordinatesDict for this particular "node-chain"
            if not (key_tuple[0] == x and key_tuple[1] == y):
                #only take direct neighbor grid
                if abs(key_tuple[0]-x) <= 1 and abs(key_tuple[1]-y) <=1:
                    # get rid of (1,1) distance diagonal neigbor as well
                    if not (abs(key_tuple[0]-x) == 1 and abs(key_tuple[1]-y) ==1):
                        innerDict[tuple([x,y])] = temporaryWorldPlaceholderDict[x,y] # ++++++++++++++++++++++++++++++++++ VALUES
                       
            #world is round (add edges to edges)
            
            # X either 0 or worldx-1
            if key_tuple[0] == 0 and key_tuple[1] == y:
                innerDict[tuple([worldx-1,y])] = temporaryWorldPlaceholderDict[worldx-1,y] # ++++++++++++++++++++++++++++++++++ VALUES                      
            if key_tuple[0] == worldx-1 and key_tuple[1] == y:
                innerDict[tuple([0,y])] = temporaryWorldPlaceholderDict[0,y] # ++++++++++++++++++++++++++++++++++ VALUES
            
            # Y either 0 or worldx-1
            if key_tuple[0] == x and key_tuple[1] == 0:
                innerDict[tuple([x,worldy-1])] = temporaryWorldPlaceholderDict[x,worldy-1] # ++++++++++++++++++++++++++++++++++ VALUES
            if key_tuple[0] == x and key_tuple[1] == worldy-1:
                innerDict[tuple([x,0])] = temporaryWorldPlaceholderDict[x,0] # ++++++++++++++++++++++++++++++++++ VALUES
            
           
    coordinatesDict[key_tuple] = innerDict
    
if localDebug:
    print("Full Coordinate Listing:")
    pprint.pprint(coordinatesDict)




grid_map_graph = CustomDirectedGraph(coordinatesDict)
the_problem = CritterHuntPerformanceProblem(startCoordinate, performanceScoreGoal, grid_map_graph)
node = breadth_first_graph_search(the_problem)


print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ")
if node != None:
    print("Node action:", node.action, ", state:", node.state, "parent:", node.parent, "path_cost:", node.path_cost,
          "performance_score:", node.performance_score, "depth:", node.depth)
    print("Path was:", node.path())
    for node in node.path():
        print("State", node.state, "Pathcost", node.path_cost)    
    print("Solution is:", node.solution())
else:
    print("Returned:", node)


