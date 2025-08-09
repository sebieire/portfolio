
from collections import deque
from . import settings # import module
from . thing_objects import Hut # import things
from . huntsman import Huntsman # import 
from . search_classes import Node # import 




# ---------------------------------------------------------------------------
# - - = UNINFORMED SEARCH ALGORITHMS  = - -
# ---------------------------------------------------------------------------


# ------------------------------------------
# Search (Agent - Executes Moves)
# ------------------------------------------
def search_agent(percepts):
    
    next_location, the_agent = percepts
    
    if settings.searchDebug:
        print("AGENT Location:", the_agent.location)
        print("AGENT PERCEPTS:", next_location)
    
    
    ## NORTH
    if the_agent.location[0] == 0 and next_location[0] == the_agent.world_size_x-1:        
        return "travel north"
    
    ## SOUTH
    elif the_agent.location[0] == the_agent.world_size_x-1 and next_location[0] == 0:        
        return "travel south"
    
    ## NORTH          
    elif the_agent.location[0] > next_location[0] :        
        return "travel north"
    
    ## SOUTH
    elif the_agent.location[0] < next_location[0] :        
        return "travel south"
    
    
    ## EAST
    if the_agent.location[1] == the_agent.world_size_y-1 and next_location[1] == 0:        
        return "travel east"
    
    ## WEST
    elif the_agent.location[1] == 0 and next_location[1] == the_agent.world_size_y-1:        
        return "travel west"
    
    ## EAST
    elif the_agent.location[1] < next_location[1] :        
        return "travel east"
    
    ## WEST
    elif the_agent.location[1] > next_location[1] :        
        return "travel west"


# BREADTH FIRST SEARCH
def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    # INITIAL PARENT NODE
    node = Node(problem.initial)
    if settings.searchDebug:
        print("Initial Node (state):", node.state)
    
    # if initial node is goal - return
    if problem.goal_test(node.state): # GOAL TEST
        return node
    # else add to frontier (deque)
    frontier = deque([node])
    explored = set()
    
    # in frontier
    while frontier:
        # take node out from left of deque (FIFO)
        node = frontier.popleft()
        if settings.searchDebug:
            print("--------> Investigating - Node taken from frontier:", node)
            print("Its action:", node.action, ", state:", node.state, "parent:", node.parent, "path_cost:", node.path_cost, "depth:", node.depth)        
       
        
        # and add to explored set (no duplicates in set!)
        explored.add(node.state)
        if settings.searchDebug:
            print("Explored Set so far:", explored)
            print("\n----------------------")
            print("Going through each child in current Node (expand node):")
        # each in frontier -> node.expand
        for child in node.expand(problem):
            
            if settings.searchDebug:
                print("Current cycled child is:", child)        
            
            # if not already explored and not in frontier (either solution or add to frontier)
            if child.state not in explored and child not in frontier:                
                if problem.goal_test(child.state): # GOAL TEST
                    return child
                frontier.append(child)
                
                if settings.searchDebug:
                    print("Node appended to frontier:", child)
    return None


# DEPTH FIRST SEARCH
def depth_first_graph_search(problem):
    """Search the deepest nodes in the search tree first.
        Search through the successors of a problem to find a goal.
        The argument frontier should be an empty queue.
        Does not get trapped by loops.
        If two paths reach a state, only use the first one. [Figure 3.7]"""
    frontier = [(Node(problem.initial))]  # Stack
    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and
                        child not in frontier)
    return None



# DEPTH LIMITED SEARCH
def depth_limited_search(problem, limit=20): 
    """[Figure 3.17]"""

    limit = settings.dls_limit # ADJUST LIMIT (GLOBAL VAR)
    
    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


    


