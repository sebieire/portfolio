# WalkSAT with Tabu Search Implementation
# A stochastic local search algorithm for solving Boolean satisfiability problems


import sys
import random
import time
import matplotlib.pyplot as plt
import statistics
from collections import deque


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --- - - = = SETTINGS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

use_console_args = True # if 'False' will use global variables instead
check_solution = True # will (double) check found solution in the end
tabu_special_case = False # required to disable tabu list for remaining clauses
enable_diagnostics = True # record time / gather data (requires check_solution)

# -- console output options
console_output_cnf = False
console_walksat_debug = False
console_solution_info = False


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = GLOBAL VARIABLES = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Set random seed for reproducibility
random_seed = 42

# below come only into effect if use_console_args == False
_fileName = 'uf20-01.cnf'  # Default test file
_executions = 100 # sys.argv[2]
_maxIterations = 1000 # sys.argv[3]
_maxRestarts = 10 # sys.argv[4]
_probability = 0.4 # 1 = 100% random selection / 0 = 100% min negative gain # sys.argv[5]
_maxTabuListLength = 5 # sys.argv[6]


# diagnostic vars
iteration_tracker = []
restart_tracker = []

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --- - - = = WALKSAT = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def walksat(cnf_formula, max_iterations, max_restarts, probability, max_tabu_list_length):
    """
    * runs walksat algorithm
    * if a solution is found:
        * will return a dictionary of variables with True/False values
        * otherwise will return 'None'
    """
    
    tabu_list = []
    
    # - - - - - - - - - - - - - - - - -
    # setup required vars
    all_variables_dict = {} # holds all variables (keys) with T/F assignment (values)
    
    # - - - - - - - - - - - - - - - - -
    # primary loop (max restarts)
    for r in range(max_restarts):
    
        # - - - - - - - -
        # STEP 1: initialise (or re-assign) dictionary with random (True/False) values for all existing vars in CNF formula
        all_variables_dict = set_variable_dict_random(cnf_formula, all_variables_dict)
        
        # - - - - - - - -
        # STEP 2: get initial cost (indicies of false clauses)
        false_clauses_indicies = get_initial_false_clause_indicies(cnf_formula, all_variables_dict)
        cost = len(false_clauses_indicies)
        
        
        # <<<< - - - - info/debug - - - - >>>>
        if console_walksat_debug:
            print("\n= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ")
            print(" * * * * * * * * * * RUN (RESTART):", r," * * * * * * * * * *")
            print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = = = ")
            
            print("\nFull Variable Dict (Initial Random Assignment):\n", all_variables_dict)
        
        # - - - - - - - - - - - - - - - - -
        # STEP 3: inner loop (iterations)
        for i in range(max_iterations):
            
            # - - - - - - - -
            # STEP 4: CHECK if all clauses are True (cost == 0) -> return varibles configuration
            if cost == 0:
                if console_walksat_debug:
                    print("\nAll clauses are satisfied! Restart:", r,  "and iteration:", i ,". Returning configuration for testing.")
                # +++ diagnostics
                if enable_diagnostics:
                    iteration_tracker.append(i)
                    restart_tracker.append(r)
                        
                return all_variables_dict
            
            # tabu special case:
            # if cost is less than 2 (only 1 False clauses remaining), tabu list will potentially block almost all 'good' variables from switching and needs to be deactivated
            # a secondary failsafe is also implemented inside the 'get_variable_from_clause' function
            
            if cost < 2:
                tabu_special_case = True
            else:
                tabu_special_case = False
                
            # <<<< - - - - info/debug - - - - >>>>
            if console_walksat_debug:
                print("\n+++++++++++++++++++++++++++++++++++++")
                print(" - - - - - ITERATION:", i+1, " - - - - -")
                print("+++++++++++++++++++++++++++++++++++++\n")
                print("Current Cost:", cost)
            
            
            # - - - - - - - -
            # STEP 5: CORE FUNCTIONALITY
            
            # choose random clause from all unsat clauses
            random_clause = cnf_formula[random.choice(false_clauses_indicies)]
            
            # get variable (implements core algorithm through multiple functions)
            variable = get_variable_from_clause(cnf_formula, random_clause, all_variables_dict, probability, tabu_list, tabu_special_case)
            
            # tabu list update
            tabu_list = update_tabu_list(tabu_list, variable, max_tabu_list_length)           
          
            # - - - - - - - -
            # STEP 6: swap selected variable in all_variables_dict
            if all_variables_dict[variable] == True:
                all_variables_dict[variable] = False
            elif all_variables_dict[variable] == False:
                all_variables_dict[variable] = True
            
            # <<<< - - - - info/debug - - - - >>>>
            if console_walksat_debug:
                print("\nxxxxx TABU LIST xxxxx\n", tabu_list)
                #print("\nFull Variable Dict After Swap:\n", all_variables_dict)
            
            
            # - - - - - - - -
            # STEP 7: get next set of false clauses indicies after 1 var swap
            false_clauses_indicies = get_false_clause_indicies(cnf_formula, all_variables_dict, false_clauses_indicies, variable)
            cost = len(false_clauses_indicies)               
    
    return None # if no solution found



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --- - - = = HELPER FUNCTIONS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


def get_variable_from_clause(cnf_formula, selected_clause, all_variables_dict, probability, tabu_list, tabu_special_case):
    """
    * 'parent function': uses other helper functions step by step to obtain variable(s) until one is found
    * calls in order:
        * 'search_variable_with_no_negative_gain'
        * 'search_variable_minimal_negative_gain' (includes 2 options)
        
    * for each step if it finds multiple variables it will pick one of them at random
    * a loop in second step insures that a variable is always picked
    * will take 'tabu_list' into account and offer alternative variable in the event that they clash
    * returns:
        * a single variable
    * takes CNF, clause, dictionary of variables, probability, tabu list and option special case flag as parameters
    """

    the_holy_variable = None # evidently very hard to find... :)
    
    # ------------------------------------------------------
    # SEARCH 'NO NEGATIVE GAIN' VAR --------->
    # ------------------------------------------------------
    
    no_negative_gain_vars = search_variable_with_no_negative_gain(cnf_formula, selected_clause, all_variables_dict)
    
    # EVALUATE:
    # if any found
    if len(no_negative_gain_vars) > 0:
            
        # MORE THAN 1 variables found
        if len(no_negative_gain_vars) > 1:
            
            the_holy_variable = random.choice(no_negative_gain_vars)
            
            # tabu list holds
            if not tabu_special_case:
                variableIsTabu, alternativeList = variable_on_tabu_list(the_holy_variable, tabu_list, no_negative_gain_vars)
                # if tabu
                if variableIsTabu:
                    # check if alt list is not empty
                    if len(alternativeList) > 0:
                        the_holy_variable = random.choice(alternativeList)
                    else:
                        the_holy_variable = None
                    
            
        # ONLY 1 variable found
        else:
            the_holy_variable = no_negative_gain_vars[0]
            
            # tabu list holds
            if not tabu_special_case:
                variableIsTabu = variable_on_tabu_list(the_holy_variable, tabu_list)
                # if tabu
                if variableIsTabu:           
                    the_holy_variable = None
    

    # <<<< - - - - info/debug - - - - >>>>
    if console_walksat_debug:
        print("\n>>> >>> >>> 'No Negative Gain' Search Results: <<< <<< <<<")
        if the_holy_variable is not None:
            print("SUCCESS! Variables found with no negative gain:", no_negative_gain_vars, "Selected Var:", the_holy_variable)
        else:
            print("FAILED! No zero gain variables have been found. (Clause:", selected_clause,")")
    
    # return result (if one has been found by now) and exit function
    if the_holy_variable is not None:
        return the_holy_variable    
    
    # --> if none of above returned a var -> continue        
    
    # ------------------------------------------------------
    # ALTERNATIVE VARS --------->
    # ------------------------------------------------------      
    
    # below will only run once in most cases
    # however if the tabu list interfers a maximum of 2 Failed runs are allowed
    # if no pass, the tabu list will be temporarily disabled on the 3rd run to force a variable selection
    # this works as an additional failsafe similar to the 'tabu_special_case' parameter to avoid 'None' being returned
    
    threshold_counter=0
    while threshold_counter < 3:        
        
        
        alt_var_list = search_variable_minimal_negative_gain(cnf_formula, selected_clause, all_variables_dict, probability)        
        
        if len(alt_var_list) > 0:
            
            # turn literals (neg to pos)
            for i in range(len(alt_var_list)):
                if alt_var_list[i] < 0:                    
                    alt_var_list[i] = -alt_var_list[i]                    
            
            # MORE THAN 1 variables found
            if len(alt_var_list) > 1:
                
                the_holy_variable = random.choice(alt_var_list)
                
                # tabu list holds
                if not tabu_special_case:
                    variableIsTabu, alternativeList = variable_on_tabu_list(the_holy_variable, tabu_list, alt_var_list)
                    # if tabu
                    if variableIsTabu:
                        # check if alt list is not empty
                        if len(alternativeList) > 0:
                            the_holy_variable = random.choice(alternativeList)
                        else:
                            the_holy_variable = None
                            
            # ONLY 1 variable found
            else:
                the_holy_variable = alt_var_list[0]
                
                # tabu list holds
                if not tabu_special_case:                    
                    variableIsTabu = variable_on_tabu_list(the_holy_variable, tabu_list)
                    # if tabu
                    if variableIsTabu:           
                        the_holy_variable = None
                        
        # <<<< - - - - info/debug - - - - >>>>
        if console_walksat_debug:
            print("\n>>> >>> >>> 'Minimum Gain' Search Results: <<< <<< <<<")
            if the_holy_variable is not None:
                print("SUCCESS! Variables found (random or minimum gain):", alt_var_list, "Selected Var:", the_holy_variable)
            else:
                print("FAILED! Failed to obtain a variable! -- rerun! Search returned variable list:", alt_var_list)
         
        if the_holy_variable is not None:
            break # break while loop and return variable
        else:
            if threshold_counter >= 1:
                tabu_special_case = True
                # <<<< - - - - info/debug - - - - >>>>
                if console_walksat_debug:
                    print("-> Notice: Tabu List has been temporarily disabled for one run.")
            threshold_counter+=1
           
    return the_holy_variable
        
    
    
    
    #rand_index = random.choice(range(len(no_negative_gain_vars))) # better than just random.choice if index is required
    #the_holy_variable = no_negative_gain_vars[rand_index]


def search_variable_with_no_negative_gain(cnf_formula, selected_clause, all_variables_dict):
    """    
    * will 'try' to find a variable that incurs no negative gain (no clause that is currently True to become False)    
    * if it finds multiple variables that fullfill above criteria it will return all    
    * returns:
        * a variable list or 'None'
    * takes CNF, clause and dictionary of variables as parameters
    """
    var_list = []
    
    for literal in selected_clause:
        
        literal_swap_has_negative_gain = False # initial assumption
        
        # go through each clause
        for clause in cnf_formula:
            
            # FILTER: if literal in clause
            if literal in clause or -literal in clause:
                
                # CHECK: if clause True (the only time a negative gain can occur)
                if check_clause(clause, all_variables_dict) == True:
                    # cover both cases (if exception use the other case)
                    try:
                        lit_pos = clause.index(literal)
                    except:
                        lit_pos = clause.index(-literal)    
                    
                    # prevent overriding or modifying the actual clause!
                    temp_clause = [item for item in clause]
                    # now swap literal value in copy
                    temp_clause[lit_pos] = -temp_clause[lit_pos]                    
                    
                    # CHECK AGAIN: if clause is now False -> negative gain has occured!
                    if check_clause(temp_clause, all_variables_dict) == False:
                        literal_swap_has_negative_gain = True
        
        # add to list if no negative gain
        if not literal_swap_has_negative_gain:
            if literal < 0: # make sure it is not negative before adding
                literal = -literal
            var_list.append(literal)
    
    
    # return list
    return var_list


def search_variable_minimal_negative_gain(cnf_formula, selected_clause, all_variables_dict, probability):
    """    
    * with chance of 'probability' select a random variable from 'selected_clause' or
    * with chance of 1 - 'probability' select the variable from 'selected_clause' with the minimal negative gain (if a tie = pick one of them at random)
    * returns:
        * a variable list or 'None'
    * takes CNF, clause and dictionary of variables and probability value as parameters
    """
    
    literal_list = []
    temp_neg_gain_counter_dict = {}
    choice = random.choices(['random_variable','min_negative_gain'], weights=[probability, 1-probability], k=1)        
    
    # CHOICE 1
    if choice[0] == 'random_variable':
        literal_list.append(random.choice(selected_clause))
    
    # CHOICE 2
    elif choice[0] == 'min_negative_gain':
        
        for literal in selected_clause:
        
            # go through each clause
            for clause in cnf_formula:
                
                # FILTER: if literal in clause
                if literal in clause:
                    
                    # CHECK: if clause True (the only time a negative gain can occur)
                    if check_clause(clause, all_variables_dict) == True:
                        # add or increase counter for this literal (measure frequencies)
                        if literal in temp_neg_gain_counter_dict:
                            temp_neg_gain_counter_dict[literal]+=1
                        else:
                            temp_neg_gain_counter_dict[literal] = 1

        # find all 'keys' (= literals) with minimum occurance
        min_occurrence = min(temp_neg_gain_counter_dict.values())
        literal_list = [key for key, value in temp_neg_gain_counter_dict.items() if value == min_occurrence]                        

    
    # return list
    return literal_list
    

def variable_on_tabu_list(a_variable, tabu_list, alternative_variables_list=None):
    """
    * checks if variables are on the tabu list
    * returns:
        * True or False
        * if alternative list is passed then in also checks those and returns modified list
    * takes variable, tabu list and optional list as parameters
    """
    
    variable_is_tabu = False
    alternative_variable_choices = []
    
    # check if variable is tabu
    if a_variable in tabu_list:
        variable_is_tabu = True
    
    # if alt list was passed then check those as well
    if alternative_variables_list != None:
        for some_var in alternative_variables_list:
            if some_var not in tabu_list:
                alternative_variable_choices.append(some_var)
        
        # return both (2nd could be empty - needs to be checked!)
        return variable_is_tabu, alternative_variable_choices
    
    # else return one
    return variable_is_tabu



def update_tabu_list(tabu_list, variable, max_tabu_list_length):
    """
    * handles FIFO tabu list
    * returns:
        * new tabu list        
    * takes tabu list, variable to add, max allowed tabu list length
    """    
    
    fifo_list = deque(tabu_list)    
    
    # add one
    fifo_list.appendleft(variable)
    
    # remove one if max length has been reached
    if len(fifo_list) > max_tabu_list_length:
        fifo_list.pop()
      
    return list(fifo_list)
    

def get_initial_false_clause_indicies(cnf_formula, all_variables_dict):
    """
    * called as one 'initial' step for each run (restart)
    * cycles through all clauses in CNF
    * returns:
        * a list with CNF indicies of false clauses (the cost can be derived from its length!)        
    * takes CNF and dictionary of variables as parameters
    """
    
    false_clauses_indicies = []   
    index_counter = 0
    
    # find initial indicies of false clauses (evaluates all clauses in CNF)
    for clause in cnf_formula:
        
        # if clause evaluates to False
        if check_clause(clause, all_variables_dict) == False:
            false_clauses_indicies.append(index_counter)
                    
        index_counter+=1
                    
    return false_clauses_indicies
              

def get_false_clause_indicies(cnf_formula, all_variables_dict, previous_false_clauses_indicies, flipped_variable):
    """
    * called in every iteration (inner loop)
    * lighter (more selective) version of 'get_initial_false_clause_indicies' function
        * does NOT cycle data of all clauses in CNF but implements filters and checks first    
    """
    
    new_false_clauses_indicies = []    
    
    # - - - - - - - - - - -
    # go through all previous false clauses and check which are still false -> add to new list
    for index in previous_false_clauses_indicies:
        # if clause evaluates to False
        if check_clause(cnf_formula[index], all_variables_dict) == False:
            new_false_clauses_indicies.append(index)
    
    # - - - - - - - - - - -
    # check remaining (previously all sat clauses) for any unsat
    index_counter = 0
    for clause in cnf_formula:
        
        # FILTER: making sure to ONLY select clauses that were not already selected above (clauses that were previous SAT and may now be UNSAT)
        if index_counter not in previous_false_clauses_indicies:
            
            # FILTER: only look through clauses that contain the flipped variable (all other could not have been affected obviously)
            if flipped_variable in clause or -flipped_variable in clause:
                
                # CHECK: if clause evaluates to False
                if check_clause(clause, all_variables_dict) == False:                    
                    new_false_clauses_indicies.append(index_counter)
                    
        index_counter+=1
        
    return new_false_clauses_indicies            

    
def check_clause(clause, all_variables_dict):
    """
    * returns True if clause is satisfied or False if not
    * takes clause and dictionary of variables as parameters
    """
    clauseTruthList = []
    
    for literal in clause:
        keySwitched = False
        if literal < 0:
            key = -literal
            keySwitched = True # switched
        else:
            key = literal
    
        if all_variables_dict[key]: # key == True
            if keySwitched:
                clauseTruthList.append(False)
            else:
                clauseTruthList.append(True)
                
        else: # key == False
            if keySwitched: 
                clauseTruthList.append(True)
            else:
                clauseTruthList.append(False)

    if True in clauseTruthList:
        return True
    else:
        return False


def set_variable_dict_random(cnf_formula, all_variables_dict):
    """
    * returns a dictionary with all variables that currently 
    exist in the passed cnf_formula as its keys
    * randomly assigns True/False values to each  
    * if passed dict is empty:
        * this means this is the first run -> create a new one
    * else: use the same and reasign random values (saves step of cycling all clauses yet again on each run)
    """
    
    if all_variables_dict: # if not empty (just re-assign "new" random values)
        
        for key in all_variables_dict:
            all_variables_dict[key] = random.choice([True, False])        
        return all_variables_dict
    
    else: # if empty (= first run -> create a new one)
        newVariableDict = {} # initial 
        
        # iterate through clauses in cnf formula
        for clause in cnf_formula:        
            # iterate each variable
            for variable in clause:            
                # if negative make positive
                if variable < 0:
                    variable = -variable
                # check if not in Dict already - then add
                if variable not in newVariableDict:
                    newVariableDict[variable] = random.choice([True, False]) # random
                    
        return newVariableDict
           

     
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --- - - = = READ FILE / CHECK SOLUTION = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def read_cnf_file(fileName):
    """
    * read in cnf file (filename = argv[1])
    * ignores 'c' & 'p' lines
    * returns full clause (list)
    """
    
    file = open(fileName, 'r')    
    fullClause = []
    
    # go through each line
    for single_line in file:
        
        # each line into list
        lineData = single_line.split()        
        
        # end of file as indicated - stop
        if lineData[0] == '%':
            break
        
        # - - - make sure not reading c or p lines (rest are clauses)
        if ['c','p'].count(lineData[0]) == 0:
            
            singleClause = [] # set/reset current single clause
            
            for variable in lineData:
                
                literal = int(variable) # str to in
                
                # skip rest of the loop
                if literal == 0:
                    continue
                # get clause elements
                else:
                    singleClause.append(literal)
            
            fullClause.append(singleClause)
    
    return fullClause



def check_solution(sat_clauses, solution):
    """
    * checks if solution is valid for given list of clauses
    """
    
    # all clauses must be true for SAT CNF to be true
    allClausesTrue = True # initial
    
    # go through each clause
    for clause in sat_clauses:        
        
        clauseTruthList = [] # set / reset per clause
        
        #go through each literal in current clauses
        for literal in clause:            
            
            keySwitched = False
            
            # if literal negative turn into positive (key) and then take opposite truth value from dict
            if literal < 0:
                key = -literal
                keySwitched = True # key switched
            else:
                key = literal
            
            # if any of the keys (literals) True then clause = True            
            if solution[key]: # if True
                if keySwitched:
                    clauseTruthList.append(False)
                else: 
                    clauseTruthList.append(True)
                    
            else: # if False
                if keySwitched:
                    clauseTruthList.append(True)
                else: 
                    clauseTruthList.append(False)
        
        # at least one literal must be true for clause to be true      
        if True not in clauseTruthList:
            allClausesTrue = False
    
    return allClausesTrue

  
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --- - - = = RUN = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def run():
    """
    * run
    """
    
    # use console args
    if use_console_args:
        
        fileName = sys.argv[1]
        executions = int(sys.argv[2])
        maxIterations = int(sys.argv[3])
        maxRestarts = int(sys.argv[4])
        probability = float(sys.argv[5])        
        maxTabuListLength = int(sys.argv[6])
        
    # use global vars
    else:
        fileName = _fileName
        executions = _executions
        maxIterations = _maxIterations
        maxRestarts = _maxRestarts
        probability = _probability
        maxTabuListLength = _maxTabuListLength
        
        
    # read CNF from file
    complete_cnf = read_cnf_file(fileName)
    if console_output_cnf:
        print("\nCNF:\n", complete_cnf)    
    
    
    solution_counter = 0
    success_runtimes_list = []
    
    for i in range(executions):
        
        random.seed(random_seed + i*1000)     
        
        # +++ diagnostics
        if enable_diagnostics:
            start_t = time.process_time()
        
        # run algorithm and get solution if any
        solution_dict = walksat(complete_cnf, maxIterations, maxRestarts, probability, maxTabuListLength)        
        
        # +++ diagnostics
        if enable_diagnostics:
            stop_t = time.process_time()
            run_time = stop_t - start_t
        
        # check solutions
        if solution_dict != None:
            solution = check_solution(complete_cnf, solution_dict)
            
            if solution:
                solution_counter+=1
                
                # +++ diagnostics
                if enable_diagnostics:
                    success_runtimes_list.append(run_time)                    
                
            if console_solution_info:
                print("The solution is:", solution)
        else:            
            if console_solution_info:
                print("\n\nNo Solution Has Been Found for max of", maxRestarts, "restarts and",maxIterations,"iterations!" )
                
                
    print("\n\nOut of:",executions,"executions ",solution_counter,"solutions were found.")
    print((solution_counter/executions)*100, "% success rate.")
    
    # +++ diagnostics
    if enable_diagnostics:
                
        # sort list        
        success_runtimes_list.sort()
        
        # list of k successful runs
        successful_runs = list(range(0,len(success_runtimes_list),1))
        successful_runs[:] = [x / executions for x in successful_runs]        
        
        plt.plot(success_runtimes_list,successful_runs)
        plt.ylabel('P (solve)')
        plt.xlabel('run-time (sys+cpu)')
        plt.show()
        
        # additional data
        print("\n++++++++++++++++++")
        print(" - - ANALYSIS - -")
        print("++++++++++++++++++\n")
        
        try:
            print("Runtimes Mean:", statistics.mean(success_runtimes_list))
            print("Runtimes Median:", statistics.median(success_runtimes_list))        
            print("Runtimes Max:", max(success_runtimes_list))
            print("Runtimes Min:", min(success_runtimes_list))
        except:
            pass

        try:
            print("Runtimes Std Deviation:", statistics.stdev(success_runtimes_list))
            print("Runtimes Variance:", statistics.variance(success_runtimes_list))
        except:
            pass
       
        print("\nAverage Rounds Restarted (All Executions):", statistics.mean(restart_tracker))
        
        print("\n--- ITERATIONS DATA ---")
        plt.hist(iteration_tracker, bins=10, rwidth=0.8, label='Iterations', alpha=0.8)
        plt.ylabel('Occurance');
        x_label = 'Iteration when solution was found (max:' + str(maxIterations) + ')'
        plt.xlabel(x_label)        
   

run()


