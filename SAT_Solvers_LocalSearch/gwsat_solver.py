# GWSAT SAT Solver Implementation
# A stochastic local search algorithm for solving Boolean satisfiability problems


import sys
import random
import time
import matplotlib.pyplot as plt
import statistics

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --- - - = = SETTINGS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

use_console_args = True # if 'False' will use global variables instead
check_solution = True # will (double) check found solution in the end
enable_diagnostics = True # record time / gather data (requires check_solution)

# -- console output options
console_output_cnf = False
console_gwsat_debug = False
console_solution_info = False


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = GLOBAL VARIABLES = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# Set random seed for reproducibility
random_seed = 42

# below come only into effect if use_console_args == False
_fileName = 'uf20-01.cnf'  # Default test file
_executions = 100 # sys.argv[2]
_maxIterations = 1000 # # sys.argv[3]
_maxRestarts = 10 # # sys.argv[4]
_walkProbability = 0.4 # 1 = 100% walk / 0 = 100% gsat # # sys.argv[5]

# diagnostic vars
iteration_tracker = []
restart_tracker = []

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --- - - = = GWSAT = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def gwsat(cnf_formula, max_iterations, max_restarts, walk_probability):
    """
    * runs gwsat algorithm
    * if a solution is found:
        * will return a dictionary of variables with True/False values
        * otherwise will return 'None'
    """
    
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
        # STEP 2: get initial cost / initial literals of false clauses
        false_clauses_indicies, literals_frequency_in_unsat_dict = get_initial_false_clause_indicies_and_literal_frequencies(cnf_formula, all_variables_dict)
        cost = len(false_clauses_indicies)                
        
        
        # <<<< - - - - info/debug - - - - >>>>
        if console_gwsat_debug:
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
                if console_gwsat_debug:
                    print("\nAll clauses are satisfied! Restart:", r,  "and iteration:", i ,". Returning configuration for testing.")
                # +++ diagnostics
                if enable_diagnostics:
                    iteration_tracker.append(i)
                    restart_tracker.append(r)
                return all_variables_dict
            
            
            # - - - - - - - -
            # STEP 5: random choice of: GSAT or WALK step
            choice = random.choices(['gsat','walk'], weights=[1-walk_probability, walk_probability], k=1)
            
            
            # gsat
            if choice[0] == 'gsat': 
            
                # - - - - - - - -
                # STEP 6-A: from unsat dictionary get variable (key in dict) occuring the most (max)
                if len(literals_frequency_in_unsat_dict) != 0:
                    max_occurrence = max(literals_frequency_in_unsat_dict.values())
                    literals_list_max_occurance = [key for key, value in literals_frequency_in_unsat_dict.items() if value == max_occurrence]
                else: # should not occur unless there is an error or no occurances (cost == 0)
                    max_occurrence = None
                    literals_list_max_occurance = None
                
                # - - - - - - - -
                # STEP 7-A: select one var from max occurance list (at random if more than 1)
                var_to_swap = random.choice(literals_list_max_occurance) # random
                if var_to_swap < 0:
                    var_to_swap = -var_to_swap #neg -> pos
            
            # walk
            else: 
                # - - - - - - - -
                # STEP 6-B: randomly select a clause from all unsat clauses
                walk_clause = cnf_formula[random.choice(false_clauses_indicies)] # random clause (from all false clauses)
                
                # - - - - - - - -
                # STEP 7-B: select one literal of that clause (at random)
                var_to_swap = random.choice(walk_clause)
                if var_to_swap < 0:
                    var_to_swap = -var_to_swap #neg -> pos
          
            # - - - - - - - -
            # STEP 8: swap selected variable in all_variables_dict
            if all_variables_dict[var_to_swap] == True:
                all_variables_dict[var_to_swap] = False
            elif all_variables_dict[var_to_swap] == False:
                all_variables_dict[var_to_swap] = True
            
            
            # <<<< - - - - info/debug - - - - >>>>
            if console_gwsat_debug:
                print("\n+++++++++++++++++++++++++++++++++++++")
                print(" - - - - - ITERATION:", i+1, " - - - - -")
                print("+++++++++++++++++++++++++++++++++++++\n")
                print("- - - Step Choice:", choice[0], "- - -\n")
                print("Current Cost:", cost)
                
                if choice[0] == 'gsat':
                    print("Max occurance value:", max_occurrence)
                    print("Literals list (of that max occ value):", literals_list_max_occurance)
                else:
                    print("Walk Clause:", walk_clause)
                    
                print("Swap Variable (selected at random): ------------------>", var_to_swap)
                print("\nFalse Clauses Indicies:", false_clauses_indicies)
                print("All Literals in Unsat (amount):", literals_frequency_in_unsat_dict)
                print("\nFull Variable Dict After Swap:\n", all_variables_dict)
            
            
            # - - - - - - - -
            # STEP 9: get next set of false clauses indicies & literals frequency after 1 var swap
            false_clauses_indicies, literals_frequency_in_unsat_dict = get_false_clause_indicies_and_literal_frequencies(cnf_formula, all_variables_dict, false_clauses_indicies, var_to_swap)
            cost = len(false_clauses_indicies)               
    
    return None # if no solution found



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# --- - - = = HELPER FUNCTIONS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def get_initial_false_clause_indicies_and_literal_frequencies(cnf_formula, all_variables_dict):
    """
    * called as one 'initial' step for each run (restart)
    * cycles through all clauses in CNF
    * returns:
        * a list with CNF indicies of false clauses (the cost can be derived from its length!)
        * a dictionary of all literals present in those clauses (keys) and their frequency (values)
    * takes CNF and dictionary of variables as parameters
    """
        
    false_clauses_indicies = []
    literals_frequency_in_unsat = {}
    index_counter = 0
    
    # find initial indicies of false clauses (evaluates all clauses in CNF)
    for clause in cnf_formula:
        
        # if clause evaluates to False
        if check_clause(clause, all_variables_dict) == False:
            false_clauses_indicies.append(index_counter)
            
            # from all unsat clauses - get literal frequency (-> dict)
            for literal in clause:
                if literal in literals_frequency_in_unsat:
                    literals_frequency_in_unsat[literal] += 1 # increase by 1
                else:
                    literals_frequency_in_unsat[literal] = 1
                    
        index_counter+=1
                    
    return false_clauses_indicies, literals_frequency_in_unsat



def get_false_clause_indicies_and_literal_frequencies(cnf_formula, all_variables_dict, previous_false_clauses_indicies, flipped_variable):
    """
    * called in every iteration (inner loop)
    * lighter (more selective) version of 'get_initial_false_clause_indicies_and_literal_frequencies' function
        * does NOT cycle data of all clauses in CNF but implements filters and checks first
    * this works no matter if gsat or walk is running
    """
    
    new_false_clauses_indicies = []
    literals_frequency_in_unsat = {}
    
    # - - - - - - - - - - -
    # go through all previous false clauses and check which are still false -> add to new list
    for index in previous_false_clauses_indicies:
        # if clause evaluates to False
        if check_clause(cnf_formula[index], all_variables_dict) == False:
            new_false_clauses_indicies.append(index)            
            
            # from all unsat clauses - get literal frequency (-> dict)
            for literal in cnf_formula[index]:
                if literal in literals_frequency_in_unsat:
                    literals_frequency_in_unsat[literal] += 1
                else:
                    literals_frequency_in_unsat[literal] = 1
    
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
                    
                    # from all unsat clauses - get literal frequency (-> dict)
                    for literal in clause:
                        if literal in literals_frequency_in_unsat:
                            literals_frequency_in_unsat[literal] += 1
                        else:
                            literals_frequency_in_unsat[literal] = 1
    
        index_counter+=1
        
    return new_false_clauses_indicies, literals_frequency_in_unsat
            
            

    
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
        walkProbability = float(sys.argv[5])
        
    # use global vars
    else:
        fileName = _fileName
        executions = _executions
        maxIterations = _maxIterations
        maxRestarts = _maxRestarts
        walkProbability = _walkProbability
        
        
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
        solution_dict = gwsat(complete_cnf, maxIterations, maxRestarts, walkProbability)        
        
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
                print("\n\nNo Solution Has Been Found for", maxRestarts, "restarts and",maxIterations,"iterations!" )
                
    print("\n\nOut of:",executions,"executions:",solution_counter,"solutions were found.")
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
        
        print("Runtimes Mean:", statistics.mean(success_runtimes_list))
        print("Runtimes Median:", statistics.median(success_runtimes_list))
        print("Runtimes Max:", max(success_runtimes_list))
        print("Runtimes Min:", min(success_runtimes_list))
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


