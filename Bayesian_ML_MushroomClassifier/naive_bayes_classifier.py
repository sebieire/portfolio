"""
Naive Bayes Classification Implementation
Data processing and probability calculations for machine learning
"""


import pandas as pd
import pprint as pp

from lib import settings
from lib.custom_naive_bayes_learner import NaiveBayesLearner


# just for testing within main (running on console instead of iPy book)
run_from_notebook = True


def read_in_dataFrame(file_name, column_names=None):
    """
        read in the data content to Pandas dataframe
    """    
    
    if column_names:
        the_dataframe = pd.read_csv(file_name, header=None, names=column_names)
    else:
        the_dataframe = pd.read_csv(file_name, header=None)
        
    return the_dataframe


def get_subset_random_shuffled(dataFrame, amount):
    """
        returns an subset of length amount from the dataFrame 
        that has been shuffled previously
        useful if majority of same class is listed mostly together in chunks
    """
    
    shuffled_df = dataFrame.sample(frac=1)
    
    subset = shuffled_df[:amount]
    
    return subset


def calculate_prior_class_probabilities(dataFrame, class_label='class'):
    """
        calculates prior class probabilities for given dataFrame
        default name of class label is 'class'
    """    
    
    class_count = {}
    
    # loop through dataFrame (class column) entries
    for entry in dataFrame[class_label]:
        
        # first time / is not yet included
        if entry not in class_count:
            class_count[entry] = 1
        else:
            class_count[entry] = class_count[entry] + 1            
    
    class_prior_probs = {}
    # now go through keys in class_count and calc prior prob per class
    for key in class_count:
        class_amount = class_count[key]
        class_prior_probs[key] = class_amount/ len(dataFrame)
        
        if settings.display_output:
            print("For class:", key, "the prior probability is:", class_prior_probs[key])
            
    return class_prior_probs


def calculate_evidence_probabilities(dataFrame, omit_class_label=True, class_label='class'):
    """
        calculates evidence probabilities for given dataFrame
        option to omit class column - default name of class label is 'class'
    """
    
    feature_count_dict = {}
    
    for df_column in dataFrame:
        
        # omit class
        if omit_class_label and df_column == class_label:
            
            continue
            
        else:
            feature_count_dict[df_column] = {}
            
            # loop through current col entries
            for entry in dataFrame[df_column]:
                
                # first time / is not yet included
                if entry not in feature_count_dict[df_column]:
                    feature_count_dict[df_column][entry] = 1
                else:
                    feature_count_dict[df_column][entry] += 1
            
            
    for outer_key in feature_count_dict:
        for inner_key in feature_count_dict[outer_key]:
            
            feature_count_dict[outer_key][inner_key] = feature_count_dict[outer_key][inner_key] / len(dataFrame)
        
    if settings.display_output:
        print("The probabilities of evidence for the given subset data are as follows:\n")
        pp.pprint(feature_count_dict)        

    return feature_count_dict
   

def calculate_probability_of_likelihood_of_evidences(df, class_label='class'):
    """
        calculates probability of likelihood of evidences (given class)
        in Bayes Theorem specifics: P(E|H)
    """
    
    likelihood_dict = {}
    dataFrame = df
    class_col_id = class_label    
    
    # loop through exclusive class only (no duplicates)
    for a_class in dataFrame[class_col_id].drop_duplicates(): 

        # all records with that class
        all_a_class_related_records = dataFrame[ dataFrame[class_col_id] == a_class ]        
        
        # dict placeholder for inner dict
        likelihood_dict[a_class] = {}
        
        # now cycle through cols
        for df_column in dataFrame:
            
            # all BUT the matching class col
            if df_column is not class_col_id:
                
                # current col entries
                for col_entry in dataFrame[df_column].drop_duplicates():
                    
                    # prep id for dict
                    inner_dict_id = str(df_column + "_" + col_entry)
                    
                    # find match where current column has current entry AND class is current class (on same position!)
                    evidence = dataFrame[(dataFrame[df_column] == col_entry) & (dataFrame[class_col_id] == a_class)]                                       
                    
                    likelihood_dict[a_class][inner_dict_id] = len(evidence) / len(all_a_class_related_records)

    
    if settings.display_output:
        print("The probabilities of likelihood of evidence for the given subset data are as follows:\n")
        pp.pprint(likelihood_dict)
        
    return likelihood_dict


def get_train_test_split_data(dataFrame, percentage_train_split=0.95, shuffle=True):
    """
        splits given dataframe in a train & test split
        shuffle option in the event that records are clustered
    """
    
    # shuffle option
    if shuffle:
        the_df = dataFrame.sample(frac=1).reset_index(drop=True)
    else:
        the_df = dataFrame    
    
    # split into test & train
    split_at_position = int(len(the_df) * percentage_train_split)        
    df_train_set = the_df[:split_at_position] # 0 to break
    df_test_set = the_df[split_at_position:] # break to end    

    # return both    
    return df_train_set, df_test_set


# ########################################
# MAIN / IF NOT NOTEBOOK --> TESTING ONLY
# ########################################  


if not run_from_notebook:

    # Main / Run
    if __name__ == "__main__":
        
        # params
        display_output = True
        debug_on = False

        # init settings        
        settings.init_settings(show_output=display_output, debug=debug_on)
        
        # file
        file = "./data/agaricus-lepiota.data"
        class_header_label = 'class'
        column_headers = [class_header_label, 'cap-shape', 'cap-surface', 'cap-color', 'bruises',
                          'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape',
                          'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring',
                          'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color',
                          'population', 'habitat'
                          ]
        
        # read data
        df = read_in_dataFrame(file, column_names=column_headers)
        subset = get_subset_random_shuffled(df, 500)
        
        # 1
        #calculate_prior_class_probabilities(subset, class_label=class_header_label)
        
        # 2
        #calculate_evidence_probabilities(subset, class_label=class_header_label)
        
        # 3
        #calculate_probability_of_likelihood_of_evidences(subset, class_label=class_header_label)
        
        # 4
        # split data into train & test
        train_set, test_set = get_train_test_split_data(df, percentage_train_split=0.90, shuffle=True)
        
        # 5 --------------->
        # NAIVE BAYES LEARNER CLASS (testing here)
        NBL = NaiveBayesLearner(train_set)
        
        NBL.fit_data()        
        
        # PREDICTION
        predicted_classes = []
        true_classes = test_set[class_header_label].tolist()
        
        # cycle through each row in the test set
        for index, row in test_set.iterrows():
            
            feature_row = {} # placholder dict
            
            # go through each header name
            for header_name in column_headers:
                # avoid class label
                if header_name is not class_header_label:
                    
                    # build the dict with each key = header name and value = value of current row
                    feature_row[header_name] = row[header_name]            
            
            #predicted_class, prediction_percentages = NBL.predict_(feature_row)
            predicted_class = NBL.predict_class_from_features(feature_row)
            predicted_classes.append(predicted_class)
        
        
        # predict the accuracy
        accuracy_counter = 0
        for i in range(len(predicted_classes)):
            if predicted_classes[i] == true_classes[i]:
                accuracy_counter += 1
                
        print("The Accuracy is:", accuracy_counter / len(predicted_classes))
        
        
        from sklearn.metrics import confusion_matrix
        conf_matrix = confusion_matrix(predicted_classes, true_classes)
        print(conf_matrix)
        