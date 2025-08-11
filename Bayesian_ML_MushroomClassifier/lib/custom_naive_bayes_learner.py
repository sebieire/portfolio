"""
    Custom Naive Bayes Learner Implementation
    Machine learning classifier with probability calculations
"""

import pprint as pp
from . import settings


class NaiveBayesLearner():
    """
        A custom implementation of a Naive Bayes Learner algorithm.
        Provides a fit and predict class as per ML practices.
        Implements a number of helper functions as previously used in task 2.1.
    """
    
    #####################################################
    # Constructor
    #####################################################
    
    def __init__(self, dataFrame, class_label='class'):
        
        # passed vars
        self.dataFrame = dataFrame
        self.class_label = class_label
        
        # data placeholders
        self.prior_class_probs = {}
        self.evidence_probs = {}
        self.likelihood_probs = {}
    
    
    #####################################################
    # Fit Function
    #####################################################
    
    def fit_data(self):
        """
            fit function (custom)
            runs helper functions to accumulate (fit) data
        """        
        self.calculate_prior_class_probabilities()
        self.calculate_evidence_probabilities()
        self.calculate_probability_of_likelihood_of_evidences()
        
    
    #####################################################
    # Predict Function
    #####################################################
    
    def predict_class_from_features(self, feature_rows_dict):
        """
            predict function (custom)
            predicts result for one input dict (= row or 'feature vector') at a time for given class
        """
        
        # placeholders
        predictions_dict = {}
        prediction_values_sum = 0
        
        # loop through exclusive class only (no duplicates)
        for a_class in self.dataFrame[self.class_label].drop_duplicates():
            
            # current cycled class prior probability
            a_class_prior_prob = self.prior_class_probs[a_class]
            likelihood = a_class_prior_prob
            
            # go trough the dict
            for feature in feature_rows_dict:
                str_key = str(feature + "_" + feature_rows_dict[feature])
                likelihood = likelihood * self.likelihood_probs[a_class][str_key]
            
            predictions_dict[a_class] = likelihood            
        
        # combined probability of each class for the feature vector
        for prediction in predictions_dict:
            prediction_values_sum += predictions_dict[prediction] # sum up
            
        # now normalize the values in predictions_dict accordingly (divide by sum) to get correct values
        # also find max value
        max_value = 0
        predicted_class = None
        for prediction in predictions_dict:
            predictions_dict[prediction] = predictions_dict[prediction] / prediction_values_sum
            
            # if max value -> predicted class found
            if predictions_dict[prediction] > max_value:
                predicted_class = prediction            
        
        # return values       
        return predicted_class
        
    
    #####################################################
    # Helpers (previously used functions - here modified)
    #####################################################    
        
    def calculate_prior_class_probabilities(self):
        """        
            calculates prior class probabilities for given dataFrame        
            --> updates: self.prior_class_probs
        """    
    
        class_count = {}
        
        # loop through dataFrame (class column) entries
        for entry in self.dataFrame[self.class_label]:
            
            # first time / is not yet included
            if entry not in class_count:
                class_count[entry] = 1
            else:
                class_count[entry] += 1        
        
        # now go through keys in class_count and calc prior prob per class
        for key in class_count:
            class_amount = class_count[key]
            self.prior_class_probs[key] = class_amount/ len(self.dataFrame)            
        
            # allow debug mode
            if settings.display_output & settings.debug_mode:
                print("For class:", key, "the prior probability is:", self.prior_class_probs[key])      
          
            
    def calculate_evidence_probabilities(self, omit_class_label=True):
        """
            calculates evidence probabilities for given dataFrame            
        """
        
        for df_column in self.dataFrame:
            
            # omit class
            if omit_class_label and df_column == self.class_label:
                
                continue
                
            else:
                self.evidence_probs[df_column] = {}
                
                # loop through current col entries
                for entry in self.dataFrame[df_column]:
                    
                    # first time / is not yet included
                    if entry not in self.evidence_probs[df_column]:
                        self.evidence_probs[df_column][entry] = 1
                    else:
                        self.evidence_probs[df_column][entry] += 1
                
                
        for outer_key in self.evidence_probs:
            for inner_key in self.evidence_probs[outer_key]:
                
                self.evidence_probs[outer_key][inner_key] = self.evidence_probs[outer_key][inner_key] / len(self.dataFrame)
            
        # allow debug mode
        if settings.display_output & settings.debug_mode:
            print("The probabilities of evidence for the given subset data are as follows:\n")
            pp.pprint(self.evidence_probs)
            
            
            
    def calculate_probability_of_likelihood_of_evidences(self):
        """
            calculates probability of likelihood of evidences (given class)
            in Bayes Theorem specifics: P(E|H)
        """                       
        
        # loop through exclusive class only (no duplicates)
        for a_class in self.dataFrame[self.class_label].drop_duplicates(): 
    
            # all records with that class
            all_a_class_related_records = self.dataFrame[ self.dataFrame[self.class_label] == a_class ]        
            
            # dict placeholder for inner dict
            self.likelihood_probs[a_class] = {}
            
            # now cycle through cols
            for df_column in self.dataFrame:
                
                # all BUT the matching class col
                if df_column is not self.class_label:
                    
                    # current col entries
                    for col_entry in self.dataFrame[df_column].drop_duplicates():
                        
                        # prep id for dict
                        inner_dict_id = str(df_column + "_" + col_entry)
                        
                        # find match where current column has current entry AND class is current class (on same position!)
                        match = self.dataFrame[ (self.dataFrame[df_column] == col_entry) & (self.dataFrame[self.class_label] == a_class) ]                                       
                        
                        self.likelihood_probs[a_class][inner_dict_id] = len(match) / len(all_a_class_related_records)
    
        
        if settings.display_output & settings.debug_mode:
            print("The probabilities of likelihood of evidence for the given subset data are as follows:\n")
            pp.pprint(self.likelihood_probs)            
        
        
                

