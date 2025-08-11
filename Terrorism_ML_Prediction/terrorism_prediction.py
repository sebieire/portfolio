# Terrorism Group Prediction using Machine Learning
# A comprehensive ML project analyzing the Global Terrorism Database


"""
NOTES: 
The below code will run "out of the box". The run() function is the start. (IMPORTANT! globalterrorismdb_0718dist.csv must be in same folder!)

By default it will run experiment_two but with hyper_parameter_opt off.
Individual parameters in each experiment can be changed to get desired outcome.

numberOfMostActiveGroups is currently set to 3 but can be changed if desired below on line 75.

"""

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from collections import Counter
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer
from sklearn.cluster import DBSCAN

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(precision=3, suppress=True) # numpy screenfriendly print options


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = DEBUG SETTINGS & FLAGS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -


# info & debug
enableNullValueOutputAtInitStage = False # shows the null value count of features at init stage

# exports
exportCleanFullDataFrames = False # will export clean readable CSV versions of the full DF
exportCleanExtractedDataFrames = False # will export clean readable CSV versions of the extracted DF

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = GLOBAL VARS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

# note: some of the variables below do not actually need to be declared here
# however for an easier understanding (and overview) of what vars are being used
# through out those are declared as placeholders / globals:

RANDOM_SEED = 1234
numberOfMostActiveGroups = 3
df = pd.DataFrame() # holds data at init stage
df_mostActiveGroups = pd.DataFrame() # holds data for x top active groups
encodedCategoryTargetsDict = {} # encoded int values + original str values of classifier


# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = INITIALIZE DATA = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def preselect_and_read_data():
    ''' 
    * preselect relevant columns from data
    * read in CSV with those columns
    * change order of columns
    * changes columns names to be more descriptive
    '''
    
    global df
    
    # select only relevant columns
    data_columns = ['iyear','imonth','iday','country','region','provstate','city','latitude',
                'longitude','multiple','success','suicide','attacktype1','targtype1',
                'targsubtype1','gname','individual','weaptype1','nkill','nwound']
    
    # read in csv with selected columns
    df = pd.read_csv('globalterrorismdb_0718dist.csv',usecols=data_columns, encoding='ISO-8859-1')
    
    # reorder cols so target class is first column
    df = df[['gname','iyear','imonth','iday','country',
             'region','provstate','city','latitude', 'longitude',
             'multiple','success','suicide','attacktype1','targtype1','targsubtype1',
             'individual','weaptype1','nkill','nwound']]
 
def rename_cols():
    ''' 
    * changes columns names to be more descriptive
    '''
    global df
    
    # rename cols
    df = df.rename(columns={'gname':'TerroristGroup', 'iyear':'Year', 'imonth':'Month', 'iday':'Day', 'country':'Country',
                            'region':'GlobalRegion', 'provstate':'CountryRegion', 'city':'City', 'latitude':'Latitude', 'longitude':'Longitude',
                            'multiple':'MultipleAttacks', 'success':'SuccessfulAttack', 'suicide':'SuicideAttack', 'attacktype1':'AttackType',
                            'targtype1':'Target', 'targsubtype1':'TargetSubType', 'individual':'IndividualAttack', 'weaptype1':'WeaponType',
                            'nkill':'CasualtiesDead', 'nwound':'CasualtiesWounded'})
 

    
    
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = PRE-PROCESSING FUNCTIONS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def manual_prepro_missing_values():
    '''
    * manual pre-processing of missing values where possible
    * either 0 or 'unknown' (as applicable)
    '''
    
    # all missing values that should (empirically) be 0
    df['MultipleAttacks'] = df['MultipleAttacks'].replace(np.nan, 0)
    df['CasualtiesDead'] = df['CasualtiesDead'].replace(np.nan, 0)
    df['CasualtiesWounded'] = df['CasualtiesWounded'].replace(np.nan, 0)
    df['TargetSubType'] = df['TargetSubType'].replace(np.nan, 0)
    
    # all missing values that should (empirically) be 'unknown'
    df['CountryRegion'] = df['CountryRegion'].replace(np.nan, 'unknown')
    df['City'] = df['City'].replace(np.nan, 'unknown')
    
        
def manual_prepro_str_to_lower_case():
    '''
    * all lower case strings
    * will fix different capitalization of the same words (e.g. 'Unknown' & 'unknown')
    '''
    
    df['TerroristGroup'] = df['TerroristGroup'].str.lower()
    df['CountryRegion'] = df['CountryRegion'].str.lower()
    df['City'] = df['City'].str.lower()    

   
def get_top_active_groups(amount=2):
    '''
    * returns dataframe with most active X terrorist groups (default = 2)
    * NOTE: first one (pos 0) is always 'unknown' - so starting position is 1 (not 0!)
    '''
    
    # increment by one due to pos 0 = 'unknown'
    amount = amount+1 
    # get top X groups (includes 'unknown')
    top_X_groups = df['TerroristGroup'].value_counts()[:amount].index.tolist()

    counter = 0
    tempDataFrame = pd.DataFrame()
    
    for value in top_X_groups:        
        # only proceed if not position (value) 0
        if counter>0 :
            # get current group only from df
            df_currentGroup = df.loc[df['TerroristGroup'].isin([value])]
            # setup concat and then merge
            concatDFS = [tempDataFrame,df_currentGroup]
            tempDataFrame = pd.concat(concatDFS)
        counter+=1    
    
    # return df
    return tempDataFrame


def nominal_encode_values(dataFrame, valueList):
    '''
    * encodes categorical data (valueList) of the passed dataframe
    * returns dataframe
    '''
    global encodedCategoryTargetsDict
    global numberOfMostActiveGroups
    
    encoder = OrdinalEncoder()    
    
    for value in valueList:
        dataFrame[value] = encoder.fit_transform(dataFrame[[value]])
        
        # this will pair key and string of the classification target into a dict
        if value == 'TerroristGroup': #only if classification target
            key = 0 # initial
            for item in encoder.categories_[0]:
                encodedCategoryTargetsDict[key] = item
                key+=1
    
    return dataFrame


def scaling_normalisation_min_max(dataFrame):
    '''
    * min max data normalisation        
    * returns a numpy array
    '''
    
    min_max_scaler = MinMaxScaler()
    tempArray = min_max_scaler.fit_transform(dataFrame)    
    return tempArray
    
    '''
    # ignore first column as that is the classifier
    tempArray = min_max_scaler.fit_transform(dataFrame[dataFrame.columns[1:20]])    
    print(tempArray)
    newDataFrame = pd.DataFrame(data=tempArray[0:,0:],
                                index=tempArray[0:,0],
                                columns=dataFrame.columns[1:20])
        
    # adding the classifier column back in (as integer values in this case)
    tGroup_col_numeric = pd.to_numeric(dataFrame['TerroristGroup'], downcast='integer')      
    newDataFrame.insert(0,'TerroristGroup',tGroup_col_numeric.values) 
    
    return newDataFrame
    '''

def scaling_standardization(dataFrame):
    '''
    * standard scaling
    * returns a numpy array
    '''    
    standardScaler = StandardScaler()
    tempArray = standardScaler.fit_transform(dataFrame)    
    return tempArray


def outliers_univariate(dataFrame, id_min_max_bracket, showInfo=False):
    '''
    * preprocess outliers (univariate):    
        * id_min_max_bracket: dictionary with id (column) and min and max values (inclusive!) bracket of values to include
        * showInfo: will visualise before/after changes as scatterplot and output useful data
    * returns dataFrame
    '''
    startSize = dataFrame.shape[0]
    if showInfo:
        print("\n- - - - - - - - - - - - - - - - - - - - - - - ")
        print("\n- - = = Univariate Outlier Detection = = - -")
        print("\nShape before outlier deletion:", dataFrame.shape)
    
    for key in id_min_max_bracket.items():
        
        currentSize = dataFrame.shape[0]
        
        if showInfo: # Visualise Before
            print("\n- - Before - - ")
            print("Key & Bracket:", key)
            visualise_outliers_individual(dataFrame[key[0]])            
        
        dataFrame = dataFrame[ (dataFrame[key[0]] >= key[1][0]) & (dataFrame[key[0]] <= key[1][1]) ]
        
        if showInfo: # Visualise After
            print("\n- - After - - ")
            print("Length after outlier deletion:", dataFrame.shape[0])
            print("Deleted rows (current outlier):", currentSize-dataFrame.shape[0]) # how many rows this round
            visualise_outliers_individual(dataFrame[key[0]])            
            
    if showInfo:        
        print("\n ---> Deleted rows altogether (univariate outlier processing):", startSize-dataFrame.shape[0]) # how many rows deleted altogether
    
    return dataFrame

    
    

def outliers_multivariate(the_data_array, twoColNamesToCluster, the_eps=0.5, minSamples=4, returnAsDF=False, showInfo=False):
    '''
    * preprocess multiple outliers (multivariate):
        * passed in data MUST BE SCALED!
        * may use dataFrame as input
        * twoColNames / twoColIndicies: will only process 2 indicies with DBScan (might expand this in the future)
        * showInfo: outputs scatter plot and other info if True
    * returns numpyArray (optionally as DF if it was passed in as DF and returnAsDF=True)
    '''
    
    initialRowLength = the_data_array.shape[0] # for info only
    
    if showInfo:        
        print("\n- - - - - - - - - - - - - - - - - - - - - - - ")
        print("\n- - = = Multivariate Outlier Detection = = - -")
        print("Clustering two columns:", twoColNamesToCluster[0], "and", twoColNamesToCluster[1])
        print("\nDatasize initial:", the_data_array.shape)
    
    colIndicies = [] # get indicies for both passed column names (2 only for now!)
    colIndicies.append(df_mostActiveGroups.columns.get_loc(twoColNamesToCluster[0]))
    colIndicies.append(df_mostActiveGroups.columns.get_loc(twoColNamesToCluster[1]))    
    
    # - - - - - - - - 
    # Convert to npArray
    if not isinstance(the_data_array, np.ndarray):                
        the_data_array = the_data_array.to_numpy()
    
    # - - - - - - - -
    # Remove/check for NaN values (drop rows in that case)
    the_data_array = the_data_array[~np.isnan(the_data_array).any(axis=1)]
    
    if showInfo:
        print("Datasize after NaN removal:", the_data_array.shape)
        print("Removed NaN rows count", initialRowLength - the_data_array.shape[0])
        plt.scatter(the_data_array[:,colIndicies[0]], the_data_array[:,colIndicies[1]], label="Before", marker='X')
    
    # - - - - - - - -
    # Use clustering based outlier detection
    db_scan = DBSCAN(eps=the_eps, min_samples=minSamples)
    selectedCols = the_data_array[:,colIndicies] # cluster passed in indicies
    db_scan.fit(selectedCols)
    
    # Boolean map (True = NON-outliers) / used here to get all non-outlier indicies from data_array
    boolNonOutlierMap = db_scan.labels_!=-1 # this step could be omitted (easier to read this way)
    the_data_array = the_data_array[boolNonOutlierMap]
    if showInfo:
        print("\nDatasize after outlier removal:", the_data_array.shape)
        print("Removed noisy samples count:", Counter(db_scan.labels_)[-1])
        
        plt.scatter(the_data_array[:,colIndicies[0]], the_data_array[:,colIndicies[1]], label="After", marker='.')
        plt.legend(loc='lower right')
        plt.show()    
        
    # - - - - - - - -
    # Return as dataframe
    if returnAsDF:
        newDataFrame = pd.DataFrame(data=the_data_array[0:,0:],
                                    index=the_data_array[0:,0],
                                    columns=df_mostActiveGroups.columns[0:20])
        
        return newDataFrame # return without labels
    
    # Return as numpy array
    else:        
        return the_data_array


def get_split_test_and_train_data(dataFrame, testSize=0.25):
    '''
    * split data into train/test features and labels
    * return as numpy objects
    '''    
    
    allFeatureData = dataFrame.drop('TerroristGroup', axis=1) # dataframe (minus classification col)
    allLabelData = dataFrame['TerroristGroup'] # series object (classification col)
    
    train_features, test_features, train_labels, test_labels = train_test_split(allFeatureData.to_numpy(),
                                                                                allLabelData.to_numpy(),
                                                                                random_state=RANDOM_SEED,
                                                                                test_size=testSize,
                                                                                shuffle=True)
    return train_features, test_features, train_labels, test_labels




# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = VISUALISATION / CONSOLE OUTPUT FUNCTIONS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -    

def visualise_outliers_sections(dataFrame, showInSections=True):
    '''
    * uses boxplot to visualise potential outliers
    * option to show all or sections
    '''
    
    if isinstance(dataFrame, np.ndarray):
        dataFrame = pd.DataFrame(data=dataFrame[0:,0:],
                                    index=dataFrame[0:,0],
                                    columns=dataFrame[0,:])
        
    
    if showInSections:
        # excludes first column (which is not normalised)
        sns.boxplot(data=dataFrame[dataFrame.columns[1:4]])
        plt.show()
        sns.boxplot(data=dataFrame[dataFrame.columns[4:10]])
        plt.show()
        sns.boxplot(data=dataFrame[dataFrame.columns[10:14]])
        plt.show()
        sns.boxplot(data=dataFrame[dataFrame.columns[14:17]])
        plt.show()
        sns.boxplot(data=dataFrame[dataFrame.columns[17:21]])
        plt.show()
    
    # all in one boxplot
    else:
        sns.boxplot(data=dataFrame)
        plt.show()
        
        '''
        #show individual (moved to other function)
        columns = list(dataFrame)
        for col in columns:
            sns.boxplot(x=dataFrame[col])
            plt.show()
        '''
        

def visualise_outliers_individual(dataFrameColumn):
    '''
    * uses boxplot to visualise potential outliers for the passed DF column    
    '''
    sns.boxplot(x=dataFrameColumn)
    plt.show()
    
def visualise_bar_chart(x_scale_objects, y_scale_values, y_label, title, labelRotation=False, setYMinMax=False, y_min_max=None):
    '''
    * displays a bar chart
    '''    
    X = np.arange(len(x_scale_objects))
    plt.bar(X, y_scale_values, align='center', alpha=0.8 )
    plt.xticks(X, x_scale_objects)
    plt.ylabel(y_label)
    plt.title(title)
    if labelRotation:
        plt.xticks(rotation=90)
    if setYMinMax:
        plt.ylim(y_min_max[0],y_min_max[1])
    plt.show()

def output_value_counts(dataFrame, target):
    '''
    * outputs value counts for target in format with their names
    '''
    
    print("\n- - = = Current Value Counts Of Column:", target ,"= = - -")
    valueCounts = dataFrame[target].value_counts()
    for key in valueCounts.keys():
        print(encodedCategoryTargetsDict[key], ":", valueCounts[key])


def output_null_value_counts(dataFrame,target):
    '''
    * outputs null value counts for target
    * if target = 'all' it will output for all columns
    '''
    
    print("\n- - = = Current Null Values For Column:", target ,"= = - -")        
    if target == 'all':
        print(dataFrame.isnull().sum())
    else:
        print(dataFrame[target].isnull().sum())



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = PRE-PROCESS STEPS - WRAPPERS = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def initialise_data():
    '''
    * combines steps required to initialise data and make first changes
    '''
    preselect_and_read_data() 
    rename_cols()
    
    if enableNullValueOutputAtInitStage:
        output_null_value_counts(df,'all')
        
    if exportCleanFullDataFrames:
        df.to_csv('global_terr_1_no_prepro.csv',index=False)
    

def preprocess_initial(numberActiveGroups, showInfo=True):
    '''
    * runs initial pre-processing steps:
        * manual data sorting
        * nominal encoding of categorical data
    * returns dataframe of most active terrorist groups
    '''
    
    manual_prepro_missing_values()
    manual_prepro_str_to_lower_case()
    
    # output and debug
    if enableNullValueOutputAtInitStage:
        output_null_value_counts(df,'all')
        
    if exportCleanFullDataFrames:
        df.to_csv('global_terr_2_prepro.csv',index=False)    
   
    # get X most active terrorist groups
    dataFrame = get_top_active_groups(numberActiveGroups)
    
    if showInfo:
        print("\n- - = =",numberActiveGroups,"MOST ACTIVE GROUPS = = - -")
        print("The", numberActiveGroups,"Most Active Groups - Instance Counts (Initial):")
        print(dataFrame['TerroristGroup'].value_counts())
    if exportCleanExtractedDataFrames:
        dataFrame.to_csv('global_terr_3_most_active.csv',index=False)   
    
    # encode (nominal encoding) categorical values (as per list)
    valueList = ['TerroristGroup','CountryRegion','City']
    dataFrame = nominal_encode_values(dataFrame,valueList)
    
    return dataFrame


def preprocess_impute(dataFrame, process_type='simple', missing_values=np.nan, strategy='mean', showInfo=False):
    '''
    * implements imputation with imputer options
        * 'simple' or 'iterative'
    '''
    
    if showInfo:
        print("\n- - - - - - - - - - - - - - - - - - - - - - - ")
        print("\n- - = = MISSING VALUE IMPUTATION = = - -")
        print("Imputation Type:", process_type)        

    # SIMPLE IMPUTER
    if process_type == 'simple':
        imputer = SimpleImputer(missing_values=missing_values,strategy=strategy)
        values = imputer.fit_transform(dataFrame)        
        dataFrame = pd.DataFrame(data = values, columns = dataFrame.columns)         
    
    # MULTIVARIATE IMPUTER
    else:        
        imputer = IterativeImputer(random_state=RANDOM_SEED)
        values = imputer.fit_transform(dataFrame)
        dataFrame = pd.DataFrame(data = values, columns = dataFrame.columns)
    
    if showInfo:        
        print("\nCurrent Null Values Per Column (After Imputation):")
        print(dataFrame.isnull().sum())
        
    return dataFrame    


def preprocess_scaling(dataFrame, scalingType='MinMax', returnAsDF=True,  includeLabelCol=True):
    '''
    * implements scaling options:
        * uses either 'MinMax' (default) or 'StandardScaler'
        * will not process label column (col1) and either return all cols (with label) or without label
    * returns either dataFrame or numpyArray
    '''
    
    # Scaler Type
    if scalingType != 'MinMax':        
        scaled = scaling_standardization(dataFrame[dataFrame.columns[1:20]])    
    else:
        scaled = scaling_normalisation_min_max(dataFrame[dataFrame.columns[1:20]])
    
    # Return as dataframe
    if returnAsDF:
        newDataFrame = pd.DataFrame(data=scaled[0:,0:],
                                    index=scaled[0:,0],
                                    columns=dataFrame.columns[1:20])
        # 1st col (label) in
        if includeLabelCol:
            # adding the classifier column back in (as unscaled integer values in this case)
            tGroup_col_numeric = pd.to_numeric(dataFrame['TerroristGroup'], downcast='integer')      
            newDataFrame.insert(0,'TerroristGroup',tGroup_col_numeric.values) 
            return newDataFrame
        
        # 1st col (label) out
        else:
            return newDataFrame # return without labels
    
    # Return as numpy array
    else:
        # 1st col (label) in
        if includeLabelCol:
            scaled = np.insert(scaled,0,dataFrame['TerroristGroup'], axis=1)            
            return scaled
        # 1st col (label) out
        else:            
            return scaled

        
def preprocess_outliers(scaledDataFrame, process_type='univariate', outliersDict=None, cluster=None, eps=0.5, min_sample=4, returnAsDF=True, showInfo=False):
    '''
    * implements outlier pre-processing with options:
        * uses either 'univariate' (default), 'multivariate' or both ('chain')
        * multivariate will currently only work with 2 values    
    '''
    
    if not isinstance(scaledDataFrame, pd.DataFrame):
        print("WARNING! (preprocess_outliers) passed in dataFrame variable is not of type pd.Dataframe! Could cause error.")
    
    # univariate
    if process_type == 'univariate' or process_type == 'chain':
        if outliersDict != None:
            newDataFrame = outliers_univariate(scaledDataFrame, outliersDict, showInfo=showInfo)
            
            if process_type == 'univariate':
                return newDataFrame
        else:
            print("Error: (preprocess_outliers) outliersDict equals None!")
    
    # multivariate
    if process_type == 'multivariate' or process_type == 'chain':
        if cluster == None:
            print("Error: (preprocess_outliers) cluster equals None!")
    
        if process_type == 'multivariate':
            newDataFrame = outliers_multivariate(scaledDataFrame, cluster, the_eps=eps, minSamples=min_sample, returnAsDF=returnAsDF, showInfo=showInfo)
            return newDataFrame
        
        # chain
        elif process_type == 'chain':
            newDataFrame = outliers_multivariate(newDataFrame, cluster, the_eps=eps, minSamples=min_sample, returnAsDF=returnAsDF, showInfo=showInfo)
            return newDataFrame
    

def preprocess_imbalance(train_features, train_labels, showInfo=False):
    '''
    * utilises SMOTE to balance data
    * returns new train feature & label data
    '''
    
    if showInfo:
        print("\n- - - - - - - - - - - - - - - - - - - - - - - ")
        print("\n- - = = IMBALANCE HANDLING (SMOTE) = = - -")
        
        print("\nTrain labels before:", len(train_labels))
        unique, counts = np.unique(train_labels, return_counts=True)
        print("Distribution before:\n",np.asarray((unique, counts)).T)
    
    smote = SMOTE(random_state=RANDOM_SEED)
    t_features, t_labels = smote.fit_sample(train_features,train_labels)
    
    if showInfo:
        print("Train labels after:", len(t_labels))        
        unique, counts = np.unique(t_labels, return_counts=True)
        print("Distribution after:\n",np.asarray((unique, counts)).T)
        
    return t_features, t_labels

    
def preprocess_feature_selection(train_features, test_features, train_labels, dropFeatures=False, dropMinThreshold=0.01, showInfo=False):
    '''
    * tree based feature selection
    * will check feature importance on train_features / train_labels
    * optional: can drop features 
        * will then return new train_features & test_features arrays
    '''
    
    rndForestClass = RandomForestClassifier(n_estimators=250, random_state=RANDOM_SEED)
    rndForestClass.fit(train_features,train_labels)
    importances = rndForestClass.feature_importances_
    
    if showInfo:
        print("\n- - - - - - - - - - - - - - - - - - - - - - - ")
        print("- - = = FEATURE SELECTION - IMPORTANCE: = = - -")
        
        # bar chart containers
        objects = []
        values = []
        
        for index in range(len(train_features[0])):
            print(df_mostActiveGroups.columns[1:][index], " : ", round(importances[index],4)*100, "%" )
            objects.append(df_mostActiveGroups.columns[1:][index])
            values.append(importances[index]*100)        
        
        # bar chart
        visualise_bar_chart(objects, values, 'Importance in %', 'Feature Importance Graph', labelRotation=True)
        print("\n- - - - - - - - - - - - - - - -\n")
    
    if dropFeatures:
        colsDeleted=0
        for index in range(len(train_features[0])):
            if importances[index] < dropMinThreshold:                
                train_features = np.delete(train_features, index-colsDeleted, 1)
                test_features = np.delete(test_features, index-colsDeleted, 1)
                colsDeleted+=1
                if showInfo:                    
                    print("Deleted Col:", df_mostActiveGroups.columns[1:][index] ,"(Index:", index, ")")
        if showInfo:
            print("\nDeleted Columns ( Amount =", colsDeleted,"):")
            print("Number of feature columns remaining:", train_features.shape[1])                    
        
    return train_features, test_features



# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = MODEL ACCURACY & EVALUATION = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def inital_multi_model_accuracy_evaluation(listOfModels, train_features, train_labels, kFoldSplits=5, showFullBreakdown=False, showInfo=False):
    '''
    * an initial trial run to evaluate various model performance on data
    * goes to through passed list of ML models
        * uses default settings for all
    * uses kFold Cross Validation for each model
    * will also look for incorrect classifications and the corresponding indicies
    '''
    
    
    kf = KFold(n_splits=kFoldSplits, shuffle=True, random_state=RANDOM_SEED) # KFold with x folds (splits)
    allResults_mean = []
    allResults_std = []
    allResults_names = []
    
    if showInfo:
        print("\n- - - - - - - - - - - - - - - - - - - - - - - ")
        print("- - = = INITIAL MODEL EVALUATION = = - -")
    
    for classifier in listOfModels:
        modelResults = [] # reset placeholder        
        
        if showInfo:            
            print("\n- - - - - - - - - - - - - - - - - -")     
            print("Model Testing:", type(classifier).__name__)
            allResults_names.append(type(classifier).__name__) # required at the end
    
        for train_index, test_index in kf.split(train_features):
            
            # train - data & target class
            classifier.fit(train_features[train_index], train_labels[train_index])
            # get incorrect classifications for each iteration
            predictedResults = classifier.predict(train_features[test_index])
            # get index of incorrect predictions
            positions = np.where(predictedResults != train_labels[test_index])
            # accuracy score
            modelResults.append(accuracy_score(predictedResults, train_labels[test_index]))
            
            if showFullBreakdown:
                print("- - -")
                print("Incorrect Classifications:", predictedResults[predictedResults != train_labels[test_index]]) 
                print("Indicies with incorrect predictions:", test_index[positions[0]])
            
        if showInfo:            
            print("\nAll Results:", modelResults)
            print("Overal mean accuracy is:", round(np.mean(modelResults),6))
            print("Standart Deviation is:", round(np.std(modelResults),6))
            allResults_mean.append(np.mean(modelResults))
            allResults_std.append(np.std(modelResults))
    
    if showInfo:
        minY = min(allResults_mean) - 0.005  #define a min value for bar chart      
        visualise_bar_chart(allResults_names, allResults_mean, 'Accuracy', 'All Models Mean Accuracy', labelRotation=True, setYMinMax=True, y_min_max=[minY,1])
        visualise_bar_chart(allResults_names, allResults_std, 'Deviation', 'All Models Standard Deviation', labelRotation=True)
        
        
def grid_search_hyper_parameter_optimisation(classifier, parameterGrid, train_features, train_labels, crossFolds=10, useCPUCores=-1, showInfo=False):
    '''
    * performs grid search with given hyperparameters and classifier
    * returns best identified hyperparameter settings (model)
    '''
    
    if showInfo:        
        print("\n- - = = GRID SEARCH HYPER PARAMETER OPTIMISATION:", type(classifier).__name__ , "= = - -")
    
    clf = GridSearchCV(classifier, parameterGrid, cv=crossFolds, n_jobs=useCPUCores)
    clf.fit(train_features, train_labels)
    
    if showInfo:
        print("Best parameters found:")
        print(clf.best_params_, "with score of", clf.best_score_)
        
    return clf.best_estimator_
    
    
    
def performance_evaluation(model,train_features, test_features, train_labels, test_labels, showInfo=True):    
    '''
    * final evaluation using test data
    * outputs visual confusion matrix (showInfo)
    '''
    
    if showInfo:
        print("\n- - - - - - - - - - - - - - - - - - - - - - - ")
        print("- - = = FINAL EVALUATION:", type(model).__name__, "= = - -")
    
    model.fit(train_features,train_labels)
    test_labels_prediction = model.predict(test_features)
    
    print("\nAccuracy Testing:")
    accuracy = accuracy_score(test_labels, test_labels_prediction)
    print("Initial accuracy score with test set:", accuracy)
    
    if showInfo:
        confusionMatrix = confusion_matrix(test_labels,test_labels_prediction)
        #print("Confusion Matrix:\n", confusionMatrix)
        
        valueList = encodedCategoryTargetsDict.values()        
        df_cm = pd.DataFrame(confusionMatrix, 
                             index = [i[:10] for i in valueList],
                             columns = [i[:10] for i in valueList])
        plt.figure(figsize = (10,8))       
        sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')   
    
        print("Full report:\n", classification_report(test_labels,test_labels_prediction))
        

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - = = RUN = = - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

def experiment_one(showInfo=True, completeOutlierVisualisation=True , output_null_values=True):
    '''
    * runs first experiment
    * this is an initial 'lite' version that completes much faster to test performance only
        * less pre-processing
        * no hyper parameter optimisation
        * no final evaluation
    '''
    
    print("\n- - = = EXPERIMENT 1 = = - -")
    
    global df_mostActiveGroups
    global numberOfMostActiveGroups
    
    #init
    initialise_data()
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 1: manual data cleaning / select correct columns / nominal encoding
    df_mostActiveGroups = preprocess_initial(numberOfMostActiveGroups, showInfo=showInfo)
    if output_null_values:
        output_null_value_counts(df_mostActiveGroups, 'all')
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 2: impute the rest of the missing values
    df_mostActiveGroups = preprocess_impute(df_mostActiveGroups, process_type='iterative', showInfo=showInfo) # 'simple' or 'iterative'        
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 3: scaling    
    df_mostActiveGroups = preprocess_scaling(df_mostActiveGroups, scalingType='MinMax', returnAsDF=True, includeLabelCol=showInfo) # 'MinMax' or 'Standardized' (StandardScaler)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 4: outliers    
    if completeOutlierVisualisation:
        visualise_outliers_sections(df_mostActiveGroups, showInSections=True)    
    
    outliersDict={'Country':[0,0.9],'AttackType':[0,0.49],'CasualtiesDead':[0,0.9],'CasualtiesWounded':[0,0.9]}    
    
    # process_type options = 'univariate', ' multivariate', 'chain'
    df_mostActiveGroups = preprocess_outliers(df_mostActiveGroups, process_type='univariate', outliersDict=outliersDict,
                                              returnAsDF=True, showInfo=showInfo)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 5: split test/train data
    train_features, test_features, train_labels, test_labels = get_split_test_and_train_data(df_mostActiveGroups)
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 6: imbalance handling (training data)
    if output_null_value_counts:
        output_value_counts(df_mostActiveGroups,'TerroristGroup')    
    train_features, train_labels = preprocess_imbalance(train_features, train_labels, showInfo=showInfo)
        
    # - - - - - - - - - -
    # prepro 7: feature selection (PLEASE NOTE -> feature drop disabled here)
    train_features, test_features = preprocess_feature_selection(train_features, test_features, train_labels,
                                                                               dropFeatures=False, dropMinThreshold=0.01, showInfo=showInfo)
    

    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ML-Models / initial accuracy evaluation ("manual" k-fold cross validation)
    # USING ONLY train_features here and leaving test_features "untouched" for final evaluation
    
    listOfClassifiers = [DecisionTreeClassifier(),LinearSVC(),KNeighborsClassifier(),RandomForestClassifier(),
                         SGDClassifier(),GradientBoostingClassifier(), BaggingClassifier()]
    
    inital_multi_model_accuracy_evaluation(listOfClassifiers, train_features, train_labels, kFoldSplits=5, showFullBreakdown=False, showInfo=True)
    
   
   


def experiment_two(runHyperParamOpt=False, showInfo=True, completeOutlierVisualisation=True , output_null_values=True):
    '''
    * runs second experiment
    * more in depth pre-processing options
        * outlier detection using both univariate and multivariate (DBScan) = 'chain'
    '''
    
    print("\n- - = = EXPERIMENT 2 = = - -")
    
    global df_mostActiveGroups
    global numberOfMostActiveGroups
    
    #init
    initialise_data()
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 1: manual data cleaning / select correct columns / nominal encoding
    df_mostActiveGroups = preprocess_initial(numberOfMostActiveGroups, showInfo=showInfo)
    if output_null_values:
        output_null_value_counts(df_mostActiveGroups, 'all')
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 2: impute the rest of the missing values
    df_mostActiveGroups = preprocess_impute(df_mostActiveGroups, process_type='iterative', showInfo=showInfo) # 'simple' or 'iterative'        
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 3: scaling    
    df_mostActiveGroups = preprocess_scaling(df_mostActiveGroups, scalingType='Standardized', returnAsDF=True, includeLabelCol=showInfo) # 'MinMax' or 'Standardized' (StandardScaler)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 4: outliers    
    if completeOutlierVisualisation:
        visualise_outliers_sections(df_mostActiveGroups, showInSections=True)    
    
    outliersDict={'Country':[-2,3],'AttackType':[-2,2],'CasualtiesDead':[-2,9],'CasualtiesWounded':[-2,30]}
    cluster = ['Latitude','Longitude']
    
    # process_type options = 'univariate', ' multivariate', 'chain'
    df_mostActiveGroups = preprocess_outliers(df_mostActiveGroups, process_type='chain', outliersDict=outliersDict, 
                                             cluster=cluster, eps=0.3, min_sample=10, returnAsDF=True, showInfo=showInfo)
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 5: split test/train data
    train_features, test_features, train_labels, test_labels = get_split_test_and_train_data(df_mostActiveGroups)
        
    # - - - - - - - - - - - - - - - - - - - - - - - - - 
    # prepro 6: imbalance handling (training data)
    if output_null_value_counts:
        output_value_counts(df_mostActiveGroups,'TerroristGroup')    
    train_features, train_labels = preprocess_imbalance(train_features, train_labels, showInfo=showInfo)
        
    # - - - - - - - - - -
    # prepro 7: feature selection
    train_features, test_features = preprocess_feature_selection(train_features, test_features, train_labels,
                                                                               dropFeatures=True, dropMinThreshold=0.01, showInfo=showInfo)

    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # ML-Models / initial accuracy evaluation ("manual" k-fold cross validation)
    # USING ONLY train_features here and leaving test_features "untouched" for final evaluation
    
    listOfClassifiers = [DecisionTreeClassifier(),LinearSVC(),KNeighborsClassifier(),RandomForestClassifier(),
                         SGDClassifier(),GradientBoostingClassifier(), BaggingClassifier()]
    
    inital_multi_model_accuracy_evaluation(listOfClassifiers, train_features, train_labels, kFoldSplits=5, showFullBreakdown=False, showInfo=True)
    
   
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Hyper Parameter Optimisation Of Selected Models
    
    if runHyperParamOpt:
    
        classifier = KNeighborsClassifier()
        parameterGrid = [{'n_neighbors':list(range(1,100)), 'p':[1,2,3,4,5,6,7,8,9]}]
        result = grid_search_hyper_parameter_optimisation(classifier, parameterGrid, train_features, train_labels, showInfo=True)
        print(result)
        
        
        classifier = DecisionTreeClassifier()
        parameterGrid = [{'criterion':['gini','entropy'],'splitter':['best','random'],'min_samples_split':list(range(2,40))}]
        result = grid_search_hyper_parameter_optimisation(classifier, parameterGrid, train_features, train_labels, showInfo=True)
        print(result)
        
        
        classifier = RandomForestClassifier()
        parameterGrid = [{'n_estimators':list(range(1,200)),'criterion':['gini','entropy'],'min_samples_split':list(range(2,40))}]
        result = grid_search_hyper_parameter_optimisation(classifier, parameterGrid, train_features, train_labels, showInfo=True)
        print(result)
    
    
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    # Final Models With Evaluation    
    model = KNeighborsClassifier(n_neighbors=10, p=2, metric='minkowski', weights='uniform')
    performance_evaluation(model, train_features, test_features, train_labels, test_labels, showInfo=True)
    
    model = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_split=2)
    performance_evaluation(model, train_features, test_features, train_labels, test_labels, showInfo=True)
    
    model = DecisionTreeClassifier(criterion='gini', min_samples_split=3, splitter='best')
    performance_evaluation(model, train_features, test_features, train_labels, test_labels, showInfo=True)



def run():
    
    # experiment one
    #experiment_one(showInfo=True, completeOutlierVisualisation=True, output_null_values=True)
    
    # experiment two (disabled hyper parameter opt -> takes ages)
    experiment_two(runHyperParamOpt=False, showInfo=True, completeOutlierVisualisation=True, output_null_values=True)    

# To run the analysis, uncomment the line below:
# run()

