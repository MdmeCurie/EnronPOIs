# %load poi_id.py
#!/usr/bin/python

from __future__ import division
import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data, test_classifier

import numpy as np
import pandas as pd

from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.feature_selection import SelectPercentile, f_classif, chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler 
from sklearn.preprocessing import Imputer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn import tree, neighbors
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import recall_score, accuracy_score, precision_score

### Functions for Data Extraction and train_test_split ###
###
### Extract features and labels from dataset for local testing
def feature_extraction(mydata_dict, features_lineup):
    data = featureFormat(mydata_dict, features_lineup, sort_keys = True)
    labels, features = targetFeatureSplit(data)
    return labels,features

### Split sets for cross validation train/test data 
def split_sets(features, labels, test_amt, r_state): 
    from sklearn.cross_validation import train_test_split
    f_train, f_test, l_train, l_test =     train_test_split(features, labels, test_size=test_amt, random_state=r_state)
    return f_train, f_test, l_train, l_test  
###
###########################################################

### Task 1: Select Features to Use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi". 
##### 'email_address' fails featureFormat() as it is string not float, all other features imported 

features_list = ['poi', 'salary', 'bonus','deferral_payments','total_payments',  
                 'exercised_stock_options','restricted_stock','total_stock_value',
                 'expenses','deferred_income',
                 'long_term_incentive', 'other',
                 'restricted_stock_deferred', 'loan_advances', 'director_fees', 
                 'to_messages', 'shared_receipt_with_poi','from_messages',      
                 'from_this_person_to_poi', 'from_poi_to_this_person'             
                ] 

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

num_entries = len(data_dict)
print ("There are %s entries in the original data_file"%(num_entries))


        
### Task 2: Remove outliers - remove TOTAL, Lockhardt Eugene E (all NaNs), and The Agency in the Park    

print ("Rows with excessive empty values >=18:")    
for namen in data_dict:
    count = 0
    for things, values in data_dict[namen].items():
        if values == 'NaN':
            count +=1
    if count >=18:
        print namen, count

outlier = data_dict.pop('LOCKHART EUGENE E')
outlier = data_dict.pop('THE TRAVEL AGENCY IN THE PARK')

## Remove Total as Statistical outlier found abnormal hist/distribution, found upon examination as pandas dataframe
outlier = data_dict.pop('TOTAL')                       


### Negative outliers found in deferred_income and restricted stock deferred
### Entries for 'BELFER ROBERT' & 'BHATANGAR SANJAY' corrected as confirmed by enron61702insiderpay.pdf

data_dict['BELFER ROBERT']['deferred_income']  = -102500
data_dict['BELFER ROBERT']['deferral_payments']= 'NaN'
data_dict['BELFER ROBERT']['expenses'] = 3285
data_dict['BELFER ROBERT']['director_fees'] = 102500 
data_dict['BELFER ROBERT']['total_payments'] = 3285
data_dict['BELFER ROBERT']['exercised_stock_options']= 'NaN'
data_dict['BELFER ROBERT']['restricted_stock']= 44093
data_dict['BELFER ROBERT']['restricted_stock_deferred']= -44093
data_dict['BELFER ROBERT']['total_stock_value'] = 'NaN'

data_dict['BHATNAGAR SANJAY']['other']= 'NaN' 
data_dict['BHATNAGAR SANJAY']['expenses']= 137864
data_dict['BHATNAGAR SANJAY']['director_fees']= 'NaN'
data_dict['BHATNAGAR SANJAY']['total_payments']= 137864
data_dict['BHATNAGAR SANJAY']['exercised_stock_options']= 15456290 
data_dict['BHATNAGAR SANJAY']['restricted_stock']= 2604490
data_dict['BHATNAGAR SANJAY']['restricted_stock_deferred']= -2604490
data_dict['BHATNAGAR SANJAY']['total_stock_value']= 15456290

### Task 3: Create new feature(s)
### created dataframe from dict with features as cols and names as index
### https://stackoverflow.com/questions/13575090/construct-pandas-dataframe-from-items-in-nested-dictionary

df_data = pd.DataFrame.from_dict({(i): data_dict[i]
                                  for i in data_dict.keys()},orient='index')

## Outliers values with statistically high values - TOTAL discovered - removed above
#print df_data['bonus'].describe()
#df_data['bonus'].hist()
#df_data['salary'].hist()
#print df_data[df_data['bonus'] >0.8e8]

num_pois = df_data[df_data['poi']==True].sum()
print("There are %s entries classified as POIs"%(num_pois))
### List of Features that are numbers
numbers = list(df_data)  
numbers.remove('email_address') 
numbers.remove('poi')

### Ensure pd.dataframe values are 'float' for mathematical operations
for trait in [numbers]:
    df_data[trait] = df_data[trait].astype('float')

### New features:
 
df_data['take_home'] = df_data['salary'] + df_data['bonus']
df_data['percent_exercised'] = df_data['exercised_stock_options']/df_data['total_stock_value']
df_data['response_rate'] = df_data['from_messages']/df_data['to_messages']
df_data['poi_response'] = df_data['from_this_person_to_poi']/df_data['from_poi_to_this_person']
df_data['delta_response'] = df_data['from_this_person_to_poi']-df_data['from_poi_to_this_person']

### Replace Inf in poi_response with .max() + 10 to put at top of scale
m = df_data.loc[df_data['poi_response'] != np.inf, 'poi_response'].max() + 10
df_data['poi_response'].replace(np.inf,m,inplace=True)

new_features=['take_home', 'percent_exercised', 'response_rate', 'poi_response', 'delta_response']
numbers = numbers + new_features 

df_data.dropna(0,'all')  ##Drop rows with all empty features - technically unnecessary as no rows completely empty

print ("Shape of new dataframe:", df_data.shape)

### Imputation Data Prepocessing (replace NaNs)

### Manual Imputation
imputed_mean = df_data.copy()   
imputed_median = df_data.copy()
imputed_zero = df_data.copy()
for col in numbers:
    ave =  imputed_mean[col].mean()
    imputed_mean[col] = imputed_mean[col].replace(np.nan, ave)
    imputed_median[col] = imputed_median[col].replace(np.nan, ave)
    imputed_zero[col] = imputed_zero[col].replace(np.nan, 0)

## How many entries are there per Feature??    
print ("Number of NaNs per Feature of 143 ")
print df_data.isnull().sum(axis=0)  #https://stackoverflow.com/questions/30059260/python-pandas-counting-the-number-of-missing-nan-in-each-row

## Feature values, numbers and negative values
print ("Breakdown of Feature Value Entries:")

for items in features_list:
    value_exists = 0
    pos_poi = 0
    neg_poi = 0 
    for names in data_dict:
        if data_dict[names][items] != "NaN":
            value_exists = value_exists + 1
            if data_dict[names]['poi'] == 1:
                pos_poi = pos_poi+1
            else:    neg_poi = neg_poi+1
    print '%25s %8d entries %8d POIs %8d non-POIs %8.1f%% POIs'%(items, value_exists, pos_poi, neg_poi,(pos_poi/value_exists)*100.0)

print ("Features with Negative Values:")    
for items in features_list:
    neg_value = 0 
    for names in data_dict:
        if data_dict[names][items] != "NaN" and data_dict[names][items] <0:
            neg_value = neg_value + 1
    if neg_value >0:
        print '%25s %8d'%(items, neg_value)
        

my_features = ['poi','salary', 'bonus','total_payments',  
               'exercised_stock_options','restricted_stock','total_stock_value',
               'expenses','deferred_income',
               'long_term_incentive', 'other',
               'to_messages', 'shared_receipt_with_poi','from_messages',      
               'from_this_person_to_poi', 'from_poi_to_this_person',
              # 'restricted_stock_deferred','loan_advances','director_fees',    ## Remove! - too few values
               'deferral_payments',                                            ## Remove? - 73% NaNs
               'take_home', 'percent_exercised', 'response_rate', 'poi_response', 'delta_response' ## New features
              ]

### Store new features and corrections to my_dataset dictionary for easy export below.
my_dataset = df_data.to_dict(orient='index')               ## no pre-imputation, use imputation in estimatore pipeline


######################
## Rank Features with various Feature Selection Methods
######################

labels, features_raw = feature_extraction(my_dataset, my_features)

imp_mean = Imputer(missing_values=np.nan, strategy='median')
features= imp_mean.fit_transform(features_raw)

print ("########  Feature Ranking  ########")
## VarianceThreshold object to rank feature variances
thresholder = VarianceThreshold()
high_variance = thresholder.fit(features)
## List Features with Ranked variances (descending)
t_vars = thresholder.variances_
#t_vars_sort = np.argsort(thresholder.variances_) ## ascending
t_vars_sort = (-thresholder.variances_).argsort()##https://stackoverflow.com/questions/16486252/is-it-possible-to-use-argsort-in-descending-order

print "########  VarianceThreshold:"
for i in t_vars_sort:
    print '%23s   %10.2e'%(my_features[i+1],t_vars[i])

######################
## SelectKBest Ranking Featues, target(labels), select k features
kbest = SelectKBest(f_regression).fit(features, labels)
print "########  SelectKBest:"
for f in (-kbest.scores_).argsort():
    print '%23s   %8.2e %10.2e'%(my_features[f+1], kbest.scores_[f], kbest.pvalues_[f])
    
#####################   Removed - Identical Scores to SelectKBest 
## Select Percentile, default selection function: the 10% most significant features
#selectp = SelectPercentile(f_classif, percentile=10)
#selectp.fit(features, labels)
#print "##########  SelectPercentile:"
#for f in (-scores).argsort():
#    print '%23s   %8.2e %10.2e'%(my_features[f+1], selectp.scores_[f], selectp.pvalues_[f])
  

#####################      
## Feature Importance with Extra Trees Classifier
## https://machinelearningmastery.com/feature-selection-machine-learning-python/
model = ExtraTreesClassifier(n_estimators = 1000).fit(features,labels)
feature_scores = model.feature_importances_

names = list(my_features)
names.pop(0)

for score, fname in sorted(zip(feature_scores, names), reverse=True):
     print '%23s   %8.3f'%(fname, score)


### Highly correlated features
## Review Features: correlation matrix pandas, boxplot, statistics
## https://stackoverflow.com/questions/17778394/list-highest-correlation-pairs-from-a-large-correlation-matrix-in-pandas

#print imputed_median.describe()
#pd.DataFrame.hist(imputed_median)

s= df_data.corr()
s_order = s.unstack().sort_values(ascending=False)
dup= 0
eliminated = ['poi','restricted_stock_deferred','loan_advances','director_fees']
for key, value in s_order.iteritems():
    if key[0] in eliminated or key[1] in eliminated:
        continue
    if dup == value:
        continue
    if value > 0.8 and value !=1:
        print key, value
        dup = value

### Top Features Sorted as Ranked for Median Imputation by Select KBest/Percentile
my_features = ['poi', 'exercised_stock_options', 'total_stock_value', 
               #'bonus', 
               'take_home', 
               'salary', 'deferred_income',
               'total_payments',
               'long_term_incentive', 'restricted_stock',
               'shared_receipt_with_poi',
               'from_poi_to_this_person',
               #'other',
               'from_this_person_to_poi', #'expenses',
               #'to_messages',
               #'response_rate' , #'delta_response',
               #'from_messages'#,  'poi_response', 'percent_exercised'
              ]
labels, features = feature_extraction(my_dataset, my_features)

print ">>> Selected Features: ", len(my_features)-1, my_features

###########################
## Feature Union Pipeline for feature selection/reduction with PCA()/SelectKBest optimization
## http://scikit-learn.org/stable/auto_examples/compose/plot_compare_reduction.html

labels, features = feature_extraction(my_dataset, my_features)
sss = StratifiedShuffleSplit(labels, 50, test_size = 0.3, random_state = 42)

pca = PCA()
selectk = SelectKBest()
united_features = FeatureUnion([("pca", pca),("select", selectk)])

pipe_features = Pipeline([("impute",Imputer(strategy = 'median')),
                          ("scale", MinMaxScaler()),
                          ("features", united_features),
                          ('classify', GaussianNB())
                         ])

## Grid Search Parameters for Feature Selection/Reduction
K_FEATURES_OPTIONS = [1,2,3]
N_COMPS = [3,4,5]       
S_FUNC = [f_regression, f_classif, chi2]
              
param_grid_f = [{'features__pca__n_components': N_COMPS,
                 'features__pca__whiten': [True, False],
                 'features__select__k': K_FEATURES_OPTIONS,
                 'features__select__score_func': S_FUNC
                }]

grid_feature = GridSearchCV(pipe_features, param_grid=param_grid_f, cv=sss)
grid_feature.fit(features, labels)

print "Pipeline Feature Union PCA()/SelectKBest Best Score/Parameters:"
#mean_scores = np.array(grid.cv_results_['mean_test_score'])
#print mean_scores
print grid_feature.best_score_
print grid_feature.best_params_

## Test Feature Selection with Prediction Scores
clf_feature_selection = grid_feature.best_estimator_

print("<<<Tester Results>>>")
test_classifier(clf_feature_selection, my_dataset, my_features)


############
## Pipeline for Classifier Reviews: limited parameter tuning of different classifiers
############

labels, features = feature_extraction(my_dataset, my_features)
sss = StratifiedShuffleSplit(labels, 50, test_size = 0.3, random_state = 42)

## Combine PCA and Univariate Selection using Parameters Determined Previously in Feature Union ##
pca = PCA(n_components = 3, whiten=True)
selectk = SelectKBest(k=2, score_func = chi2)
united_features = FeatureUnion([("pca", pca),("select", selectk)])

pipe_cr = Pipeline([('impute',Imputer(strategy = 'median')),
                    ('scale', MinMaxScaler()),
                    ('features', united_features),
                    ('classify', SVC())
                   ])

##Classifier Parameters
C_OPTIONS = [1, 25]                #SVC
SPLITS = [2, 30]                   #Decision Tree
WEIGHTS = ['distance', 'uniform']  #K Nearest Neighbors
NACHBARN = [5, 30]                 #K Nearest Neighbors
ESTIMATES = [10, 50]               #Ada Boost/Random Forest

param_grid_cr = [
    {
        'classify': [SVC()],
        'classify__C': C_OPTIONS 
    },
    #{ 
    #    'classify': [GaussianNB()]   #Used as default in Feature Union Determination
    #},
    {
        'classify': [tree.DecisionTreeClassifier()],
        'classify__min_samples_split': SPLITS 
    },
    {
        'classify': [neighbors.KNeighborsClassifier()],
        'classify__n_neighbors': NACHBARN,
        'classify__weights': WEIGHTS
    },
    {
        'classify': [RandomForestClassifier()],
        'classify__n_estimators': ESTIMATES
    },
    {
        'classify': [AdaBoostClassifier()],
        'classify__n_estimators': ESTIMATES
    }
]


grid = GridSearchCV(pipe_cr, param_grid=param_grid_cr, cv= sss)
grid.fit(features, labels)

mean_scores = np.array(grid.cv_results_['mean_test_score'])

print "Pipeline Scores of Various Classifiers"
print mean_scores
print "BEST ESTIMATOR SCORE:", grid.best_score_
print "BEST PARAMETERS:", grid.best_params_

clf_classifier_review = grid.best_estimator_

print
print("<<<Tester Results Classifier Review>>>")
test_classifier(clf_classifier_review, my_dataset, my_features)

#########################################################
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
##http://scikit-learn.org/0.16/auto_examples/model_selection/grid_search_digits.html#example-model-selection-grid-search-digits-py
## 



## Load/Prepare dataset
labels, features = feature_extraction(my_dataset, my_features)
sss = StratifiedShuffleSplit(labels, 75, test_size = 0.3, random_state = 42)

pca = PCA(n_components = 3, whiten=True)
selectk = SelectKBest(k = 2, score_func = chi2)
united_features = FeatureUnion([("pca", pca),("select", selectk)])


pipe3 = Pipeline([("impute",Imputer(strategy = 'median')),
                  ("scale", MinMaxScaler()),
                  ("features", united_features),
                  ('classify',GaussianNB())
                  ])

## Set tuning parameters for cross-validation
tune_GNBparameters = [{'classify__priors': [None]}]

tune_KNNparameters = [{'classify__n_neighbors': [11,14],
                       'classify__weights':['uniform', 'distance'],
                       'classify__algorithm': ['auto'],
                       'classify__leaf_size':[5,8],
                       'classify__p':[2,1]
                      }]

cees = [1,20,50] # C values
tune_SVCparameters = [{'classify__kernel': ['rbf'],
                       'classify__C': cees},
                       #'classify__gamma': [1e-5, 1e-6, 'auto']},
                      {'classify__kernel': ['linear'],
                       'classify__C': cees},
                      {'classify__kernel': ['poly'],   
                       'classify__C': cees, 
                       'classify__degree':[3, 4, 5]}
                       #'classify__gamma': [1e-5, 1e-6, 'auto']}
                     ]

tune_RFparameters = [{'classify__n_estimators': [5,10,15],
                      'classify__criterion': ['gini','entropy'],
                      'classify__min_samples_split': [2,3,5],
#                     'clasify__min_samples_leaf': [1,2,3],
                      'classify__max_features': ['auto', 1, 0.5]
                     }]

tune_ADAparameters = [{'features__select__k': [2,3],
                       'classify__n_estimators': [12, 50],
                       'classify__algorithm': ['SAMME.R', 'SAMME'],
                       'classify__learning_rate': [1, 0.4]}
                     ]

print(">>>>>Chosen Classifier GaussionNB()<<<<<\n")
gs = GridSearchCV(pipe3, tune_GNBparameters, cv=sss)
gs.fit(features, labels)
clf = gs.best_estimator_
print("Best score:",gs.best_score_)
print("Best parameters:")
print(gs.best_params_)

print("<<<Tester Scores on Final Estimator>>>")
test_classifier(clf, my_dataset, my_features)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

features_list = my_features
dump_classifier_and_data(clf, my_dataset, features_list)
