import csv
import os
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn_hierarchical_classification.classifier import HierarchicalClassifier
from sklearn_hierarchical_classification.constants import ROOT

from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import  SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.preprocessing import (
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
    minmax_scale,
)


#CHANGE TO YOUR OWN SETTING
PATH_OUT = "file.csv"
FEATURE_CSV_FOLDER='path_features'

def get_files(feature_name,path='Features'):
    files = []
    for r, d, f in os.walk(path):
        for file in f:
            if feature_name in file and '.csv' in file:
                files.append(os.path.join(r, file))
    return files
def scale01(X):
    # Normalize the data
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X
def stack_features(files):
    X = []
    Y = []
    for file in files:
        with open(file, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                X.append(row[3:])
                Y.append(file.split('\\')[-2])

    X = np.array(X).astype(np.float64)
    Y = np.array(Y)
    #X = scale01(X)
    return X,Y


folder = FEATURE_CSV_FOLDER
feature_names  = []
#get all files in folder that are .csv and remove the .csv from the name
for r, d, f in os.walk(folder):
    for file in f:
        if '.csv' in file:
            feature_names.append(file[0:-4])
feature_names = [x for x in feature_names ]
feature_names = set(feature_names)
feature_names = list(feature_names)
print(feature_names)


class_hierarchy = {
        ROOT: ["CLS_1", "CLS_2", 'CLS_3'],
        "CLS_1": ["CLS_1NonTarget", "CLS_1Target"],
        "CLS_2": ["CLS_2NonTarget", "CLS_2Target"],
        "CLS_3": ["CLS_3NonTarget", "CLS_3Target"]
    }
folds = [
    #5,
    10]
num_features = [10,25,50,100,None]
names_selectors = [
    'chi2',
    'f_classif',
    'mutual_info_classif']
scoring = ['precision_macro','precision_micro', 'recall_macro','recall_micro','f1_macro','f1_micro','accuracy']
classifiers = [
    'RandomForestClassifier',
    'KNeighborsClassifier',
    'SVC',
    'StackedClassifier',
    'GradientBoostingClassifier',
    ]

scalers = [
    'StandardScaler',
    'MinMaxScaler',
    'MaxAbsScaler',
    'RobustScaler',
    'QuantileTransformerGaussian',
    'PowerTransformer',
    'QuantileTransformerUniform',
    'Normalizer']

print("Number of combinations: ",2*len(folds)*len(num_features)*len(names_selectors)*len(classifiers)*len(scalers)*32)

base_classifiers = [
    ('rf', RandomForestClassifier()),
    #('et', ExtraTreesClassifier()),
    #('lr', LogisticRegression()),
    ('knn', KNeighborsClassifier()),
    ('svm', SVC())
]
# Initialize the meta-classifier
meta_classifier = LogisticRegression()
def classify(X,y,num_features,selector_name,folds,csv_name,feature_name,class_hierarchy,scoring,classifier,scaler):
    if scaler == 'StandardScaler':
        scaler = StandardScaler()
    elif scaler == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaler == 'MaxAbsScaler':
        scaler = MaxAbsScaler()
    elif scaler == 'RobustScaler':
        scaler = RobustScaler()
    elif scaler == 'QuantileTransformerGaussian':
        scaler = QuantileTransformer(output_distribution='normal')
    elif scaler == 'PowerTransformer':
        scaler = PowerTransformer()
    elif scaler == 'QuantileTransformerUniform':
        scaler = QuantileTransformer(output_distribution='uniform')
    elif scaler == 'Normalizer':
        scaler = Normalizer()
    #print lowest and highest value of the data before scaling
    #print(np.min(X),np.max(X))
    X = scaler.fit_transform(X)
    X = scale01(X)
    
    num_attributes = X.shape[1]
    original_num_attributes = num_attributes
    if num_features !=None and num_attributes <= num_features:
        return
    if num_features != None:
        if num_attributes > num_features:
            if selector_name == 'chi2':
                selector = SelectKBest(chi2, k=num_features)
            elif selector_name == 'f_classif':
                selector = SelectKBest(f_classif, k=num_features)
            elif selector_name == 'mutual_info_classif':
                selector = SelectKBest(mutual_info_classif, k=num_features)
            X = selector.fit_transform(X, y)
            num_attributes = X.shape[1]
    if classifier == 'RandomForestClassifier':
        base_estimator = RandomForestClassifier( random_state=42,)
    elif classifier == 'SVC':
        base_estimator = SVC( random_state=42, probability=True)
    elif classifier == 'GradientBoostingClassifier':
        base_estimator = GradientBoostingClassifier( random_state=42)
    elif classifier == 'KNeighborsClassifier':
        base_estimator = KNeighborsClassifier( )
    elif classifier == 'StackedClassifier':
        base_estimator = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)
    

    clf = base_estimator
    cv_results = cross_validate(clf, X, y, cv=folds, scoring=scoring,n_jobs=-1)
    #print(cv_results)
    keys=sorted(cv_results.keys())
    print(keys)
    keys_with_std = []
    for key in keys:
        keys_with_std.append(key)
        keys_with_std.append(key+'_std')
    #add the feature name to the results
    keys_with_std.append('feature')
    keys_with_std.append('number_fetures')
    keys_with_std.append('original_number_fetures')
    keys_with_std.append('selector_name')   
    keys_with_std.append('scaler')
    keys_with_std.append('classifier')
    
    #remove the time keys
    keys_with_std = [k for k in keys_with_std if 'time' not in k]
    for key in keys_with_std:
        if '_std' in key:
            cv_results[key] = cv_results[key[:-4]].std()
    for key in keys:
        if '_std' not in key:
            cv_results[key] = cv_results[key].mean()
    not_avg = ['feature','selector_name','number_fetures','original_number_fetures','selector_name','scaler','classifier']
    cv_results['feature'] = feature_name
    cv_results['number_fetures'] = num_attributes
    cv_results['original_number_fetures'] = original_num_attributes
    cv_results['selector_name'] = selector_name
    cv_results['scaler'] = scaler
    cv_results['classifier'] = classifier

    cv_results = {k:v for k,v in cv_results.items() if 'time' not in k}   
    #round the results to 3 decimal places if they are not the feature name
    cv_results = {k:round(v,3) if k not in not_avg else v for k,v in cv_results.items()}
    #use pandas to write the results to a csv file without the index
    df = pd.DataFrame(cv_results, index=[0])
    if os.path.isfile(csv_name):
        df.to_csv(csv_name, mode='a', header=False,index=False)
    else:
        df.to_csv(csv_name,index=False)



for classifier in classifiers:
    for feature_name in feature_names:
        files = get_files(feature_name,path=folder)
        X,y = stack_features(files)
        y = y.astype(str)
        X = X.astype(np.float32)
        for scaler in scalers:
            for selector_name in names_selectors:
                for num_feature in num_features:
                    for fold in folds:
                        csv_name = PATH_OUT
                        print(X.shape)
                        classify(X,y,num_feature,selector_name,fold,csv_name,feature_name,class_hierarchy,scoring,classifier,scaler)