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
import sys

#CHANGE TO YOUR OWN SETTING
PATH_OUT = "file.csv"
FEATURE_CSV_FOLDER='path_features'


if __name__ == "__main__":
    args = sys.argv
    print(args)
    seed = int(args[1])
    num_features = int(args[2])
    print(seed)

    def get_files(feature_name,path='path'):
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

    
    
    feature_names  = []
    #get all files in FEATURE_CSV_FOLDER that are .csv and remove the .csv from the name
    for r, d, f in os.walk(FEATURE_CSV_FOLDER):
        for file in f:
            if '.csv' in file:
                feature_names.append(file[0:-4])
    feature_names = [x for x in feature_names ]
    feature_names = set(feature_names)
    feature_names = list(feature_names)
    print(feature_names)
    print(len(feature_names))



    def get_selector_clf_scaler(num_features):
        names_selectors = [
            'chi2',
            'f_classif',
            'mutual_info_classif']

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
            'Normalizer'
        ]
        #get a random selector
        selector_name = np.random.choice(names_selectors)
        #get a random classifier
        classifier = np.random.choice(classifiers)
        classifier_name = classifier
        #get a random scaler
        scaler = np.random.choice(scalers)
        scaler_name = scaler
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


        if selector_name == 'chi2':
            selector = SelectKBest(chi2, k=num_features)
        elif selector_name == 'f_classif':
            selector = SelectKBest(f_classif, k=num_features)
        elif selector_name == 'mutual_info_classif':
            selector = SelectKBest(mutual_info_classif, k=num_features)


        if classifier == 'RandomForestClassifier':
            clf = RandomForestClassifier( random_state=42,)
        elif classifier == 'SVC':
            clf = SVC( random_state=42, probability=True)
        elif classifier == 'GradientBoostingClassifier':
            clf = GradientBoostingClassifier( random_state=42)
        elif classifier == 'KNeighborsClassifier':
            clf = KNeighborsClassifier( )
        elif classifier == 'StackedClassifier':
            meta_classifier = LogisticRegression()
            base_classifiers = [
            ('rf', RandomForestClassifier()),
            ('knn', KNeighborsClassifier()),
            ('svm', SVC())
            ]
            clf = StackingClassifier(estimators=base_classifiers, final_estimator=meta_classifier)
        return selector,clf,scaler,selector_name,classifier_name,scaler_name
 
    
    

    
    #permute the features names
    np.random.seed(seed)
    np.random.shuffle(feature_names)
    # Initialize the scaler, selector and classifier
    
    selector,clf,scaler,selector_name,classifier_name,scaler_name = get_selector_clf_scaler(num_features)
    print(feature_names)
    X_preprocessed = []
    y_preprocessed = []
    #train a classifier for each feature
    for feature_name in feature_names:
            print(feature_name)
            files = get_files(feature_name,path=FEATURE_CSV_FOLDER)
            X,y = stack_features(files)
            y = y.astype(str)
            X = X.astype(np.float32)
            num_attributes = X.shape[1]
            X = scaler.fit_transform(X)
            X = scale01(X)
            if num_attributes > num_features:     
                    X = selector.fit_transform(X, y)
            # add the preprocessed data to the list
            X_preprocessed.append(X)
            y_preprocessed.append(y)

    print(len(X_preprocessed))
    print(X_preprocessed[0].shape)

    #first step is to tran a classifier for each feature and save the scores 
    def classify(X,y,clf,feature_names):
        cv_results = cross_validate(clf, X, y, cv=10, scoring=['f1_macro'],n_jobs=-1)
        keys=sorted(cv_results.keys())
        keys_with_std = []
        for key in keys:
            keys_with_std.append(key)
            keys_with_std.append(key+'_std')
        #add the feature name to the results
        keys_with_std.append('feature')
        #remove the time keys
        keys_with_std = [k for k in keys_with_std if 'time' not in k]
        for key in keys_with_std:
            if '_std' in key:
                cv_results[key] = cv_results[key[:-4]].std()
        for key in keys:
            if '_std' not in key:
                cv_results[key] = cv_results[key].mean()
        not_avg = ['feature']
        cv_results['feature'] = feature_names
        cv_results = {k:v for k,v in cv_results.items() if 'time' not in k}   
        #round the results to 3 decimal places if they are not the feature name
        cv_results = {k:round(v,3) if k not in not_avg else v for k,v in cv_results.items()}
        return cv_results


    cv_results_list = []
    for i in range(len(X_preprocessed)):
        X = X_preprocessed[i]
        y = y_preprocessed[i]
        feature_name = feature_names[i]
        print(feature_name)
        cv_results = classify(X,y,clf,feature_name)
        cv_results_list.append(cv_results)
    print(X_preprocessed[0].shape)

    def iterative_step(X_values,y,names,scores,clf):
        current_values = []
        current_names = []
        new_values = []
        new_scores = []
        new_names = []
        for i in range(0,len(X_values),2):
            #print(i)
                
            if len(X_values) % 2 == 0 and i != len(X_values)-1:
                X1 = X_values[i]
                X2 = X_values[i+1]
                X = np.concatenate((X1,X2),axis=1)
                X = scaler.fit_transform(X)
                X = scale01(X)
                if X.shape[1]> num_features:
                    X = selector.fit_transform(X, y)
                current_values.append(X)
                current_names.append(names[i]+'_'+names[i+1])
            else:
                X = X_values[i]
                current_values.append(X)
                current_names.append(names[i])
        for i in range(len(current_values)):
            #train a classifier on the merged data
            X = current_values[i]
            feature_name = current_names[i] 
            cv_results = classify(X,y,clf,feature_name)
            score = cv_results['test_f1_macro']
            score_l = scores[i*2]['test_f1_macro']
            right_index = i*2+1
            if len(scores) <= right_index:
                score_r = score_l
                print(cv_results['feature'],scores[i*2]['feature'],scores[i*2]['feature'])
            else:
                score_r = scores[i*2+1]['test_f1_macro']
                print(cv_results['feature'],scores[i*2]['feature'],scores[i*2+1]['feature'])

                
            
            print(score,score_l,score_r)
            
            #determine the max score
            if score > score_l and score > score_r:
                #keep the merged feature
                print("Keep: conc")
                new_scores.append(cv_results)
                new_values.append(X)
                new_names.append(feature_name)
            elif score_l >= score_r:
                print("Keep: l")
                new_scores.append(scores[i*2])
                new_values.append(X_preprocessed[i*2])
                new_names.append(names[i*2])
            else:
                print("Keep: r")
                new_scores.append(scores[i*2+1])
                new_values.append(X_preprocessed[i*2+1])
                new_names.append(names[i*2+1])
            
        return new_values,new_names,new_scores

    y = y_preprocessed[0]
    print("current values: ", cv_results_list)
    print("current names: ", feature_names)
    print(len(X_preprocessed))

    while(True):
        X_preprocessed,feature_names,cv_results_list = iterative_step(X_preprocessed,y,feature_names,cv_results_list,clf)
        print("current values: ", cv_results_list)
        print("current names: ", feature_names)
        if len(X_preprocessed) == 1:
            df = pd.DataFrame(cv_results_list[0], index=[0])
            #add seed to the results, selector,clf,scaler
            df['seed'] = seed 
            df['selector'] = selector_name
            df['classifier'] = classifier_name
            df['scaler'] = scaler_name
            if os.path.isfile(PATH_OUT):
                df.to_csv(PATH_OUT, mode='a', header=False,index=False)
            else:
                df.to_csv(PATH_OUT, mode='a', header=True,index=False)
            break         


        
                