"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""

import pickle
from sklearn.tree import DecisionTreeClassifier
from enigmx.utils import dataPreparation_forTuning
from enigmx.hyperparameter_tuning import clfHyperFit

#definition of model for tuning
model_for_tuning = DecisionTreeClassifier()

#definition of parameters for tuning based on model
params_for_tuning = {
    'max_leaf_nodes': list(range(2, 20)), 
    'min_samples_split': [2, 3, 4]
    }

#csv_path: where your .csv data is allocated
csv_path = "D:/data_split_stacked/DATA_TUNING_EXOMODEL.csv"

#path to save the exogenous tunned model 
path_for_exogenous_model = "D:/data_split_stacked/"

#features to work during the hypertuning
features_names = ["tick", "imbalance", "volume", "volatility", "fracdiff"]

#label name in the .csv features-label ingested
label_name = ['tripleBarrier']


###STEPS


#1. FIRST FUNCTION: data preparation | return fixed X, y and t1 
X, y, t1 = dataPreparation_forTuning(
    csv_path, features_names, label_name
    )

#2. MAIN CLASSIFIER HYPERTUNNING | return model well-tunned
exogenous_model = clfHyperFit(
    feat=X, lbl=y, t1=t1, 
    param_grid=params_for_tuning, pipe_clf=model_for_tuning
    )

#3. SAVING MODEL USING PICKLE
pickle_filename = "exogenous_model.pkl"  

filepath = r'{}{}'.format(path_for_exogenous_model, pickle_filename)
pickle.dump(exogenous_model, open(filepath, 'wb'))