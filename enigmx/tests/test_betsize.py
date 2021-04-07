"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""
import pickle
import pandas as pd
from enigmx.betsize import *
from enigmx.utils import dataPreparation_forTuning



model_path = "D:/data_split_stacked/exogenous_model.pkl"

exogenous_model = pickle.load(open(model_path, 'rb'))

print(exogenous_model )
print(temp)
#0) path to save the endogenous betsize metalabelling model
path_endogenous_model = "D:/feature_importance/"

#i) path where tunning data is allocated
csv_path = "D:/feature_importance/STACKED_ENDO_VOLUME_MDA.csv"

#ii) path where exogenous model is allocated (pickle file)
model_path = "D:/feature_importance/exogenous_model.pkl"


#iv) data preparation for tuning | 't1' is not useful
X, y, t1 = dataPreparation_forTuning(
    csv_path, features_names, label_name
    )
X = X.values
y = y.values


###STEPS

# 1) Open Exogenous Model
exogenous_model = pickle.load(open(model_path, 'rb'))

# 2) Predict using Exogenous Model
exogenous_predictions = exogenous_model.predict(X)


# 3) Ingest Predictions, Features & Real Labels in BetSize instance
betsizeInstace = BetSize(X, exogenous_predictions, y, endogenous_model='rf')

# 4) Endogenous Metalabelling Model Generation from BetSize Instance
endogenous_model = betsizeInstace.get_betsize()

# 5) Saving Endogenous Metalabelling Model using Pickle
filepath = r'{}{}'.format(path_endogenous_model, "endogenous_model.pkl")
pickle.dump(endogenous_model, open(filepath, 'wb'))
