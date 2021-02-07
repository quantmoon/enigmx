"""
@author: Quantmoon Technologies
webpage: https://www.quantmoon.tech//
"""


from sklearn.svm import NuSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (
    RandomForestClassifier,
    BaggingClassifier
    )
from sklearn.tree import DecisionTreeClassifier
from enigmx.test_features import FeatureImportance, PCA_QM, fitting_to_pca
from sklearn.ensemble import GradientBoostingClassifier  


#parameters
pathOut = "D:/feature_importance/"
minWLeaf = 0


#####################Feature Importance#######################################

#model 1
ga = GradientBoostingClassifier(random_state=0)

#model 2    
rf = RandomForestClassifier(random_state = 42,max_features=1)

#classifier 1
clf=DecisionTreeClassifier(criterion='entropy',max_features=1,
                              class_weight='balanced',
                               min_weight_fraction_leaf=minWLeaf)

#classifier 2
clf=BaggingClassifier(base_estimator=clf,
                          max_features=1.,
                          oob_score=True)
nb = GaussianNB()      
nu = NuSVC(nu=.0099, kernel = 'poly')

################### Paramter for Feature Importance ##########################

method = "MDA"
model_selected = clf

individuals_csv = "D:/data_single_stacked/"
rolling_window = 50

instance = FeatureImportance(individuals_csv, 
                             window = rolling_window, 
                             win_type='gaussian', 
                             add_parameter=25)

print("*****"*10)
print("1. Generating Stacked")

stacked_ = instance.__getStacked__()

stacked_.to_csv(
    pathOut + 'STACKED_WINDOW' + str(rolling_window) + '.csv', 
    index=False
    )

print(" ")


print("*****"*10)
print("2. Get MDA & MDI Feature Importance")

instance.get_feature_importance(pathOut, 'MDA', nb)

print(" ")



print("*****"*10)
print("3. Compute PCA")


df_matrix = fitting_to_pca(pathOut+'STACKED_WINDOW50.csv')

print(
PCA_QM(df_matrix).get_pca(save_and_where = (True, pathOut))
)

print(" ")