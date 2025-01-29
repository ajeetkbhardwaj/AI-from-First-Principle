"""
Grid Search Cross Validation Implementation
1. Cross Validation : 
- We need a chunk of data from given training data (70% of train)
- Randomly Select such that each chunk don't have same data points

2. Grid Searching :
- Train Data Chunk:(Features, labels)
- ML Model : Classifier or Regressor etc
- Hyperparameters : {k, n}
- Number of Folds

"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from sklearn.metrics import accuracy_score

#%% functions
def get_random_indices(X_train):
    """
    Getting random samples of indices which represents the 70% of train data

    Arguments :
       X_train
    Outputs :
       List of randomly selected indices of data
    """
    # 1. list range [0, len(X_train)]
    # 2. 
    indices = random.sample(range(0, len(X_train)), int(0.7*len(X_train)))

def KNNGridSearchCV(X_train, y_train, knn_clf, hparams, folds):
    """
    
    """
    TAS_Mean = []
    CVAS_Mean = []
    # loop for each hyper parameter
    for k in tqdm(hparams['n_nbd']):
        TAS_folds = []
        CVAS_folds = []
        # loop for each fold
        for fold in range(folds):
            TIS_Current_fold = get_random_indices(X_train)
            CVIS_Current_fold = list(set(list(range(1, len(X_train))))) - set(TIS_Current_fold)
            
            XT_current_fold = X_train[TIS_Current_fold]
            yT_current_fold = y_train[TIS_Current_fold]

            XCV_current_fold = X_train[CVIS_Current_fold]
            YCV_current_fold = y_train[CVIS_Current_fold]


            model = knn_clf
            model.n_neighbors = k
            model.fit(XT_current_fold, yT_current_fold)

            y_predCV = model.predict(XCV_current_fold)
            CVAS_folds.append(accuracy_score(YCV_current_fold, y_predCV))

            y_predTrain = model.predict(XT_current_fold)
            TAS_folds.append(accuracy_score(yT_current_fold, y_predTrain))
        
        # Update mean se for current value of k in n_nbd
        TAS_Mean.append(np.mean(np.array(TAS_folds)))
        CVAS_Mean.append(np.mean(np.array(CVAS_folds)))
    return (TAS_Mean, CVAS_Mean)

#%%
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
X,y = make_classification(n_samples=10000, 
                          n_features=2, 
                          n_informative=2, 
                          n_redundant= 0, 
                          n_clusters_per_class=1, 
                          random_state=60)

X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,
                                                    random_state=42)


plt.scatter(X_test[:, 0], 
            X_test[:, 1], 
            c=y_test)

#%%

from sklearn.neighbors import KNeighborsClassifier

knn_clf = KNeighborsClassifier()

hparams = {'n_nbd': [3,5,7,9,11,13,15,17,19,21,23]}
folds = 5

TSA, CVSA = KNNGridSearchCV(X_train, y_train, knn_clf, hparams, folds)

plt.plot(hparams['n_nbd'], TSA, label='Train Accuracy Score')
plt.plot(hparams['n_nbd'], CVSA, label='CV Accuracy Score')
plt.title('Hyper-parameter K of KNN Algo VS Accuracy Plot')
plt.legend()
plt.show()
# %%
