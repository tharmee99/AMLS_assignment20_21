import os
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from B2.b2_test_train import B2
from itertools import product

os.chdir('..')

# If True, metrics are computed, if False, metrics are viewed/analyzed
GET_CV_METRICS = True

CARTOON_DATASET_DIR = os.path.join("Datasets", "cartoon_set")

taskB2 = B2(CARTOON_DATASET_DIR, os.path.join("B1", "temp"))

if GET_CV_METRICS:

    tuned_parameters = {'n_estimators': [100, 500], 'criterion': ['gini', 'entropy'], 'max_depth': [10, 100, 1000]}

    cv_clf = GridSearchCV(RandomForestClassifier(), tuned_parameters, scoring='%s_macro' % 'precision')

    cv_clf.fit(taskB2.X_train, taskB2.y_train)

    print(cv_clf.best_params_)
    means = cv_clf.cv_results_['mean_test_score']
    stds = cv_clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, cv_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

pass