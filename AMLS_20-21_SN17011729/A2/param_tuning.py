import os
import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from A2.a2_test_train import A2
from itertools import product

os.chdir('..')

# If True, metrics are computed, if False, metrics are viewed/analyzed
GET_CV_METRICS = True

CELEB_DATASET_DIR = os.path.join("Datasets", "celeba")

taskA2 = A2(CELEB_DATASET_DIR, os.path.join("A2", "temp"))

if GET_CV_METRICS:

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},
                        {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'degree': [3, 5, 8]},
                        {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

    cv_clf = GridSearchCV(SVC(), tuned_parameters, scoring='%s_macro' % 'precision')

    cv_clf.fit(taskA2.X_train, taskA2.y_train)

    print(cv_clf.best_params_)
    means = cv_clf.cv_results_['mean_test_score']
    stds = cv_clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, cv_clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))

pass