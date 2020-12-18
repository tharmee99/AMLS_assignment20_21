import sys
import time

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import defs
import numpy as np

class RandForest_model(defs.Model):
    def __init__(self, n_estimators=100, criterion='gini', max_depth=5):
        super(RandForest_model, self).__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth)
        self.confusion_matrix = None
        pass

    def train(self, X_train, y_train):
        start_time = time.time()
        self.model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        print("Random Forests model took {}s to train".format(time_taken), file=sys.stderr)
        pass

    def test(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return y_pred, accuracy