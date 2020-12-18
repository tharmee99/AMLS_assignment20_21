import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.svm import SVC
import defs
import numpy as np

class SVM_model(defs.Model):
    def __init__(self, kernel='poly', C=100.0, degree=5, gamma='scale'):
        super(SVM_model, self).__init__()
        print("Instantiating SVC model...")
        self.model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
        self.confusion_matrix = None
        # self.model = KNeighborsClassifier(n_neighbors=25)
        pass

    def train(self, X_train, y_train):
        # X_train = np.reshape(X_train, (X_train.shape[0], -1))
        print("Training SVC model...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        print("SVC model took {}s to train".format(time_taken))
        pass

    def test(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        accuracy = accuracy_score(y_test, y_pred)
        return y_pred, accuracy