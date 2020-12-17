import time

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import defs
import numpy as np

class SVM_model(defs.Model):
    def __init__(self, kernel='rbf', C=1.0, degree=3, gamma='scale'):
        super(SVM_model, self).__init__()
        print("Instantiating SVC model...")
        self.model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
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
        accuracy = accuracy_score(y_test, y_pred)
        return y_pred, accuracy