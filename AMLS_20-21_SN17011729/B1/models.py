import time
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import defs


class SVM_model(defs.Model):
    def __init__(self, kernel='linear', C=1.0, degree=3, gamma='scale'):
        super(SVM_model, self).__init__()
        print("Instantiating SVC model...")
        self.model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
        pass

    def train(self, X_train, y_train):
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


class Logistic_model(defs.Model):
    def __init__(self):
        print("Instantiating Logistic Regression model...")
        self.model = LogisticRegression(verbose=1, max_iter=5e3)
        pass

    def train(self, X_train, y_train):
        print("Training Logistic Regression model...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        time_taken = time.time() - start_time
        print("Logistic Regression model took {}s to train".format(time_taken))
        pass

    def test(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        return y_pred, accuracy