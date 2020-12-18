import time

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
import defs

class SVM_model(defs.Model):
    def __init__(self, kernel='poly', C=100.0, degree=5, gamma='scale'):
        super(SVM_model, self).__init__()
        self.model = SVC(kernel=kernel, C=C, degree=degree, gamma=gamma)
        self.confusion_matrix = None
        pass

    def train(self, X_train, y_train):
        # X_train = np.reshape(X_train, (X_train.shape[0], -1))
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