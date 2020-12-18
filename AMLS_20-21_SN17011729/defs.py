import math
import pickle
import sys
import time

import cv2
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from functools import reduce

def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h) as we would normally do
    # with OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return x, y, w, h


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def view_image(img, window_name="image"):
    cv2.imshow(window_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def euclidean_distance(a, b):
    """
    Function to compute the euclidean distance between two points

    :param a: a tuple or list or length 2 representing the first coordinate
    :param b: a tuple or list of length 2 representing the second coordinate
    :return: a float representing the distance between the two coordinates
    """
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)


class Model:
    """
    A class that is subclassed by each of the models implemented. This ensures that a common train test reset function
    can be called and thereby allows interchangability of models during testing.
    """
    def __init__(self):
        self.requires_validation_set = False
        self.requires_flat_input = False
        pass

    def train(self, X_train, y_train):
        raise NotImplemented("Model Train function needs to be implemented")

    def test(self, X_test, y_test):
        raise NotImplemented("Model Test function needs to be implemented")

    def reset(self):
        raise NotImplemented("Model Reset function needs to be implemented")


class Task:
    """
    A class that is subclassed for each of the 4 tasks in this project. It contains all of the functions to read data,
    split data, save/load intermediate results as well as test/train methods.
    """
    def __init__(self, name, dataset_dir, temp_dir, label_feature):
        self.name = name
        self.dataset_dir = dataset_dir
        self.temp_dir = temp_dir
        self.label_feature = label_feature

        self.one_hot_encoded = False

        self.accepted_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
        self.test_accuracy = -1.0
        self.train_accuracy = -1.0

        self.pre_processing_time = -1.0

        self.enc = None

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_image(self, image_dir, show_img=False):
        """
        This method is implemented in each of the tasks as the pre-processing differs from task to task.

        :param image_dir: the path to the file to be pre-processed
        :param show_img: if the images should be shown as they are pre-processed (for debugging)
        :return: an array or matrix representing the features of the image to be trained on
        """
        raise NotImplemented("Preprocessing method must be implemented!")

    def get_data(self, X_arr_name='X.npy', y_arr_name='y.npy', verbose=0, force_calculation=False):
        """
        This method is called to load all the data and perform splits ready for the model to train and then test

        :param X_arr_name: The filename of the X matrix (design matrix)
        :param y_arr_name: The filename of the y array (labels array)
        :param verbose: If greater than 0, then feedback will be printed and shown via images
        :param force_calculation: If Ture, the pre-processing is carried out regardless of whether the save files exist
        """
        start_time = time.time()

        if force_calculation:
            X, y = self.build_design_matrix(verbose=verbose)
            self.save_intermediate(X, 'X')
            self.save_intermediate(y, 'y')
        else:
            try:
                X = self.read_intermediate(X_arr_name)
                y = self.read_intermediate(y_arr_name)
            except FileNotFoundError:
                X, y = self.build_design_matrix(verbose=verbose)
                self.save_intermediate(X, 'X')
                self.save_intermediate(y, 'y')

        if self.one_hot_encoded:
            self.enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            y = self.enc.fit_transform(y.reshape(-1, 1))
        else:
            self.enc = None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)

        self.pre_processing_time = time.time() - start_time
        pass

    def initialize_model(self):
        raise NotImplemented("Method to initialize model must be implemented")

    def build_design_matrix(self, dataset_dir=None, verbose=0):
        """
        This method build the design matrix and the labels array but does not split it. It is called by get_data()
        :param dataset_dir: the directory in which the dataset is provided
        :param verbose: if greater than 0, feedback will be printed and shown via pictures
        :return: design matrix X and labels array y
        """
        if dataset_dir is not None:
            image_paths = [os.path.join(dataset_dir, "img", img) for img in
                           os.listdir(os.path.join(dataset_dir, "img"))]
            labels_csv = pd.read_csv(os.path.join(dataset_dir, "labels.csv"), sep='\t')
        else:
            image_paths = [os.path.join(self.dataset_dir, "img", img) for img in
                           os.listdir(os.path.join(self.dataset_dir, "img"))]
            labels_csv = pd.read_csv(os.path.join(self.dataset_dir, "labels.csv"), sep='\t')

        if self.label_feature in labels_csv.columns:
            if 'img_name' in labels_csv.columns:
                labels = {r['img_name']: r[self.label_feature] for i, r in labels_csv.iterrows()}
            elif 'file_name' in labels_csv.columns:
                labels = {r['file_name']: r[self.label_feature] for i, r in labels_csv.iterrows()}
            else:
                raise Exception("No column for file name in labels.csv")
        else:
            raise Exception("label feature cannot be found in csv file")

        X = []
        y = []

        for img in tqdm(image_paths, desc="Pre-processing dataset", file=sys.stdout):
            to_append = self.preprocess_image(img, show_img=(verbose > 0))

            if to_append is not None:
                X.append(to_append)
                y.append(labels[os.path.basename(img)])

        X_np = np.array(X)
        y_np = np.array(y)

        return X_np, y_np

    def save_intermediate(self, obj, file_name, overwrite=True):

        if not os.path.exists(self.temp_dir):
            os.makedirs(self.temp_dir)

        if type(obj).__module__ == np.__name__:
            if overwrite or (not os.path.isfile(os.path.join(self.temp_dir, file_name + ".npy"))):
                np.save(os.path.join(self.temp_dir, file_name), obj)
            else:
                idx = 1
                file_name_new = file_name + '_' + str(idx)
                while os.path.isfile(os.path.join(self.temp_dir, file_name_new + ".npy")):
                    idx += 1
                    file_name_new = file_name + '_' + str(idx)
                np.save(os.path.join(self.temp_dir, file_name_new), obj)
        else:
            raise TypeError("{} is not supported".format(type(obj)))

    def read_intermediate(self, file_name, get_latest=False):
        file = file_name.split('.')

        if file[1] == 'npy':
            if not get_latest:
                return np.load(os.path.join(self.temp_dir, file_name))
            else:
                idx = 1
                file_name_tmp = file[0] + '_' + str(idx)
                while os.path.isfile(os.path.join(self.temp_dir, file_name_tmp + ".npy")):
                    idx += 1
                    file_name_tmp = file[0] + '_' + str(idx)
                return np.load(os.path.join(self.temp_dir, file[0] + '_' + str(idx - 1) + ".npy"))
        else:
            raise TypeError(file[1] + " files are not supported")

    def compute_kfold_cv_score(self, folds=5):
        """
        computes a k-fold cross validation score. The accuracies are tracked as well as the time taken to train, test
        and pre-process the images.

        :param folds: The number of folds to performs
        :return: a dictionary of different metrics
        """
        skf = StratifiedKFold(n_splits=folds)

        metrics = {}

        self.get_data(force_calculation=True)

        metrics["pre_processing_time"] = self.pre_processing_time

        metrics["training_times"] = []
        metrics["testing_times"] = []
        metrics["accuracies"] = []

        if self.one_hot_encoded:
            y_temp = self.enc.inverse_transform(self.y_train)
        else:
            y_temp = self.y_train

        for train_index, test_index in skf.split(self.X_train, y_temp):
            X_train, X_test = self.X_train[train_index], self.X_train[test_index]
            y_train, y_test = self.y_train[train_index], self.y_train[test_index]

            start_time = time.time()
            self.train(X_train, y_train)
            train_done_time = time.time()
            _, acc = self.test(X_test, y_test)
            metrics["training_times"].append(train_done_time-start_time)
            metrics["testing_times"].append(time.time()-train_done_time)
            metrics["accuracies"].append(acc)

        return metrics

    def train(self, X_train=None, y_train=None):
        '''
        The train function that in turn calls the model.train function

        :param X_train: If supplied, the supplied X_train and y_train will be used to train the model
        :param y_train: If supplied, the supplied X_train and y_train will be used to train the model
        '''
        if self.model is None:
            raise Exception("Model has not been initialized")

        self.initialize_model()

        if X_train is None or y_train is None:

            if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
                raise Exception("Data has not been imported")

            if self.model.requires_validation_set:
                self.model.train(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test))
            else:
                self.model.train(self.X_train, self.y_train)

            _, self.train_accuracy = self.test(self.X_train, self.y_train)

            # TODO : Save trained model
        else:
            if self.model.requires_validation_set:
                self.model.train(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test))
            else:
                self.model.train(self.X_train, self.y_train)
        pass

    def test(self, X_test=None, y_test=None):
        """
        Tests the trained model on test data

        :param X_test: If provided, the model will be trained on the provided data, otherwise on self.X_test
        :param y_test: If provided, the model will be trained on the provided data, otherwise on self.y_test
        :return:
        """
        if self.model is None:
            raise Exception("Model has not been initialized")

        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise Exception("Data has not been imported")

        if X_test is not None:
            return self.model.test(X_test, y_test)
        else:
            y_pred, self.test_accuracy = self.model.test(self.X_test, self.y_test)
            return y_pred, self.test_accuracy

    def print_results(self):
        """
        Prints a summary of the metrics
        """
        if self.train_accuracy < 0:
            print("No model has been trained")
        elif self.test_accuracy < 0:
            print("Model has not been tested on test dataset")
        else:
            print("Train Accuracy:\t{}\nTest Accuracy:\t{}".format(self.train_accuracy, self.test_accuracy))


if __name__ == '__main__':
    print("Test")
