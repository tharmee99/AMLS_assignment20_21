import math
import sys

import cv2
import numpy as np
import os
import pandas as pd

from sklearn.model_selection import train_test_split
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


def factors(n):
    return set(reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def euclidean_distance(a, b):
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

class Model:
    def __init__(self):
        self.requires_validation_set = False
        pass

    def train(self, X_train, y_train):
        raise NotImplemented("Model Train function needs to be implemented")

    def test(self, X_test, y_test):
        raise NotImplemented("Model Test function needs to be implemented")


class Task:
    def __init__(self, name, dataset_dir, temp_dir, label_feature):
        self.name = name
        self.dataset_dir = dataset_dir
        self.temp_dir = temp_dir
        self.label_feature = label_feature

        self.accepted_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
        self.test_accuracy = -1.0
        self.train_accuracy = -1.0

        self.oh_enc_categories = None

        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_image(self, image_dir, show_img=False):
        raise NotImplemented("Preprocessing method must be implemented!")

    def get_data(self, X_arr_name='X.npy', y_arr_name='y.npy', one_hot_encoded=False, verbose=0):
        try:
            X = self.read_intermediate(X_arr_name)
            y = self.read_intermediate(y_arr_name)
        except FileNotFoundError:
            X, y = self.build_design_matrix(verbose=verbose)
            self.save_intermediate(X, 'X')
            self.save_intermediate(y, 'y')

        if one_hot_encoded:
            enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
            y = enc.fit_transform(y.reshape(-1, 1))
            self.oh_enc_categories = enc.categories_

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2)
        pass

    def build_design_matrix(self, dataset_dir=None, verbose=0):
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
            raise TypeError(type(obj) + " is not supported")

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

    def train(self):
        if self.model is None:
            raise Exception("Model has not been initialized")

        if self.X_train is None or self.X_test is None or self.y_train is None or self.y_test is None:
            raise Exception("Data has not been imported")

        if self.model.requires_validation_set:
            self.model.train(self.X_train, self.y_train, validation_data=(self.X_test, self.y_test))
        else:
            self.model.train(self.X_train, self.y_train)

        _, self.train_accuracy = self.test(self.X_train, self.y_train)

        # TODO : Save trained model using self.save_intermediate

        pass

    def test(self, X_test=None, y_test=None):
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
        if self.train_accuracy < 0:
            print("No model has been trained")
        elif self.test_accuracy < 0:
            print("Model has not been tested on test dataset")
        else:
            print("Train Accuracy:\t{}\nTest Accuracy:\t{}".format(self.train_accuracy, self.test_accuracy))


if __name__ == '__main__':
    print("Test")
