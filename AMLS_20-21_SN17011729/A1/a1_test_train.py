import defs
import cv2
import os
import numpy as np

class A1(defs.Task):
    def __init__(self, dataset_dir, temp_dir):
        super().__init__(name='Task A1 - Gender Classification',
                         dataset_dir=dataset_dir,
                         temp_dir=temp_dir,
                         label_feature='gender')

    def preprocess_image(self, image_dir):
        arr = cv2.imread(image_dir, 0)
        print(arr.shape)
        return arr

    def train(self, x_train, y_train):
        pass

    def test(self, x_train, y_train):
        pass


if __name__ == '__main__':

    root_dir = os.path.join(os.getcwd(), os.pardir)
    A1 = A1(os.path.join(root_dir, "Datasets", "celeba"), os.path.join(os.getcwd(), "temp"))

    # X_raw, y = A1.build_design_matrix()
    #
    # A1.save_intermediate(X_raw, "X_raw")
    # A1.save_intermediate(y, "y")
    A1.read_intermediate('random_arr.npy', get_latest=False)
    pass