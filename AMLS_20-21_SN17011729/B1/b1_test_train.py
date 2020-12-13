import defs
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
from B1 import models


class B1(defs.Task):
    def __init__(self,
                 dataset_dir,
                 temp_dir,
                 rescaled_image_dim=(50, 50)):
        super().__init__(name='Task B1 - Face Shape Classification',
                         dataset_dir=dataset_dir,
                         temp_dir=temp_dir,
                         label_feature='face_shape')

        self.rescaled_image_dim = rescaled_image_dim
        self.model = models.SVM_model()
        self.get_data()

    def preprocess_image(self, image_dir):
        img = cv2.imread(image_dir, 0)

        # Resize image
        resized = cv2.resize(img, self.rescaled_image_dim, interpolation=cv2.INTER_AREA)

        # Reshape to append to design matrix
        # TODO : Reshape size conditional on self.model
        out_img = np.reshape(resized, (-1))

        # Scale values to (0,1)
        norm_img = out_img/255.0

        return norm_img


if __name__ == '__main__':
    root_dir = os.path.join(os.getcwd(), os.pardir)
    B1 = B1(os.path.join(root_dir, "Datasets", "cartoon_set"), os.path.join(os.getcwd(), "temp"))
    pass