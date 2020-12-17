import defs
import cv2
import os
import tensorflow as tf

from A1.models import CNN_model

class A1(defs.Task):
    def __init__(self, dataset_dir, temp_dir):
        super().__init__(name='Task A1 - Gender Classification',
                         dataset_dir=dataset_dir,
                         temp_dir=temp_dir,
                         label_feature='gender')

        self.one_hot_encoded = True

        self.cnn_kernel_size = (5, 5)
        self.cnn_filters = 8
        self.layer_activation = 'relu'
        self.dense_layer_size = 100

        self.rescaled_image_dim = (54, 44)
        self.colour = True

        self.initialize_model()

    def initialize_model(self):
        self.get_data()
        self.model = CNN_model(self.rescaled_image_dim, 2,
                               cnn_filters=self.cnn_filters,
                               cnn_kernel_size=self.cnn_kernel_size,
                               layer_activation=self.layer_activation,
                               dense_layer_size=self.dense_layer_size)

    def preprocess_image(self, image_dir, show_img=False):
        cv_read_flag = 1 if self.colour else 0
        # Read image
        image = cv2.imread(image_dir, cv_read_flag)
        # Resize image
        resized = cv2.resize(image, self.rescaled_image_dim, interpolation=cv2.INTER_AREA)

        if show_img:
            temp = cv2.resize(resized, (image.shape[1], image.shape[0]), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            defs.view_image(temp)

        norm_img = resized / 255.0

        if not self.colour:
            norm_img = norm_img.reshape(*self.rescaled_image_dim, 1)

        return norm_img


if __name__ == '__main__':
    pass
