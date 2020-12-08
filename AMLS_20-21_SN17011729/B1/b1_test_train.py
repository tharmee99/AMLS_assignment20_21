import defs
import cv2
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

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

    def instantiate_model(self, num_of_labels=5, show_summary=False):

        # Convolutional Neural Network in tensorflow
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(8, (5, 5), activation='relu', input_shape=(*self.rescaled_image_dim, 1)),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Conv2D(8, (5, 5), activation='relu'),
            tf.keras.layers.MaxPool2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(100, activation='relu'),
            tf.keras.layers.Dense(num_of_labels, activation='softmax')
        ])

        # Compile model with loss function and optimizer
        self.model.compile(loss='categorical_crossentropy',
                           metrics=['accuracy'],
                           optimizer='adam')

        # Show summary of layer parameters and shapes
        if show_summary:
            self.model.summary()

        # Read/Build dataset
        if os.path.isfile(os.path.join(self.temp_dir, 'X.npy')) and os.path.isfile(os.path.join(self.temp_dir, 'y.npy')):
            X = self.read_intermediate('X.npy')
            y = self.read_intermediate('y.npy')

            # TODO : Check if imported dims agree with self.rescaled_image_dim

        else:
            X, y = self.build_design_matrix()
            self.save_intermediate(X, 'X')
            self.save_intermediate(y, 'y')

        # Onehot encode labels
        enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
        y = enc.fit_transform(y.reshape((-1, 1)))

        # Test-train split of dataset
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=42)

    def preprocess_image(self, image_dir, image_scale=0.1):
        img = cv2.imread(image_dir, 0)

        # Resize image
        resized = cv2.resize(img, self.rescaled_image_dim, interpolation=cv2.INTER_AREA)

        # TODO : Equalize histogram of image

        # Reshape to append to design matrix
        resized = np.reshape(resized, (1, *resized.shape, 1))

        # Scale values to (0,1)
        scaled = resized/255.0

        return scaled

    def train(self):
        self.model.fit(self.X_train, self.y_train, epochs=25, validation_data=(self.X_test, self.y_test))

        # TODO : Save trained model using self.save_intermediate

        pass

    def test(self, X_test=None, y_test=None):
        pass


if __name__ == '__main__':
    root_dir = os.path.join(os.getcwd(), os.pardir)
    B1 = B1(os.path.join(root_dir, "Datasets", "cartoon_set"), os.path.join(os.getcwd(), "temp"))
    pass