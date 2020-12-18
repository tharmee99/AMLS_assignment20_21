import cv2
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import defs
import tensorflow as tf
from matplotlib import pyplot as plt


class CNN_model(tf.keras.Model, defs.Model):
    def __init__(self,
                 input_shape,
                 num_of_classes,
                 cnn_filters=8,
                 cnn_kernel_size=(3, 3),
                 layer_activation='relu',
                 dense_layer_size=1000,
                 colour=True):

        super(CNN_model, self).__init__()
        self.requires_validation_set = True
        self.requires_flat_input = False
        self.history_loss = None
        self.confusion_matrix = None

        channels = 3 if colour else 1

        self.conv_layer1 = tf.keras.layers.Conv2D(cnn_filters, cnn_kernel_size, activation=layer_activation,
                                                  input_shape=(*input_shape, channels))
        self.pool_layer1 = tf.keras.layers.MaxPool2D(2, 2)
        self.conv_layer2 = tf.keras.layers.Conv2D(cnn_filters, cnn_kernel_size, activation=layer_activation)
        self.pool_layer2 = tf.keras.layers.MaxPool2D(2, 2)
        self.flatten_layer = tf.keras.layers.Flatten()
        self.dense_layer1 = tf.keras.layers.Dense(dense_layer_size, activation=layer_activation)
        self.dense_layer2 = tf.keras.layers.Dense(num_of_classes, activation='softmax')

        self.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    def call(self, inputs, training=None, mask=None):
        x = self.conv_layer1(inputs)
        x = self.pool_layer1(x)
        x = self.conv_layer2(x)
        x = self.pool_layer2(x)
        x = self.flatten_layer(x)
        x = self.dense_layer1(x)
        x = self.dense_layer2(x)
        return x

    def train(self, X_train, y_train, validation_data=None):

        self.history_loss = self.fit(X_train, y_train,
                                     epochs=7,
                                     validation_data=validation_data,
                                     shuffle=True,
                                     verbose=1)

        # self.view_accuracy_history()
        pass

    def test(self, X_test, y_test):
        y_pred = self.call(X_test)
        y_pred_class = np.argmax(y_pred, axis=-1)
        y_test_class = np.argmax(y_test, axis=-1)

        self.confusion_matrix = confusion_matrix(y_pred_class, y_test_class)
        return y_pred, accuracy_score(y_pred_class, y_test_class)

    def view_accuracy_history(self):

        x = np.arange(1, 21, step=1)

        plt.plot(x, self.history_loss.history['loss'], label="train loss")
        plt.plot(x, self.history_loss.history['val_loss'], label="test loss")
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.xticks(x)
        plt.show()

    # TODO : Refine get_filter_output methods

    def get_filter1_output(self, X_sample):

        if len(X_sample.shape) < 4:
            X_sample = X_sample.reshape(1, *X_sample.shape)

        x = self.conv_layer1(X_sample)
        x = self.pool_layer1(x)
        x_np = x.numpy()

        for i in range(x_np.shape[0]):
            fig, axs = plt.subplots(3, 3)

            print(X_sample[i].shape)

            rgb_img = cv2.cvtColor(np.uint8(X_sample[i]*255), cv2.COLOR_BGR2RGB)
            resized_original = cv2.resize(rgb_img, (178, 218), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            axs[0, 0].imshow(resized_original)
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')

            for j in range(x_np.shape[-1]):
                filt_out = x_np[i, :, :, j]
                resized_filt_out = cv2.resize(filt_out, (178, 218), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

                x_plot = int((j+1)//3)
                y_plot = int((j+1) % 3)

                axs[x_plot, y_plot].imshow(resized_filt_out, cmap='gray')
                axs[x_plot, y_plot].set_title("Conv Layer 1, Filter {}".format(j+1))
                axs[x_plot, y_plot].axis('off')

            plt.show()

        return x

    def get_filter2_output(self, X_sample):

        if len(X_sample.shape) < 4:
            X_sample = X_sample.reshape(1, *X_sample.shape)

        x = self.conv_layer1(X_sample)
        x = self.pool_layer1(x)
        x = self.conv_layer2(x)
        x = self.pool_layer2(x)
        x_np = x.numpy()

        for i in range(x_np.shape[0]):
            fig, axs = plt.subplots(3, 3)

            rgb_img = cv2.cvtColor(np.uint8(X_sample[i] * 255), cv2.COLOR_BGR2RGB)
            resized_original = cv2.resize(rgb_img, (178, 218), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            axs[0, 0].imshow(resized_original)
            axs[0, 0].set_title('Original Image')
            axs[0, 0].axis('off')

            for j in range(x_np.shape[-1]):
                filt_out = x_np[i, :, :, j]
                resized_filt_out = cv2.resize(filt_out, (178, 218), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)

                x_plot = int((j + 1) // 3)
                y_plot = int((j + 1) % 3)

                axs[x_plot, y_plot].imshow(resized_filt_out, cmap='gray')
                axs[x_plot, y_plot].set_title("Conv Layer 2, Filter {}".format(j+1))
                axs[x_plot, y_plot].axis('off')

            plt.show()

        return x
