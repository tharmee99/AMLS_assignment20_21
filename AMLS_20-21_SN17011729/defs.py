import numpy as np
import os
import pandas as pd


class Task:
    def __init__(self, name, dataset_dir, temp_dir, label_feature):
        self.name = name
        self.dataset_dir = dataset_dir
        self.temp_dir = temp_dir
        self.label_feature = label_feature

        self.accepted_formats = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif")
        self.test_accuracy = -1.0
        self.train_accuracy = -1.0

    def preprocess_image(self, image_dir):
        raise NotImplemented("Preprocessing method must be implemented!")

    def build_design_matrix(self):

        # Identify which format the images are stored in
        file_ext = -1
        for idx, ext in enumerate(self.accepted_formats):
            if os.path.isfile(os.path.join(self.dataset_dir, "img", "0" + ext)):
                file_ext = idx
                break

        # If iamge cannot be found or is of incorrect format, throw exception
        if file_ext == -1:
            raise FileNotFoundError("Dataset directory was not found. Ensure that the directory is correct and images "
                                    "are of supported format")

        # Start building the design matrix by reading the first image
        image_path = os.path.join(self.dataset_dir, "img", "0" + self.accepted_formats[file_ext])
        X = self.preprocess_image(image_path)

        # Iterate through each image and preprocess it before appending it to the design matrix
        file_idx = 1
        image_path = os.path.join(self.dataset_dir, "img", str(file_idx) + self.accepted_formats[file_ext])
        while os.path.isfile(image_path):
            X = np.append(X, self.preprocess_image(image_path))
            file_idx += 1
            image_path = os.path.join(self.dataset_dir, "img", str(file_idx) + self.accepted_formats[file_ext])
            print(file_idx)

        # Read and build the label vector
        csv = pd.read_csv(os.path.join(self.dataset_dir, "labels.csv"), sep='\t')

        if self.label_feature in csv.columns:
            y = csv[self.label_feature].to_numpy()
        else:
            raise Exception("label feature cannot be found in csv file")

        return X, y

    def train(self, x_train, y_train):
        raise NotImplemented("Training method must be implemented!")

    def test(self, x_train, y_train):
        raise NotImplemented("Testing method must be implemented")

    def print_results(self):
        if self.train_accuracy < 0:
            print("No model has been trained")
        elif self.test_accuracy < 0:
            print("Model has not been tested on test dataset")
        else:
            print("Train Accuracy: {}\nTest Accuracy: {}".format(self.train_accuracy, self.train_accuracy))


if __name__ == '__main__':
    print("Test")
