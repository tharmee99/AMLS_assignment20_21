import dlib

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

        self.missed_faces = 0
        self.duplicate_faces = 0
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        self.get_data()

    def __get_dlib_mouth_landmarks(self, img, show_img=False, original_size=(500, 500)):

        img = cv2.equalizeHist(img)

        face_areas = np.zeros((1, 1))
        face_shapes = np.zeros((136, 1), dtype=np.int64)

        rects = dlib.rectangles()
        rect = dlib.rectangle(140, 200, 360, 410)

        rects.append(rect)

        for (i, rect) in enumerate(rects):
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            temp_shape = self.predictor(img, rect)
            temp_shape = defs.shape_to_np(temp_shape)

            # convert dlib's rectangle to a OpenCV-style bounding box
            # [i.e., (x, y, w, h)],
            #   (x, y, w, h) = face_utils.rect_to_bb(rect)
            (x, y, w, h) = defs.rect_to_bb(rect)
            face_shapes[:, i] = np.reshape(temp_shape, [136])
            face_areas[0, i] = w * h

        # find largest face and keep
        dlibout = np.reshape(np.transpose(face_shapes[:, np.argmax(face_areas)]), [68, 2])

        jawline = dlibout[:16]

        if show_img:
            for coord in jawline:
                cv2.circle(img, (coord[0], coord[1]), 1, (0, 0, 0), -1)

            temp = cv2.resize(img, original_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            defs.view_image(temp)

        return dlibout

    def preprocess_image(self, image_dir, show_img=False):
        img = cv2.imread(image_dir, 0)

        self.__get_dlib_mouth_landmarks(img)

        x_mid = img.shape[0]//2

        img = img[:, :x_mid]

        # Resize image
        resized = cv2.resize(img, self.rescaled_image_dim, interpolation=cv2.INTER_AREA)

        if show_img:
            temp = cv2.resize(img, (250, 500), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            defs.view_image(temp)

        # Reshape to append to design matrix
        # TODO : Reshape size conditional on self.model
        if self.model.requires_flat_input:
            resized = np.reshape(resized, (-1))

        # Scale values to (0,1)
        norm_img = resized/255.0

        return norm_img


if __name__ == '__main__':
    root_dir = os.path.join(os.getcwd(), os.pardir)
    B1 = B1(os.path.join(root_dir, "Datasets", "cartoon_set"), os.path.join(os.getcwd(), "temp"))
    pass