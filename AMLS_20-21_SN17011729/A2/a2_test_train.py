import os

import cv2
import dlib
import defs
import numpy as np
from matplotlib import pyplot as plt
from A2 import models


class A2(defs.Task):
    def __init__(self,
                 dataset_dir,
                 temp_dir,
                 rescaled_image_dim=(164, 134),
                 approach=1,
                 colour=False):

        super().__init__(name='Task A2 - Emotion Classification',
                         dataset_dir=dataset_dir,
                         temp_dir=temp_dir,
                         label_feature='smiling')

        self.rescaled_image_dim = rescaled_image_dim
        self.colour = colour
        self.approach = approach

        if self.approach == 1:
            self.missed_faces = 0
            self.duplicate_faces = 0
            self.detector = dlib.get_frontal_face_detector()
            self.predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

        self.get_data()

        print(self.X_train.shape)
        print(self.X_test.shape)

        if self.approach == 1 or self.approach == 2:
            self.remove_ambiguous_samples()

        if len(self.X_test.shape) == 1:
            self.X_test.reshape(-1, 1)

        if len(self.X_train.shape) == 1:
            self.X_train.reshape(-1, 1)

        self.model = models.SVM_model()


    def __get_dlib_mouth_landmarks(self, img, show_img=False, original_size=(178, 218)):
        rects = self.detector(img, 1)

        num_faces = len(rects)

        if num_faces == 0:
            self.missed_faces += 1
            num_faces = 1
        elif num_faces > 1:
            self.duplicate_faces += 1

        face_areas = np.zeros((1, num_faces))
        face_shapes = np.zeros((136, num_faces), dtype=np.int64)

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

        mouth_landmarks = dlibout[48:]

        if show_img:
            for coord in dlibout:
                cv2.circle(img, (coord[0], coord[1]), 1, (0, 0, 255), -1)

            temp = cv2.resize(img, original_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            defs.view_image(temp)

        landmark_mean = np.mean(mouth_landmarks, axis=0)

        mouth_landmarks_scaled = mouth_landmarks - landmark_mean

        abs_mouth_landmarks = np.abs(mouth_landmarks_scaled)

        max_num = np.amax(abs_mouth_landmarks)

        if max_num != 0:
            mouth_landmarks_scaled = mouth_landmarks_scaled / max_num

        mouth_landmarks_scaled = np.reshape(mouth_landmarks_scaled, (-1))

        return mouth_landmarks_scaled

    def __get_dlib_mar(self, img, show_img=False, original_size=(178, 218)):
        rects = self.detector(img, 1)

        num_faces = len(rects)

        if num_faces == 0:
            self.missed_faces += 1
            num_faces = 1
        elif num_faces > 1:
            self.duplicate_faces += 1

        face_areas = np.zeros((1, num_faces))
        face_shapes = np.zeros((136, num_faces), dtype=np.int64)

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

        if len(rects) == 0:
            return 0

        d1 = defs.euclidean_distance(dlibout[50], dlibout[58])
        d2 = defs.euclidean_distance(dlibout[51], dlibout[57])
        d3 = defs.euclidean_distance(dlibout[52], dlibout[56])
        d4 = defs.euclidean_distance(dlibout[48], dlibout[54])

        if show_img:
            cv2.circle(img, (dlibout[50, 0], dlibout[50, 1]), 1, (0, 0, 255), -1)
            cv2.circle(img, (dlibout[58, 0], dlibout[58, 1]), 1, (0, 0, 255), -1)
            cv2.circle(img, (dlibout[51, 0], dlibout[51, 1]), 1, (0, 0, 255), -1)
            cv2.circle(img, (dlibout[57, 0], dlibout[57, 1]), 1, (0, 0, 255), -1)
            cv2.circle(img, (dlibout[52, 0], dlibout[52, 1]), 1, (0, 0, 255), -1)
            cv2.circle(img, (dlibout[56, 0], dlibout[56, 1]), 1, (0, 0, 255), -1)
            cv2.circle(img, (dlibout[48, 0], dlibout[48, 1]), 1, (0, 0, 255), -1)
            cv2.circle(img, (dlibout[54, 0], dlibout[54, 1]), 1, (0, 0, 255), -1)

            temp = cv2.resize(img, original_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            defs.view_image(temp)

        mar = (d1+d2+d3)/(3*d4)

        return mar

    def __get_canny_edges(self, img, show_img=False, original_size=(178, 218)):
        edges = cv2.Canny(img, 100, 200)
        resized = cv2.resize(edges, self.rescaled_image_dim, interpolation=cv2.INTER_AREA)

        if show_img:
            temp = cv2.resize(resized, original_size, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            defs.view_image(temp)

        resized = np.reshape(resized, (-1))

        norm_img = resized / 255.0
        return norm_img

    def preprocess_image(self, image_dir, show_img=False):
        cv_read_flag = 1 if self.colour else 0
        # Read image
        image = cv2.imread(image_dir, cv_read_flag)

        original_shape = (image.shape[1], image.shape[0])

        if self.approach == 0:
            resized = cv2.resize(image, self.rescaled_image_dim, interpolation=cv2.INTER_AREA)
            if show_img:
                temp = cv2.resize(resized, original_shape, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
                defs.view_image(temp)
            norm_img = resized / 255.0
            return norm_img
        elif self.approach == 1:
            return self.__get_dlib_mouth_landmarks(image, show_img=show_img, original_size=original_shape)
        elif self.approach == 2:
            return self.__get_dlib_mar(image, show_img=show_img, original_size=original_shape)
        elif self.approach == 3:
            return self.__get_canny_edges(image, show_img=show_img, original_size=original_shape)

    def remove_ambiguous_samples(self, test=False):
        X = []
        y = []

        if not test:
            for i in range(self.X_train.shape[0]):
                if not (np.amin(self.X_train[i]) == 0.0 and np.amax(self.X_train[i]) == 0.0):
                    X.append(self.X_train[i])
                    y.append(self.y_train[i])

            self.X_train = np.array(X)
            self.y_train = np.array(y)
        else:
            for i in range(self.X_test.shape[0]):
                if not (np.amin(self.X_test[i]) == 0.0 and np.amax(self.X_test[i]) == 0.0):
                    X.append(self.X_test[i])
                    y.append(self.y_test[i])

            self.X_test = np.array(X)
            self.y_test = np.array(y)


if __name__ == '__main__':
    root_dir = os.path.join(os.getcwd(), os.pardir)
    A2 = A2(os.path.join(root_dir, "Datasets", "celeba"), os.path.join(os.getcwd(), "temp"))

    print(A2.X_test.shape)
    print(A2.X_train.shape)

    A2.train()
    A2.test()
    A2.print_results()
    A2.remove_ambiguous_samples(test=True)
    A2.test()
    A2.print_results()

    print(A2.X_test.shape)
    print(A2.X_train.shape)
