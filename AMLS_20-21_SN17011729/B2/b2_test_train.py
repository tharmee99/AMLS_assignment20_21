import cv2
from sklearn.feature_selection import VarianceThreshold

import defs
from B2 import models


class B2(defs.Task):
    def __init__(self, dataset_dir, temp_dir, rescaled_image_dim=(40, 40)):
        super().__init__(name='Task B2 - Eye Colour Classification',
                         dataset_dir=dataset_dir,
                         temp_dir=temp_dir,
                         label_feature='eye_color')
        self.rescaled_image_dim = rescaled_image_dim
        self.lv_selector = None

        self.get_data()

        self.apply_low_variance_filter()
        self.apply_low_variance_filter(test=True)

        self.model = models.RandForest_model()

        print(type(self.model.model).__module__.split('.')[0])

        # TODO : write method to view low variance filter mask
        # TODO : include feature reduction numerics
        # asd = self.lv_selector.get_support()
        # defs.view_image(asd.reshape(*rescaled_image_dim, 3)[:,:,0].astype('float32'))
        # defs.view_image(asd.reshape(*rescaled_image_dim, 3)[:,:,1].astype('float32'))
        # defs.view_image(asd.reshape(*rescaled_image_dim, 3)[:,:,2].astype('float32'))

    def preprocess_image(self, image_dir, show_img=False):
        # Read image
        image = cv2.imread(image_dir, 1)

        x_mid = image.shape[0] // 2

        image = image[:, :x_mid]

        # Resize image
        resized = cv2.resize(image, self.rescaled_image_dim, interpolation=cv2.INTER_AREA)

        if show_img:
            temp = cv2.resize(resized, (250, 500), fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
            defs.view_image(temp)

        return resized.reshape(-1)

    def apply_low_variance_filter(self, test=False):
        if test and self.lv_selector is None:
            raise Exception("Low Variance filter should be fit on train data first before applying to test data")

        if test:
            self.X_test = self.lv_selector.transform(self.X_test)
        else:
            self.lv_selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
            self.X_train = self.lv_selector.fit_transform(self.X_train)


if __name__ == '__main__':
    print("asd")
