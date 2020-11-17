import defs


class B1(defs.Task):
    def __init__(self, dataset_dir, temp_dir):
        super().__init__(name='Task B1 - Face Shape Classification',
                         dataset_dir=dataset_dir,
                         temp_dir=temp_dir,
                         label_feature='face_shape')

    def preprocess_image(self, image_dir):
        pass

    def train(self, x_train, y_train):
        pass

    def test(self, x_train, y_train):
        pass


if __name__ == '__main__':
    print("asd")
