import defs


class A1(defs.Task):
    def __init__(self, dataset_dir, temp_dir):
        super().__init__(name='Task A1 - Gender Classification',
                         dataset_dir=dataset_dir,
                         temp_dir=temp_dir,
                         label_feature='gender')

    def preprocess_image(self, image_dir):
        pass

    def train(self, x_train, y_train):
        pass

    def test(self, x_train, y_train):
        pass


if __name__ == '__main__':
    print("asd")
