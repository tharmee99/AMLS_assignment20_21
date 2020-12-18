# README

The defs.py file contains class definitions as well as common helper functions used throughout the project. For example, the Task class defined in **defs.py** is subclassed for each task and contains functions to import data as well as other functions.

The **main.py** file is the file that runs through the whole import, pre-process, train, test cycle for each task by calling the corresponding methods.

Each folder A1, A2, B1, B2 contains the python scripts specific to that task. The **XX_test_train.py** files contain the subclass of the Task class and also contains methods for pre-processing the images. The **models.py** files contain the model used for that task. As models from different modules were implemented, this was necessary to achieve interchangeability between different models quickly. The **param_tuning.py** files contain code used to tune hyperparameters of the models.

The **shape_predictor_68_face_landmarks.dat** file is the pre-trained face landmark extractor required for task A2.

Upon execution, temp folders will be generated in each of the task's folders. In these folders, the pre-processed images will be stored as npy files. These can easily be read into numpy arrays upon further executions. This greatly reduces the time taken to execute the code.

## Required Packages

* numpy
* opencv-python
* tensorflow
* dlib
* sklearn
* pandas
* pandas
* tqdm
* matplotlib