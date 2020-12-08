import os

from A1.a1_test_train import A1
from A2.a2_test_train import A2
from B1.b1_test_train import B1
from B2.b2_test_train import B2


CARTOON_DATASET_DIR = os.path.join("Datasets", "cartoon_set")

########################################################################################################################
# Task B1

taskB1 = B1(CARTOON_DATASET_DIR, os.path.join("B1", "temp"))

taskB1.instantiate_model(show_summary=True)
taskB1.train()

########################################################################################################################


