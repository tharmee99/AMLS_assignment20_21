import os

from A1.a1_test_train import A1
from A2.a2_test_train import A2
from B1.b1_test_train import B1
from B2.b2_test_train import B2


CARTOON_DATASET_DIR = os.path.join("Datasets", "cartoon_set")
CELEB_DATASET_DIR = os.path.join("Datasets", "celeba")
# Task A1
########################################################################################################################

# taskA1 = A1(CELEB_DATASET_DIR, os.path.join("A1", "temp"))
# taskA1.train()
# taskA1.test()
# taskA1.print_results()

# taskA1.model.get_filter1_output(taskA1.X_test[0:3])
# taskA1.model.get_filter2_output(taskA1.X_test[0:3])

########################################################################################################################
# Task A2
########################################################################################################################



########################################################################################################################
# Task B1
########################################################################################################################

taskB1 = B1(CARTOON_DATASET_DIR, os.path.join("B1", "temp"))
taskB1.train()
taskB1.test()
taskB1.print_results()

########################################################################################################################
# Task B2
########################################################################################################################



########################################################################################################################

