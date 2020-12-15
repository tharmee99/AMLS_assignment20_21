import os

from A1.a1_test_train import A1
from A2.a2_test_train import A2
from B1.b1_test_train import B1
from B2.b2_test_train import B2


CARTOON_DATASET_DIR = os.path.join("Datasets", "cartoon_set")
CELEB_DATASET_DIR = os.path.join("Datasets", "celeba")
# Task A1
########################################################################################################################

print("\n\n-------------Task A1 - Gender Classification-------------\n\n")

taskA1 = A1(CELEB_DATASET_DIR, os.path.join("A1", "temp"))
taskA1.train()
taskA1.test()
taskA1.print_results()

########################################################################################################################
# Task A2
########################################################################################################################

print("\n\n-------------Task A2 - Emotion Classification-------------\n\n")

taskA2 = A2(CELEB_DATASET_DIR, os.path.join("A2", "temp"))
taskA2.train()
taskA2.test()
taskA2.print_results()
taskA2.remove_ambiguous_samples(test=True)
taskA2.test()
print("Accuracy results discarding undetected faces(dlib)")
taskA2.print_results()

########################################################################################################################
# Task B1
########################################################################################################################

print("\n\n-------------Task B1 - Face Shape Classification-------------\n\n")

taskB1 = B1(CARTOON_DATASET_DIR, os.path.join("B1", "temp"))
taskB1.train()
taskB1.test()
taskB1.print_results()

########################################################################################################################
# Task B2
########################################################################################################################

print("\n\n-------------Task B2 - Eye Colour Classification-------------\n\n")


########################################################################################################################

