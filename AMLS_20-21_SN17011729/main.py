import os
import time

from A1.a1_test_train import A1
from A2.a2_test_train import A2
from B1.b1_test_train import B1
from B2.b2_test_train import B2


CARTOON_DATASET_DIR = os.path.join("Datasets", "cartoon_set")
CELEB_DATASET_DIR = os.path.join("Datasets", "celeba")
CARTOON_ALT_DATASET_DIR = os.path.join("Datasets", "cartoon_set_test")
CELEB_ALT_DATASET_DIR = os.path.join("Datasets", "celeba_test")

# From timing the execution on my machines, this script should take less than 10 minutes to execute without any cache
start_time = time.time()

# Task A1
########################################################################################################################

print("\n\n-------------Task A1 - Gender Classification-------------\n\n")

taskA1 = A1(CELEB_DATASET_DIR, os.path.join("A1", "temp"))
taskA1.train()
taskA1.test()

X_test_A1, y_test_A1 = taskA1.build_design_matrix(dataset_dir=CELEB_ALT_DATASET_DIR)
y_test_A1 = taskA1.enc.transform(y_test_A1.reshape(-1, 1))
_, alt_test_a1 = taskA1.test(X_test=X_test_A1, y_test=y_test_A1)

########################################################################################################################
# Task A2
########################################################################################################################

print("\n\n-------------Task A2 - Emotion Classification-------------\n\n")

taskA2 = A2(CELEB_DATASET_DIR, os.path.join("A2", "temp"))
taskA2.train()
taskA2.test()

X_test_A2, y_test_A2 = taskA2.build_design_matrix(dataset_dir=CELEB_ALT_DATASET_DIR)
_, alt_test_a2 = taskA2.test(X_test=X_test_A2, y_test=y_test_A2)

# ########################################################################################################################
# # Task B1
# ########################################################################################################################

print("\n\n-------------Task B1 - Face Shape Classification-------------\n\n")

taskB1 = B1(CARTOON_DATASET_DIR, os.path.join("B1", "temp"))
taskB1.train()
taskB1.test()

X_test_B1, y_test_B1 = taskB1.build_design_matrix(dataset_dir=CARTOON_ALT_DATASET_DIR)
_, alt_test_b1 = taskB1.test(X_test=X_test_B1, y_test=y_test_B1)

# ########################################################################################################################
# # Task B2
# ########################################################################################################################

print("\n\n-------------Task B2 - Eye Colour Classification-------------\n\n")

taskB2 = B2(CARTOON_DATASET_DIR, os.path.join("B2", "temp"))
taskB2.train()
taskB2.test()

X_test_B2, y_test_B2 = taskB2.build_design_matrix(dataset_dir=CARTOON_ALT_DATASET_DIR)
X_test_B2 = taskB2.lv_selector.transform(X_test_B2)
_, alt_test_b2 = taskB2.test(X_test=X_test_B2, y_test=y_test_B2)

# ########################################################################################################################
# # Summary
# ########################################################################################################################

time_taken = time.time() - start_time
# print()
# print(time_taken)
# print("\n\n\n-------------CONFUSION MATRICES-------------\n")
#
# print(taskA1.model.confusion_matrix)
# print(taskA2.model.confusion_matrix)
# print(taskB1.model.confusion_matrix)
# print(taskB2.model.confusion_matrix)

print("\n\n\n-------------RESULTS SUMMARY-------------\n")

print("{}:".format(taskA1.name))
taskA1.print_results()
print("Test Accuracy (alt):\t{}".format(alt_test_a1))
print("{}:".format(taskA2.name))
taskA2.print_results()
print("Test Accuracy (alt):\t{}".format(alt_test_a2))
print("{}:".format(taskB1.name))
taskB1.print_results()
print("Test Accuracy (alt):\t{}".format(alt_test_b1))
print("{}:".format(taskB2.name))
taskB2.print_results()
print("Test Accuracy (alt):\t{}".format(alt_test_b2))
