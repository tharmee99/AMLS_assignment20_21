import os
import pickle

from A1.a1_test_train import A1
from itertools import product

os.chdir('..')

# If True, metrics are computed, if False, metrics are viewed/analyzed
GET_CV_METRICS = False

CELEB_DATASET_DIR = os.path.join("Datasets", "celeba")

taskA1 = A1(CELEB_DATASET_DIR, os.path.join("A1", "temp"))

if GET_CV_METRICS:

    hyperparameter_vals = {
        'cnn_kernel_size': [(3, 3), (5, 5)],
        'cnn_filters': [8, 32],
        'layer_activation': ['relu', 'sigmoid'],
        'dense_layer_size': [100, 1000],
        'colour': [True, False]
    }

    hyperparameter_keys = list(hyperparameter_vals.keys())

    metrics_list = []

    all_combs = list(product(*hyperparameter_vals.values()))

    print(len(all_combs))

    for comb in all_combs:
        taskA1.cnn_kernel_size = comb[0]
        taskA1.cnn_filters = comb[1]
        taskA1.layer_activation = comb[2]
        taskA1.dense_layer_size = comb[3]
        taskA1.colour = comb[4]

        print(comb)

        this_metric = {
            'hyperparameters': comb,
            'metrics': taskA1.compute_kfold_cv_score()
        }

        metrics_list.append(this_metric)

    with open(os.path.join(taskA1.temp_dir, 'hyperparameter_metrics.pkl'), 'wb') as f:
        pickle.dump(metrics_list, f, pickle.HIGHEST_PROTOCOL)
else:

    with open(os.path.join(taskA1.temp_dir, 'hyperparameter_metrics.pkl'), 'rb') as f:
        all_metrics = pickle.load(f)

    for data in all_metrics:
        data['avg_accuracy'] = sum(data['metrics']['accuracies']) / len(data['metrics']['accuracies'])
        print(data)
pass