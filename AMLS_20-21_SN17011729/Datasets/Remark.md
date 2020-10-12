# Datasets for trainings/testing

This folder is included in the gitignore and therefore changes are not commited. Images used for training and testing of models are to placed in this directory.

## Requried Folder Structure

Once cloned/copied, the dataset should be copied into this directory to produce a file structure as shown below. If the file structure doesn't match the one shown below, the code may not run correctly.

```
Datasets
│   Remark.md
│
└───cartoon_set
│   │   labels.csv
│   │
│   └───img
│       │   0.png
│       │   1.png
│       │   ...
│   
└───celeba
│   │   labels.csv
│   │
│   └───img
│       │   0.jpg
│       │   1.jpg
│       │   ...
│   
└───cartoon_test_set
│   │   labels.csv
│   │
│   └───img
│       │   0.png
│       │   1.png
│       │   ...
│   
└───celeba_test
│   │   labels.csv
│   │
│   └───img
│       │   0.jpg
│       │   1.jpg
│       │   ...
```

## ```celeba``` dataset

This dataset is to be used for tasks A1 and A2. A seperate folder is provided for training and testing images. The images are of celebrity faces. The ```labels.csv`` file contains the labels required for the training and testing. The first column is the index. The second column is the corresponding file name. The third column is the gender ({-1, +1}). The last column is whether the person is smiling or not smiling ({-1, +1})

## ```cartoon_set``` dataset

This dataset is to be used for tasks B1 and B2. A seperate folder is provided for training and testing iamges. The images are of cartoon characters. The ```labels.csv``` file contains the lables required for training and testing. The first column is the index. The second column is eye colour (0-4), the third column is face shape (0-4), the last column is the corresponding file name. 