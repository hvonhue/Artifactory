# Artifactory

This is the repository to the master thesis "Deep Learning Approaches for the Detection of Ramping Artifacts in Schedule Deviation Data in Belgium".

# Quick start
 The quickest way to interact with the code is to open a github codespace. Due to limited storage possibilities the training on all datasets cannot be performed in the codespace.
 It was also not possible to download all datasets to the github repository. Few sample datasets can be found in the data/processed folder.

 If you have the files of the industry testing data, please upload them (both) in the folder data/real.

## Requirements

- python 3.10


## Setup

Github codespaces:
You can run most functionalities in a github codespace (i.e. testing all presented models and recreating the measures shown)
Due to limited storage, the datasets can (or only) temporarily be loaded into the codespace.

Locally:
The requrements are stored in a conda.yaml file. To create an environment containing the dependencies, navigate to the folder containing the yaml file and run:
```console
conda env create -f conda.yml
```


## Usage
All visualization and the evaluation od the models can be done in the jupyter notebooks stored in notebooks.
Notebooks with names that start with 1_ are used to preprocess the according datasets with respect to normalization and splitting in train and validation. If the pre-downloaded datasets in data/processed are used this preprocessing has already been performed

Notebooks with 2_ contain visualizations of the artifacts in the target data and the synthetic datasets. 2_realAritfacts_example cannot be fully executed since the original data series cannot be provided.

The notebooks 3_ rank the synthetic datasets with respect to the target data, for the schedule deviation. The ranking for the system imbalance is performed in A_rank_datasets_francois

The notebook 4_test_all contains the main evaluation of all models.
4_ablation_fpboost performs the ablation about the loss with the false positive penalty.


All trained models are saved in the models folder. All can be retrained using the according train file:

FCN sliding Window:
```console
python train_FCN_slidingWindow.py --input-path ../data/processed --val-path ../data/val_files/val_SW_noCiECGT512.pkl --output-path ../data/output
```
FCN Mask:
```console
python train_FCN_mask.py --input-path ../data/processed --val-path ../data/val_files/val_mask_noCiECGT512.pkl --output-path ../data/output
```

Transformer only sliding Window:
```console
python train_TransOnly_slidingWindow.py --input-path ../data/processed --val-path ../data/val_files/val_SW_noCiECGT512.pkl --output-path ../data/output
```
Transformer only Mask:
```console
python train_TransOnly_mask.py --input-path ../data/processed --val-path ..data/val_files/val_mask_noCiECGT512.pkl --output-path ../data/output
```

FCN dense sliding Window:
```console
python train_FCN_dense_slidingWindow.py --input-path ../data/processed --val-path ../data/val_files/val_SW_noCiECGT512.pkl --output-path ../data/output
```
FCN dense Mask:
```console
python train_FCN_dense_mask.py --input-path ../data/processed --val-path ../data/val_files/val_mask_noCiECGT512.pkl --output-path ../data/output
```

FCN transformer sliding Window:
```console
python train_sliding_window.py --input-path ../data/processed --val-path ../data/val_files/val_SW_noCiECGT512.pkl --output-path ../data/output
```
FCN transformer Mask:
```console
python train_mask.py --input-path ../data/processed --val-path ../data/val_files/val_mask_noCiECGT512.pkl --output-path ../data/output
```

Changes to the architecture can be performed in the top section of the according file. This includes changing the parameters of the FCN to the adapted FCN.

The Isolation Forest ist trained and evaluated in the notebook IFOREST.

## Data

The Data included in this repository is taken from the UCR Time Series Classification Archive

DAU, Hoang Anh, et al. The UCR time series archive. IEEE/CAA Journal of Automatica Sinica, 2019, 6. Jg., Nr. 6, S. 1293-1305.
