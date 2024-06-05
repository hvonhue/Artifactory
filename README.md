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
conda env create -f conda.yaml
```
