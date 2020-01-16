# 2019 Data Science Bowl Competition (ML2019 Final Project)

## Team Member

* b05602052 電機四 舒泓諭
* r06522709 機械碩三 鄭呈毅
* r07543069 應力碩二 潘俊霖



## Required Toolkits

* python 3.7
* numpy 1.13.3
* pandas 0.6.1
* matplotlib 3.0.3
* xgboost 0.90
* shap 0.31.0
* tqdm 4.36.1
* scipy 1.2.1
* lightgbm 2.3.0
* sklearn 0.0
* tensorflow 2.0.0
* keras 2.3.1

## File Description

* `train.py` is used for training.
* `test.py` is used for testing.
* `train.sh` is used for training.
* `test.sh` is used for reproducing score on kaggle.

## How to Reproduce

1. Run `mkdir /kaggle/input/data-science-bowl-2019/` and download all file in kaggle into the `/src/kaggle/input/data-science-bowl-2019/` 
2. Run `bash test.sh ` under `src/` directory.
3. `submission.csv` is the reproducing prediction file.

## Directory 

```
   root
   +-- src
   |   +-- train.py
   |   +-- test.py
   +-- Report.pdf
   +-- README.md
   +-- train.sh
   +-- test.sh
```
