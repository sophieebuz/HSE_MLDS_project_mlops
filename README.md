# HSE_MLDS_project_mlops

Status of github actions:  
[![pre-commit](https://github.com/sophieebuz/HSE_MLDS_project_mlops/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/sophieebuz/HSE_MLDS_project_mlops/actions/workflows/pre-commit.yml)

## Wine dataset
This dataset contains 13 different parameters for wine with 178 samples. The purpose of this wine dataset in scikit-learn is to predict the best wine class among 3 classes.

## Prerequisites before launch:
  - Python >=3.10
  - Poetry

## Initial configeration:
  1. run `git clone git@github.com:sophieebuz/HSE_MLDS_project_mlops.git`
  2. run `poetry install` to install a virtual environment
  3. run `poetry run pre-commit install`
  4. run `poetry run pre-commit run --all-files` to check the correct work of pre-commit

## Experiments reproduction
 - open a new terminal, go to the project folder, then enter the command `poetry run mlflow server --host 127.0.0.1 --port 8080` to raise mlflow service locally
 - `poetry run python train.py`
 - `poetry run python infer.py`
