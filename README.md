# HSE_MLDS_project_mlops

Status of github actions:  
[![pre-commit](https://github.com/sophieebuz/HSE_MLDS_project_mlops/actions/workflows/pre-commit.yml/badge.svg)](https://github.com/sophieebuz/HSE_MLDS_project_mlops/actions/workflows/pre-commit.yml)

## Prerequisites before launch:
  - Python >=3.10
  - Poetry

## Initial configeration:
  1. run `git clone git@github.com:sophieebuz/HSE_MLDS_project_mlops.git`
  2. run `poetry install` to install a virtual environment
  3. run `poetry run pre-commit install`
  4. run `poetry run dvc pull` to get all datasets and models (may take a few dozens of minutes)

## Experiments reproduction
 - `poetry run train.py`
 - `poetry run infer.py`
