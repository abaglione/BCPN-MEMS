# Predicting Medication Adherence from Smart Pill Bottle Data

The code contained in this repository was used for my Ph.D. thesis work on predicting medication adherence from smart sensor streams (e.g., smart pill bottle caps). All code was authored by [Dr. Anna Baglione](https://github.com/abaglione), with adaptations from prior projects with Dr. Lihua Cai, [Dr. Sonia Baee](https://github.com/soniabaee), and [Tyler Spears](https://github.com/TylerSpears/). 

Data were collected by Dr. Kristen Wells and colleagues and were used with permission.

This README template was adapted with permission from one of [Tyler Spears](https://github.com/TylerSpears/)' templates.

## Installation
### Required Packages
This project requires the python packages:

- jupyter
- pandas
- scikit-learn
- matplotlib
- xgboost
- shap
- ...and many others

### Environment Creation
We recommend using anaconda (or variants such as miniconda or mamba) to create a new environment from the environment.yml file:

```
conda env create -f environment.yml
```

### pre-commit Hooks
This repository relies on pre-commit to run basic cleanup and linting utilities before a commit can be made. Hooks are defined in the .pre-commit-config.yaml file. To set up pre-commit hooks:

``` 
# If pre-commit is not already in your conda environment
mamba install -c conda-forge pre-commit
pre-commit install

# (Optional) run against all files
pre-commit run --all-files
```

#### nbstripout Hook Description
The nbstripout hook strips jupyter/ipython notebooks (*.ipynb files) of output and metadata. nbstripout is especially important for human subjects projects, in which keeping data (even anonymized data) out of the public cloud is necessary.

This project uses nbstripout as a pre-commit hook (see https://github.com/kynan/nbstripout#using-nbstripout-as-a-pre-commit-hook), but this causes your local working version to be stripped of output.
