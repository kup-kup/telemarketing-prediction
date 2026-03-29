# telemarketing-prediction

This repository contains a machine learning project focused on predicting the success of telemarketing campaigns. The project utilizes a [dataset](#citation) of telemarketing calls, which includes various features such as call duration, customer demographics, and previous interactions. The goal is to build a predictive model that can accurately classify whether a telemarketing call will result in a successful outcome (e.g., a sale or positive response).

**Setting Up the Environment**

1. Run `setup.ipynb`. This notebook will install the necessary dependencies for the project. Make sure to have Python and Jupyter Notebook installed on your system before running the setup.
2. Download the dataset from [here](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) and place `back-additional-full.csv` it in the `data` folder of the repository.

# Project Structure

## Main Notebooks
- `01preparation.ipynb`: This notebook contains the data preprocessing steps, including data cleaning/imputation, feature engineering, and transformation.
- `02model_main.ipynb`: This notebook contains the model training, evaluation, tuning, and interpretation steps.
- `03model_analysis.ipynb`: This notebook contains the survival analysis (using Kaplan-Meier estimator and Cox Proportional Hazards model) and sensitivity analysis for the Xgboost model.

## Directories
- `data/`: Contains the raw dataset file (`bank-additional-full.csv`).
- `output/`: Contains the output files generated from the notebooks.
- `preprocessor/`: Contains the saved preprocessor files (e.g., encoders, scalers) to be used for inference.
- `processed-data/`: Contains the processed dataset files generated from the preparation notebook.
- `model/`: Contains the saved machine learning model files to be used for inference.

# Data Description

## Input variables:
**bank client data:**
- age (numeric)
- job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
- marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
- education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
- default: has credit in default? (categorical: "no","yes","unknown")
- housing: has housing loan? (categorical: "no","yes","unknown")
- loan: has personal loan? (categorical: "no","yes","unknown")

**related with the last contact of the current campaign:**   
- contact: contact communication type (categorical: "cellular","telephone") 
- month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
- day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
- duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

**other attributes:**
- campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
- pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
- previous: number of contacts performed before this campaign and for this client (numeric)
- poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")

**social and economic context attributes:**
- emp.var.rate: employment variation rate - quarterly indicator (numeric)
- cons.price.idx: consumer price index - monthly indicator (numeric)     
- cons.conf.idx: consumer confidence index - monthly indicator (numeric)     
- euribor3m: euribor 3 month rate - daily indicator (numeric)
- nr.employed: number of employees - quarterly indicator (numeric)

## Output variable (desired target):
- y - has the client subscribed a term deposit? (binary: "yes","no")

## Missing Attribute Values
There are several missing values in some categorical attributes, all coded with the "unknown" label. These missing values can be treated as a possible class label or using deletion or imputation techniques. 

# Citation
[1] Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306.