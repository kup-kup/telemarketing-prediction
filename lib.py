import pandas as pd
import numpy as np

"""
This module contains utility functions for data preparation for the telemarketing prediction project.
All functions' names start with `tp_` to avoid naming conflicts and to indicate their purpose.
"""

def tp_simple_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    - transform `y` to boolean
    - drop duration column
    - transform pdays to previously_contacted (boolean)
    """
    df_transformed = df.copy()
    df_transformed['y'] = df_transformed['y'].map({'yes': True, 'no': False})
    df_transformed.drop(columns=['duration'], inplace=True)
    df_transformed['previously_contacted'] = df_transformed['pdays'].apply(lambda x: 0 if x == 999 else 1)
    df_transformed.drop(columns=['pdays'], inplace=True)
    return df_transformed

def tp_encode(df: pd.DataFrame, drop_first=True) -> pd.DataFrame:
    """
    - one-hot encode `job`, `marital`, `education`, `contact`, `month`, and `day_of_week`
    - ordinal encode `education`
    """
    df_transformed = df.copy()
    to_one_hot = [
        'marital', 'job', 'housing', 'loan', 'poutcome',
        'default', 'contact', 'month', 'day_of_week',
    ]
    df_transformed = pd.get_dummies(df_transformed, columns=to_one_hot, drop_first=drop_first)
    education_mapping = {
        "unknown": np.nan, # to be imputed later
        "illiterate": 0,
        "basic.4y": 1,
        "basic.6y": 2,
        "basic.9y": 3,
        "high.school": 4,
        "professional.course": 5,
        "university.degree": 6,
    }
    df_transformed['education'] = df_transformed['education'].map(education_mapping)
    return df_transformed

def tp_test_mcar(df: pd.DataFrame, alpha: float = 0.05, missing_cols: list|None = None) -> pd.DataFrame:
    """
    Return dataframe with columns and its p-value from MCAR test. 
    Perform this before encoding, as the tests rely on categorical variables.
    """
    res = []
    if missing_cols is None:
        missing_cols = ["job", "marital", "education", "default", "housing", "loan"]

    categorical_columns = df.select_dtypes(include=['object', 'str']).columns.tolist()
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Perform pairwise Chi-Square tests to check for MCAR
    from scipy.stats import chi2_contingency
    for miss_col in missing_cols:
        for cat_col in categorical_columns:
            if (miss_col == cat_col):
                continue
            contingency_table = pd.crosstab(df[cat_col], df[miss_col] == 'unknown')
            chi2, p, _, _ = chi2_contingency(contingency_table)

            res.append((miss_col, cat_col, p))
    
    # Perform Mann-Whitney U tests for numerical columns
    from scipy.stats import mannwhitneyu
    for miss_col in missing_cols:
        for num_col in numerical_columns:
            if (miss_col == num_col):
                continue
            group_with_missing = df[df[miss_col] == 'unknown'][num_col].dropna()
            group_without_missing = df[~(df[miss_col] == 'unknown')][num_col].dropna()

            if len(group_with_missing) == 0 or len(group_without_missing) == 0:
                continue

            stat, p = mannwhitneyu(group_with_missing, group_without_missing, alternative='two-sided')

            res.append((miss_col, num_col, p))
    
    return pd.DataFrame(res, columns=['missing_column', 'tested_column', 'p_value'])

# TEST FIRST
def tp_multinomial_impute(df: pd.DataFrame, missing_cols=None, random_state=42) -> pd.DataFrame:
    """
    Impute missing categorical values using multinomial logistic regression.
    Does not use LabelEncoder on the original data - only internally for fitting.
    
    Args:
        df: Input dataframe
        missing_cols: List of columns with missing values to impute
        random_state: Random state for reproducibility
    
    Returns:
        DataFrame with imputed values
    """
    from sklearn.linear_model import LogisticRegression
    if missing_cols is None:
        missing_cols = ["job", "marital", "default" ,"housing", "loan"]
    
    df_imputed = df.copy()
    
    for col in missing_cols:
        # Identify rows with missing values
        missing_mask = (df_imputed[col] == 'unknown') | df_imputed[col].isnull()
        
        if missing_mask.sum() == 0:
            continue
        
        # Prepare training data (rows without missing values)
        train_mask = ~missing_mask
        X_train = df_imputed.loc[train_mask, df_imputed.columns != col].copy()
        y_train = df_imputed.loc[train_mask, col].copy()
        
        # Encode categorical features in X_train
        X_train_encoded = pd.get_dummies(X_train, drop_first=False)
        
        # Prepare prediction data (rows with missing values)
        X_pred = df_imputed.loc[missing_mask, df_imputed.columns != col].copy()
        X_pred_encoded = pd.get_dummies(X_pred, drop_first=False)
        
        # Align columns (X_pred might have different dummy columns)
        X_pred_encoded = X_pred_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
        
        # Fit multinomial logistic regression
        model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train_encoded, y_train)
        
        # Predict missing values
        predictions = model.predict(X_pred_encoded)
        df_imputed.loc[missing_mask, col] = predictions
    
    return df_imputed