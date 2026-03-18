import pandas as pd
import numpy as np

"""
This module contains utility functions for data preparation for the telemarketing prediction project.
All functions' names start with `tp_` to avoid naming conflicts and to indicate their purpose.
"""

#############################################################
# Transformation ############################################
#############################################################

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

#############################################################
# Imputation ################################################
#############################################################
class TPMultinomialImputer:
    def __init__(self, missing_cols=None, random_state=42, logreg_opts=None):
        self.missing_cols = missing_cols if missing_cols is not None else ["job", "marital", "housing", "loan", "education"]
        self.random_state = random_state
        self.logreg_opts = logreg_opts if logreg_opts is not None else {
            'solver': 'lbfgs',
            'max_iter': 100,
        }

        self.logreg_models = {}
        self.feature_columns = {}  # Store column names for each model
    
    def fit(self, df: pd.DataFrame):
        self.df = df.copy()
        for col in self.missing_cols:
            if col == 'education':
                continue

            missing_mask = (self.df[col] == 'unknown') | self.df[col].isnull()
            
            if missing_mask.sum() == 0:
                continue
            
            X_train = self.df.loc[~missing_mask, ~self.df.columns.isin(self.missing_cols)].copy()
            y_train = self.df.loc[~missing_mask, col].copy()
            X_train_encoded = pd.get_dummies(X_train, drop_first=False)
            
            # Store the column names for later use
            self.feature_columns[col] = X_train_encoded.columns.tolist()
            
            from sklearn.linear_model import LogisticRegression
            model = LogisticRegression(**self.logreg_opts, random_state=self.random_state)
            model.fit(X_train_encoded, y_train)
            self.logreg_models[col] = model
    
    def transform(self, df: pd.DataFrame, verbose=False) -> pd.DataFrame:
        df_imputed = df.copy()
        for col in self.missing_cols:
            if col == 'education':
                continue

            missing_mask = (df_imputed[col] == 'unknown') | df_imputed[col].isnull()
            
            if missing_mask.sum() == 0:
                continue
            
            X_pred = df_imputed.loc[missing_mask, ~df_imputed.columns.isin(self.missing_cols)].copy()
            X_pred_encoded = pd.get_dummies(X_pred, drop_first=False)
            # Reindex to match training columns
            X_pred_encoded = X_pred_encoded.reindex(columns=self.feature_columns[col], fill_value=0)
            
            model = self.logreg_models[col]
            predictions = model.predict(X_pred_encoded)
            df_imputed.loc[missing_mask, col] = predictions
        
        if verbose:
            for col in self.missing_cols:
                if col == 'education':
                    continue
                print(f"\nImputation summary for '{col}':")
                summary = pd.DataFrame({
                    'Before': df[col].value_counts(),
                    'Percent Before': (df[col].value_counts() / len(df) * 100).round(2),
                    'After': df_imputed[col].value_counts()
                })
                imputed = summary['After'] - summary['Before']
                summary['Imputed'] = imputed
                summary['Percent Imputed'] = (imputed / imputed.sum()).round(2)
                summary.sort_values('Before', ascending=False, inplace=True)
                print(summary)

        return df_imputed
    
class TPKNNImputer:
    def __init__(self, missing_col=None, mapping=None, knn_opts=None):
        if missing_col is None:
            self.missing_col = "education"
            self.mapping = {
                "unknown": np.nan, # to be imputed later
                "illiterate": 0,
                "basic.4y": 1,
                "basic.6y": 2,
                "basic.9y": 3,
                "high.school": 4,
                "professional.course": 5,
                "university.degree": 6,
            }
        else:
            self.missing_col = missing_col
            self.mapping = mapping
        self.mapping_inv = {v: k for k, v in self.mapping.items()} # type: ignore
        self.knn_opts = knn_opts if knn_opts is not None else {'n_neighbors': 5}
        self.imputer = None
        self.feature_columns = None
    
    def fit(self, df: pd.DataFrame):
        assert isinstance(self.mapping, dict)
        from sklearn.impute import KNNImputer
        
        df_prep = df.copy()
        df_prep[self.missing_col] = df_prep[self.missing_col].map(self.mapping)
        
        X = df_prep.copy()
        X_encoded = pd.get_dummies(X, drop_first=False)
        
        self.feature_columns = X_encoded.columns.tolist()
        self.imputer = KNNImputer(**self.knn_opts) # type: ignore
        self.imputer.fit(X_encoded)

    def transform(self, df: pd.DataFrame, verbose=False) -> pd.DataFrame:
        assert isinstance(self.mapping, dict)
        assert self.imputer is not None
        
        df_imputed = df.copy()
        df_imputed[self.missing_col] = df_imputed[self.missing_col].map(self.mapping)
        
        # Identify rows with missing values before imputation
        missing_mask = df_imputed[self.missing_col].isnull()
        
        # Prepare data for imputation
        X = df_imputed.copy()
        X_encoded = pd.get_dummies(X, drop_first=False)
        X_encoded = X_encoded.reindex(columns=self.feature_columns, fill_value=0)
        
        # Apply imputation
        X_imputed = self.imputer.transform(X_encoded)
        X_imputed_df = pd.DataFrame(X_imputed, columns=self.feature_columns, index=df.index)
        
        # Update the original dataframe with imputed values
        df_imputed.loc[missing_mask, self.missing_col] = X_imputed_df.loc[missing_mask, self.missing_col]
        
        # Convert back to original mapping
        df_imputed[self.missing_col] = df_imputed[self.missing_col].round().astype(int).map(self.mapping_inv)
        
        # Print imputation summary
        if verbose:
            print(f"\nImputation summary for '{self.missing_col}':")
            summary = pd.DataFrame({
                'Before': df[self.missing_col].value_counts(),
                'Percent Before': (df[self.missing_col].value_counts() / len(df) * 100).round(2),
                'After': df_imputed[self.missing_col].value_counts()
            })
            imputed = summary['After'] - summary['Before']
            summary['Imputed'] = imputed
            summary['Percent Imputed'] = (imputed / imputed.sum()).round(2)
            summary.sort_values('Before', ascending=False, inplace=True)
            print(summary)
        
        return df_imputed