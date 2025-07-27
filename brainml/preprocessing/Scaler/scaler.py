import numpy as np
import pandas as pd
import logging
from typing import Union

logging.basicConfig(level=logging.INFO)

class StandardScaler:
    '''
    Standardize features by removing the mean and scaling it to unit variance.
    Formula = (x - mean) / std
    '''
    def __init__(self, with_mean: bool=True, with_std: bool=True, df: bool=False):
        self.with_mean: bool = with_mean
        self.with_std: bool = with_std
        self.mean_: Union[np.array, None] = None
        self.std_: Union[np.array, None] = None
        self.fitted: bool = False
        self.df: bool = df  # If true, output would be a Dataframe if the input was a Dataframe
        self.is_dataframe: bool = False
        self.feature_names_: str = None
        self.n_features_: int = None


    def fit(self, X: Union[np.array, pd.DataFrame]) -> "StandardScaler":
        '''
        Compute the mean and std to be used for later scaling.

        Parameters
        ----------
        X : np.ndarray OR pd.Dataframe of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        Returns
        -------
        self : object
            Returns instance itself
        '''
        if X.ndim !=2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")
        
        self.is_dataframe = isinstance(X, pd.DataFrame)
        
        self.feature_names_ = X.columns if self.is_dataframe else None
        self.n_features_ = len(X.columns) if self.is_dataframe else X.shape[1]
        self.df = True if self.is_dataframe else False
        X = np.asarray(X)

        self.mean_ = np.mean(X, axis=0) if self.with_mean else np.zeros(X.shape[1])
        self.std_ = np.std(X, axis=0) if self.with_std else np.ones(X.shape[1])

        if self.with_std and np.any(self.std_ == 0):
            logging.warning("Zero std detected; those features won't be scaled!")
            self.std_[self.std_ == 0] = 1.0

        self.fitted = True
        return self
    

    def transform(self, X: Union[np.array, pd.DataFrame]) -> Union[np.array, pd.DataFrame]:
        '''
        Perform standardization by centering and scaling.

        Parameters
        ----------
        X : np.ndarray OR pd.Dataframe of shape (n_samples, n_features)
            The data which will be centered and scaled.

        Returns
        -------
        X : (np.array or pd.Dataframe)
            np.ndarray of shape (n_samples, n_features) 
                            - OR -
            pd.DataFrame of shape (n_samples, n_features)
        '''
        if not self.fitted:
            raise RuntimeError("Must call 'fit' before 'transform'")
        X = np.asarray(X)

        if not X.shape[1] == self.mean_.shape[0]:
            raise ValueError(f"Shape mismatch: Expected shape to be {self.mean_.shape[0]} but got {X.shape[1]}")
        else:
            if self.with_mean:
                X = X - self.mean_
            if self.with_std:
                X = X / self.std_
            if self.is_dataframe and self.df:
                return pd.DataFrame(X, columns=self.feature_names_)
            else:
                if not self.is_dataframe and self.df:
                    logging.info("Input was not a pandas DataFrame, hence the output will be numpy array!")
                    return X
    

    def fit_transform(self, X: Union[np.array, pd.DataFrame]) -> Union[np.array, pd.DataFrame]:
        '''
        Fit the data, then transform it.

        Parameters
        ----------
        X : np.ndarray OR pd.Dataframe of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation.

        Returns
        -------
        X : (np.array or pd.Dataframe)
            np.ndarray of shape (n_samples, n_features) 
                            - OR -
            pd.DataFrame of shape (n_samples, n_features)
        '''
        return self.fit(X).transform(X)
    
    
    def inverse_transform(self, X_scaled: Union[np.array, pd.DataFrame]) -> Union[np.array, pd.DataFrame]:
        '''
        Scale back the data to its original state.

        Parameters
        ----------
        X_scaled : np.ndarray OR pd.Dataframe of shape (n_samples, n_features)
                   Scaled data to be Unscaled.

        Returns
        -------
        X : (np.array or pd.Dataframe)
            np.ndarray of shape (n_samples, n_features),
            pd.DataFrame of shape (n_samples, n_features)
        '''
        if not self.fitted:
            raise RuntimeError("Must call 'fit' before 'inverse_transform'")
        X = np.asarray(X_scaled)
        if not X.shape[1] == self.mean_.shape[0]:
            raise ValueError(f"Shape mismatch: Expected shape to be {self.mean_.shape[0]} but got {X.shape[1]}")
        else:
            if self.with_std:
                X = X * self.std_
            if self.with_mean:
                X = X + self.mean_
            if self.is_dataframe and self.df:
                return pd.DataFrame(X, columns=self.feature_names_)
            else:
                if not self.is_dataframe and self.df:
                    logging.info("Input was not a pandas DataFrame, hence the output will be numpy array!")
                    return X
    
    def __repr__(self):
        return (f"StandardScaler(with_mean={self.mean_}, with_std={self.std_},"
                f"fitted={self.fitted}, n_features={None if self.mean_ is None else self.n_features_})")
    