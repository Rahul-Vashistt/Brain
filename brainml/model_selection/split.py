import numpy as np
import pandas as pd

def train_test_split(
    X, 
    y, 
    test_size=0.2, 
    shuffle=True, 
    random_state=None,
    return_indices=False, 
    as_frame=False
):
    """
    Split arrays or DataFrames into random train and test subsets.

    Parameters
    ----------
    X : array-like
        Features
    y : array-like
        Labels
    test_size : float
        Fraction of the dataset to use as test set
    shuffle : bool
        Whether to shuffle before splitting
    random_state : int or None
        Random seed for reproducibility

    stratify : array-like or None       -------->    Has not added yet!
        If not None, data is split in a stratified fashion using this as class labels 
        
    return_indices : bool
        If True, also return the indices used for splitting
    as_frame : bool
        If True and input was DataFrame/Series, returns output as DataFrame/Series

    Returns
    -------
    X_train, X_test, y_train, y_test [ + train_idx, test_idx if return_indices is True ]
    """

    X_is_df = isinstance(X,pd.DataFrame)
    y_is_series = isinstance(y, pd.Series)

    # Convert to arrays
    X = X.values if X_is_df else np.array(X)
    y = y.values if y_is_series else np.array(y)

    # Sanity check
    if X.shape[0] != y.shape[0]:
        raise ValueError("❌ X and y must have the same number of samples.")

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    # Shuffle the indices if needed
    if shuffle:
        rng = np.random.RandomState(random_state)
        rng.shuffle(indices)

    # Compute split point
    split = int(n_samples * (1 - test_size))
    train_indices = indices[:split]
    test_indices = indices[split:]

    # Perform the split
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    if as_frame and X_is_df:
        X_train = pd.DataFrame(X_train, columns=X.columns)
        X_test = pd.DataFrame(X_test, columns=X.columns)
    if as_frame and y_is_series:
        y_train = pd.Series(y_train, name=y.name)
        y_test = pd.Series(y_test, name=y.name)

    if return_indices:
        return X_train, X_test, y_train, y_test, train_indices, test_indices

    return X_train, X_test, y_train, y_test