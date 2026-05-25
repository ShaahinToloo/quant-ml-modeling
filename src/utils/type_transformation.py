import numpy as np
from sklearn.utils import check_array


def transform_data(X):
    """
    Only send numeric features,
    X must not contain One-Hot Encoded or Embeddings.
    """
    return check_array(X, accept_sparse=True, dtype=np.float64, ensure_all_finite=False)


def to_list(X):
    try:
        return X.tolist()
    except Exception:
        return list(X)
