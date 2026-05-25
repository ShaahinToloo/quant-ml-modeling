import os

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier


class MLPClassifierBinary:
    def __init__(
        self,
        hidden_layers=(64, 32),
        activation="tanh",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=200,
        random_state=42,
        verbose=True,
        **kwargs
    ):
        """
        Baseline MLP binary classifier using sklearn.
        """

        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver=solver,
            alpha=alpha,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            verbose=verbose,
            **kwargs
        )

        self.is_fitted = False

    def fit(self, X_train, y_train):
        """
        Train the MLP model.
        X_train: np.ndarray of shape (n_samples, n_features)
        y_train: np.ndarray of shape (n_samples,)
        """
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        return self

    def predict(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict(X)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet.")

        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        return {
            "accuracy": acc,
            "f1": f1,
            "precision": prec,
            "recall": rec,
            "roc_auc": roc_auc,
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)
        self.is_fitted = True

    def get_model(self):
        return self.model
