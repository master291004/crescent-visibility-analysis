from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def train_random_forest(
    X_train,
    y_train,
    n_estimators=100,
    max_depth=None,
    random_state=42,
    class_weight=None
):
    """
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training labels
    n_estimators : int
        Number of trees in the forest
    max_depth : int or None
        Maximum depth of each tree
    random_state : int
        Seed for reproducibility
    class_weight : dict or 'balanced'
        Class weighting strategy

    Returns
    -------
    model : RandomForestClassifier
        Trained Random Forest model
    """

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train,
    y_train,
    max_iter=1000,
    random_state=42,
    class_weight=None
):
    """
    Train a Logistic Regression classifier.

    Parameters
    ----------
    X_train : pandas.DataFrame
        Training feature matrix
    y_train : pandas.Series
        Training labels
    max_iter : int
        Maximum number of iterations
    random_state : int
        Seed for reproducibility
    class_weight : dict or 'balanced'
        Class weighting strategy

    Returns
    -------
    model : LogisticRegression
        Trained Logistic Regression model
    """

    model = LogisticRegression(
        max_iter=max_iter,
        random_state=random_state,
        class_weight=class_weight
    )

    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Generate class predictions.

    Parameters
    ----------
    model : sklearn classifier
        Trained model
    X_test : pandas.DataFrame
        Test feature matrix

    Returns
    -------
    y_pred : numpy.ndarray
        Predicted class labels
    """
    return model.predict(X_test)


def predict_proba(model, X_test):
    """
    Generate probability predictions for the positive class.

    Parameters
    ----------
    model : sklearn classifier
        Trained model
    X_test : pandas.DataFrame
        Test feature matrix

    Returns
    -------
    y_prob : numpy.ndarray
        Predicted probabilities for class 1
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    else:
        raise ValueError("This model does not support probability predictions.")
