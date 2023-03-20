import numpy as np
from sklearn.base import ClassifierMixin

from zenml.steps import step 


@step 
def evaluator(
    model: ClassifierMixin,
    x_test: np.ndarray,
    y_test: np.ndarray
) -> float:
    return model.score(x_test, y_test)