import numpy as np 

from sklearn.base import ClassifierMixin 
from sklearn.svm import SVC 

from zenml.steps import step


@step 
def svc_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray
) -> ClassifierMixin:
    """"""
    model = SVC(
        gamma=0.001,
        random_state=42
    )
    model.fit(x_train, y_train)
    return model