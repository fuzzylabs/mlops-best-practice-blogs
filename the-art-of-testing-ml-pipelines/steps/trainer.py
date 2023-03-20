import numpy as np 

from sklearn.svm import SVC 
from zenml.steps import step

@step 
def svc_trainer(
    x_train: np.ndarray,
    y_train: np.ndarray
) -> SVC:
    model = SVC(
        gamma=0.001,
        random_state=42
    )
    model.fit(x_train, y_train)
    return model