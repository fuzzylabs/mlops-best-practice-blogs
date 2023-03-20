import numpy as np 

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from zenml.steps import step, Output, BaseParameters

class DataParameters(BaseParameters):
    test_size = 0.2

@step 
def digits_data_loader(params: DataParameters) -> Output(
    x_train=np.ndarray, x_test=np.ndarray, y_train=np.ndarray, y_test=np.ndarray
):
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))

    x_train, x_test, y_train, y_test = train_test_split(
        data, 
        digits.target, 
        test_size=params.test_size, 
        shuffle=False,
        random_state=42
    )

    return x_train, x_test, y_train, y_test