from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from zenml import step


@step 
def digits_data_loader(test_size: float):
    digits = load_digits()
    data = digits.images.reshape((len(digits.images), -1))

    x_train, x_test, y_train, y_test = train_test_split(
        data, 
        digits.target, 
        test_size=test_size,
        shuffle=False,
        random_state=42
    )

    return x_train, x_test, y_train, y_test