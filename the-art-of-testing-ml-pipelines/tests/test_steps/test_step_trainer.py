import pytest 
from contextlib import nullcontext as does_not_raise

from sklearn.utils.validation import check_is_fitted
from sklearn.svm import SVC 

from steps import digits_data_loader
from steps import svc_trainer


@pytest.fixture
def data(data_parameters):
    x_train, _, y_train, _ = digits_data_loader(**data_parameters)

    return x_train, y_train


def test_model_is_fitted(data):
    x_train, y_train = data
    model = svc_trainer(x_train, y_train)

    with does_not_raise():
        check_is_fitted(model)


def test_gamma_parameter(data):
    x_train, y_train = data 

    model = svc_trainer(x_train, y_train)
    assert model._gamma == 0.001
    

def test_correct_type(data):
    x_train, y_train = data 

    assert isinstance(
        svc_trainer(x_train, y_train),
        SVC
    )