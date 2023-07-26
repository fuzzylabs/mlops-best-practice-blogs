import pytest 

from steps import digits_data_loader, svc_trainer, evaluator

@pytest.fixture
def data(data_parameters):
    x_train, x_test, y_train, y_test = digits_data_loader(**data_parameters)

    return x_train, x_test, y_train, y_test


@pytest.fixture
def model(data):
    x_train, _, y_train, _ = data
    return svc_trainer(x_train, y_train)


def test_acc_within_range(data, model):
    _, x_test, _, y_test = data
    score = evaluator(model, x_test, y_test)

    # assert that the accuracy is 0.95 +/- 0.2
    assert score == pytest.approx(0.95, rel=0.2)