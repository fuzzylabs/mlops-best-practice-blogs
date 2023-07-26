import pytest
from math import ceil
from steps import digits_data_loader

EXPECTED_DATA_LENGTH = 1797


def test_correct_data_amount(data_parameters):
    x_train, x_test, y_train, y_test = digits_data_loader(**data_parameters)

    assert (len(x_train) + len(x_test)) == EXPECTED_DATA_LENGTH


def test_correct_split(data_parameters):
    x_train, x_test, y_train, y_test = digits_data_loader(**data_parameters)

    expected_size_test = ceil(EXPECTED_DATA_LENGTH * data_parameters['test_size'])
    expected_size_train = int(EXPECTED_DATA_LENGTH * (1 - data_parameters['test_size']))

    assert len(x_test) == expected_size_test
    assert len(y_test) == expected_size_test

    assert len(x_train) == expected_size_train
    assert len(y_train) == expected_size_train