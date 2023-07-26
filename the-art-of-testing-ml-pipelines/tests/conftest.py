import pytest 


@pytest.fixture
def data_parameters():
    params = {
        'test_size': 0.2
    }
    return params