import pytest 
import os
import logging

from math import ceil
from sklearn.svm import SVC 
from zenml.logger import disable_logging
from zenml.post_execution import get_run

from pipelines import training_pipeline
from steps import (
    digits_data_loader, 
    svc_trainer, 
    evaluator
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPECTED_LENGTH = 1797

@pytest.fixture
def full_pipeline():
    return training_pipeline


@pytest.fixture
def pipeline_configuration_path():
    return BASE_DIR + '/test_pipeline_config.yaml'


@pytest.fixture()
def pipeline_run(full_pipeline, pipeline_configuration_path):
    pipeline = full_pipeline(
        digits_data_loader(),
        svc_trainer(),
        evaluator()
    )

    with disable_logging(log_level=logging.INFO):
        pipeline.run(config_path=pipeline_configuration_path, unlisted=True)

    return get_run(name='test-pipeline')


def test_pipeline_executes(pipeline_run):
    evaluator_result = pipeline_run.get_step(step='evaluator').output.read()

    assert evaluator_result == pytest.approx(0.95, rel=0.2)


def test_pipeline_loads_and_splits_correctly(pipeline_run, data_parameters):
    step_outputs = pipeline_run.get_step(step='data_loader').outputs
    x_train = step_outputs['x_train'].read()
    x_test = step_outputs['x_test'].read()
    y_train = step_outputs['y_train'].read()
    y_test = step_outputs['y_test'].read()

    expected_size_test = ceil(EXPECTED_LENGTH * data_parameters.test_size)
    expected_size_train = int(EXPECTED_LENGTH * (1 - data_parameters.test_size))

    assert (len(x_train) + len(x_test)) == EXPECTED_LENGTH
    
    assert len(x_test) == expected_size_test
    assert len(y_test) == expected_size_test

    assert len(x_train) == expected_size_train
    assert len(y_train) == expected_size_train


def test_correct_model_type(pipeline_run):
    step_ouput = pipeline_run.get_step(step='trainer').output.read()

    assert isinstance(step_ouput, SVC)