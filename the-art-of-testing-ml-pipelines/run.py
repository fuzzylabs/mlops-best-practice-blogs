from steps import (
    digits_data_loader,
    svc_trainer,
    evaluator
)
from pipelines import training_pipeline



def main():
    # pipeline = training_pipeline(
    #     data_loader=digits_data_loader(),
    #     trainer=svc_trainer(),
    #     evaluator=evaluator()
    # )
    # pipeline.run(config_path='pipeline_config.yaml')
    pipeline = training_pipeline.with_options(config_path='pipeline_config.yaml')
    pipeline()


if __name__ == '__main__':
    main()