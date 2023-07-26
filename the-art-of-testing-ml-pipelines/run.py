from pipelines import training_pipeline


def main():
    pipeline = training_pipeline.with_options(config_path='pipeline_config.yaml')
    pipeline()


if __name__ == '__main__':
    main()