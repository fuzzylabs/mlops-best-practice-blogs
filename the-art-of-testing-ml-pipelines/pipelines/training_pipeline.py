from zenml.pipelines import pipeline 

@pipeline
def training_pipeline(
    data_loader,
    trainer,
    evaluator
):
    x_train, x_test, y_train, y_test = data_loader()
    model = trainer(x_train, y_train)
    test_acc = evaluator(model, x_test, y_test)