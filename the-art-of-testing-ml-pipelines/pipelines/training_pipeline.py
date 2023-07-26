from zenml import pipeline
# from steps import digits_data_loader, svc_trainer, evaluator



@pipeline
def training_pipeline():
    x_train, x_test, y_train, y_test = digits_data_loader(test_size=0.2)
    model = svc_trainer(x_train, y_train)
    test_acc = evaluator(model, x_test, y_test)