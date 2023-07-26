# The Art of Testing Machine Learning Pipelines

Creating a machine learning model is a step-by-step process and there are tools out there, such as ZenML, which enable you to define these steps and link them together as a pipeline. It goes without saying that writing reliable code is important, and one aspect of this is thorough testing. The techniques for doing this are established in traditional software engineering, but how do they translate into MLOps? How do you ensure that the machine learning pipelines you’re writing are working as intended?

The blog post associated with this repository guides you through how to properly test your machine learning pipelines. This repository contains the code associated with that blog post. If you'd ike to get the code running yourself, then follow the steps below.

We use both `pyenv` and `poetry` here, `pyenv` manages Python versions on your system while `poetry` is a Python dependency tool. You can install these by going through the following:

- [Pyenv installation guide](https://github.com/pyenv/pyenv#installation)
- [Poetry installation guide](https://python-poetry.org/docs/)

Once these are installed and you have the repository cloned, you need to install the correct version of Python and tell `pyenv` to use it:

```bash
pyenv install 3.10
pyenv local 3.10
```

Similarly, `poetry` needs to be told to use it:

```bash
poetry env use 3.10
```

Once that's complete, you need to setup ZenML:

```bash
poetry install
poetry run zenml init
poetry run zenml integration install sklearn -y
poetry run zenml stack register zenml_testing_stack -a default -o default --set
```

From there, you can either run the pipeline itself:

```bash
poetry run python run.py
```

Or the tests:

```bash
poetry run python -m pytest tests
```