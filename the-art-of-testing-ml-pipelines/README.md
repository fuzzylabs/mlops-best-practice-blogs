```bash
pyenv install 3.10
pyenv local 3.10
poetry env use 3.10
poetry install
poetry run zenml init
poetry run zenml up
poetry run zenml integration install sklearn -y
poetry run zenml stack register zenml_testing_stack -a default -o default --set
poetry run python run.py
```