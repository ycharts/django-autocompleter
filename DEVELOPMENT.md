# Create Release

1. `rm -rf dist`
1. Update version in `autocompleter/__init__.py`
1. `pip install -r dev_requirements.txt`
1. `python -m build`
1. `twine upload dist/*`
  - Make sure you have the API token in your `~/.pypirc`.
  - It should look something like:
    ```
    [pypi]
    username = __token__
    password = pypi-XXXXXXXXXXXXXXXX
    ```
