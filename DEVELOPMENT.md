# Create Release

1. Update version in `autocompleter/__init__.py`
2. `pip install -r dev_requirements.txt`
3. `python -m build`
4. `twine upload dist/*`
