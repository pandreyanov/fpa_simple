python setup.py sdist bdist_wheel
twine upload --skip-existing -r testpypi dist/*
twine upload --skip-existing -r pypi dist/*