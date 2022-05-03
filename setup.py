import setuptools

setuptools.setup(name='simple_fpa',
version='0.1',
description='A package for the "Nonparametric inference on counterfactuals in sealed first-price auctions" paper.',
author='Pasha Andreyanov',
author_email='pandreyanov@gmail.com',
install_requires=['numpy','pandas','scipy'],
packages=setuptools.find_packages(),
zip_safe=False)