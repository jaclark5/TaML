"""
Theory aware Machine Learning (TaML)
"""
from setuptools import setup, find_packages

setup(name='TaML',
      version='1.0.0',
      description='This repository contains code to incorporate imperfect theory into machine learning for improved prediction and explainability. Specifically, it focuses on the case study of the dimensions of a polymer chain in different solvent qualities. For machine learning methods, three methods are considered: Gaussian Process Regression with heteroscedastic noise, Gaussian Process Regression with homoscedastic noise and Random Forest.',
      author='Debra J. Audus',
      author_email='debra.audus@nist.gov',
      packages=find_packages("taml"),
     )
