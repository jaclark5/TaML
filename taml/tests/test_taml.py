"""
Unit tests for TaML
"""

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

import numpy as np
import gpflow
from taml.GPRhetero import GPRhetero

def test_gpflow_version():
    """ Ensure version of gpflow is 2.2.1
    """
    assert gpflow.__version__ == '2.2.1'

def test_GPRhetero():
    """ Run sample calculation
    """
    # Generate training data
    X = np.linspace(0, 3, 2).reshape(-1, 1)
    Y = X*X-1
    NoiseVar = X*.05+.01

    # Test data and known results
    Xstar = np.linspace(1, 2, 2).reshape(-1, 1)
    muref = np.array([[1.24694676], [5.14551933]])
    varref = np.array([[0.39587969], [0.45496805]])

    # Build model
    kernel = gpflow.kernels.SquaredExponential(lengthscales=1.3)
    model = GPRhetero((X, Y), NoiseVar, kernel=kernel,
                      mean_function=gpflow.mean_functions.Constant(Y.mean()))

    mu, var = model.predict_f(Xstar)

    assert np.all(np.isclose(mu.numpy(), muref)) and np.all(np.isclose(var.numpy(), varref))

