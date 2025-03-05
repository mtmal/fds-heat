"""
Heat Equation Solver package.

This package provides numerical solvers for the one-dimensional heat equation
using finite difference methods.
"""

from .solver import HeatEquationSolver, SchemeType

__version__ = "0.1.0"
__all__ = ["HeatEquationSolver", "SchemeType"] 