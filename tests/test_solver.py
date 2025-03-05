"""
Unit tests for the heat equation solver.

This test suite verifies the correctness of the numerical schemes implemented
for solving the 1D heat equation. It checks various properties that any correct
implementation should satisfy, including stability conditions, boundary conditions,
and convergence of different schemes.
"""

import numpy as np
import pytest
from heat_solver import HeatEquationSolver, SchemeType


@pytest.fixture
def basic_solver_params():
    """
    Basic solver parameters that satisfy stability conditions.
    
    These parameters are chosen to ensure:
    1. The FTCS scheme is stable (r = α*dt/dx² ≈ 0.1 < 0.5)
    2. The grid is fine enough for accurate solutions
    3. The time step is small enough for good temporal resolution
    """
    return {
        'alpha': 0.1,      # Thermal diffusivity
        'length': 1.0,     # Domain length
        'nx': 50,          # Number of spatial points
        'nt': 100,         # Number of time steps
        'dt': 0.001,       # Time step size
    }


def test_solver_initialization(basic_solver_params):
    """
    Test that solver initializes correctly with valid parameters.
    
    This test verifies that:
    1. Parameters are correctly stored in the solver instance
    2. The spatial grid is created with the correct number of points
    3. The solver can be instantiated without errors
    """
    solver = HeatEquationSolver(**basic_solver_params)
    assert solver.alpha == basic_solver_params['alpha']
    assert solver.length == basic_solver_params['length']
    assert len(solver.x) == basic_solver_params['nx']


def test_ftcs_stability_check():
    """
    Test that FTCS solver raises error for unstable parameters.
    
    The FTCS scheme is conditionally stable with the condition:
    r = α*dt/dx² ≤ 0.5 (CFL condition)
    
    This test verifies that:
    1. The solver detects when stability conditions are violated
    2. An appropriate error is raised with unstable parameters
    3. The error message contains useful information
    """
    params = {
        'alpha': 0.1,
        'length': 1.0,
        'nx': 10,
        'nt': 100,
        'dt': 0.1,  # This will make r > 0.5
        'scheme': SchemeType.FTCS
    }
    
    with pytest.raises(ValueError, match="FTCS scheme is unstable"):
        HeatEquationSolver(**params)


def test_constant_solution(basic_solver_params):
    """
    Test that constant initial condition remains constant.
    
    This is a fundamental property of the heat equation:
    If temperature is initially uniform and boundary conditions match this temperature,
    the solution should remain constant for all time.
    
    This test verifies that:
    1. All numerical schemes preserve constant solutions
    2. No artificial numerical diffusion is introduced
    3. Boundary conditions are properly handled
    """
    # Define constant boundary conditions
    def bc_constant(t):
        return 1.0
    
    for scheme in SchemeType:
        solver = HeatEquationSolver(
            **basic_solver_params,
            scheme=scheme,
            bc_left=bc_constant,
            bc_right=bc_constant
        )
        t, u = solver.solve(lambda x: np.ones_like(x))
        
        assert np.allclose(u, 1.0), f"Constant solution failed for {scheme}"


def test_boundary_conditions(basic_solver_params):
    """
    Test that boundary conditions are respected.
    
    This test verifies that:
    1. Different boundary values are correctly applied
    2. Boundary conditions are maintained throughout the simulation
    3. All schemes properly handle non-homogeneous boundary conditions
    
    The test uses different values for left and right boundaries to ensure
    the solver can handle asymmetric conditions.
    """
    def bc_left(t):
        return 1.0
    
    def bc_right(t):
        return 2.0
    
    for scheme in SchemeType:
        solver = HeatEquationSolver(
            **basic_solver_params, 
            scheme=scheme,
            bc_left=bc_left,
            bc_right=bc_right
        )
        t, u = solver.solve(lambda x: np.zeros_like(x))
        
        assert np.allclose(u[:, 0], 1.0), f"Left BC failed for {scheme}"
        assert np.allclose(u[:, -1], 2.0), f"Right BC failed for {scheme}"


def test_schemes_convergence(basic_solver_params):
    """
    Test that all schemes converge to the same solution for long times.
    
    For the heat equation, different numerical schemes should converge to the same
    steady-state solution for t → ∞. This test verifies that:
    1. All schemes converge to the same solution
    2. The solutions are physically meaningful
    3. The numerical schemes are consistent with each other
    
    Uses a Gaussian pulse as initial condition, which is a common test case
    for heat equation solvers.
    """
    params = basic_solver_params.copy()
    params['nt'] = 1000  # Long time integration
    
    # Initial condition: Gaussian pulse
    def initial_condition(x):
        return np.exp(-(x - 0.5)**2 / 0.1)
    
    solutions = {}
    for scheme in SchemeType:
        solver = HeatEquationSolver(**params, scheme=scheme)
        t, u = solver.solve(initial_condition)
        solutions[scheme] = u[-1]  # Final solution
    
    # Compare final solutions
    for scheme1, scheme2 in [(SchemeType.FTCS, SchemeType.BTCS),
                            (SchemeType.BTCS, SchemeType.CN)]:
        assert np.allclose(
            solutions[scheme1], 
            solutions[scheme2],
            rtol=1e-3
        ), f"Solutions differ between {scheme1} and {scheme2}" 