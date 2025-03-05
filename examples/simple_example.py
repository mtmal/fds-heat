"""
Example script demonstrating the usage of the heat equation solver.
"""

import numpy as np
import matplotlib.pyplot as plt
from heat_solver import HeatEquationSolver, SchemeType
from typing import Callable, Optional


def solve_and_plot(
    alpha: float = 0.1,
    L: float = 1.0,
    nx: int = 50,
    bc_left: Callable[[float], float] = lambda t: 0.0,
    bc_right: Callable[[float], float] = lambda t: 0.0,
    initial_condition: Optional[Callable[[np.ndarray], np.ndarray]] = None
) -> None:
    """
    Solve and visualize the heat equation with given parameters and conditions.
    
    Args:
        alpha: Thermal diffusivity coefficient
        L: Domain length
        nx: Number of spatial points
        bc_left: Left boundary condition function
        bc_right: Right boundary condition function
        initial_condition: Initial temperature distribution function
    """
    # Calculate dt to ensure stability for FTCS
    dx = L / (nx - 1)
    dt = 0.4 * dx**2 / alpha  # Using r = 0.4 for safety (less than 0.5)
    nt = 100      # Number of time steps
    
    print(f"Using parameters: dx={dx:.3f}, dt={dt:.3f}, r={alpha*dt/dx**2:.3f}")
    
    # Default initial condition if none provided
    if initial_condition is None:
        initial_condition = lambda x: np.exp(-(x - L/2)**2 / 0.1)
    
    # Create solver instances for all schemes
    schemes = [SchemeType.FTCS, SchemeType.BTCS, SchemeType.CN]
    solvers = {}
    solutions = {}
    
    for scheme in schemes:
        solvers[scheme] = HeatEquationSolver(
            alpha=alpha,
            length=L,
            nx=nx,
            nt=nt,
            dt=dt,
            scheme=scheme,
            bc_left=bc_left,
            bc_right=bc_right
        )
        t, u = solvers[scheme].solve(initial_condition)
        solutions[scheme] = (t, u)
    
    # Plot results
    plt.figure(figsize=(12, 6))
    
    # Plot initial condition
    plt.plot(solvers[SchemeType.FTCS].x, solutions[SchemeType.FTCS][1][0], 
            'k--', label='Initial condition')
    
    # Plot solutions at different times
    relative_times = [0.25, 0.5, 1.0]  # Plot at 25%, 50%, and 100% of simulation time
    times_to_plot = [int(nt * r) for r in relative_times]
    styles = {SchemeType.FTCS: '-', SchemeType.BTCS: '--', SchemeType.CN: ':'}
    
    for t_idx in times_to_plot:
        physical_time = t_idx * dt  # Convert step index to actual time
        for scheme in schemes:
            t, u = solutions[scheme]
            plt.plot(solvers[scheme].x, u[t_idx], styles[scheme], 
                    label=f'{scheme.value.upper()} (t={physical_time:.3f} s)')
    
    plt.xlabel('x')
    plt.ylabel('Temperature')
    plt.title('Heat Equation Solution: Comparison of Numerical Schemes')
    plt.legend()
    plt.grid(True)
    plt.savefig('example.png', dpi=300, bbox_inches='tight')
    plt.show()


def main():
    """Example usage with different initial conditions and boundary conditions."""
    
    # Example with step function and matching boundary conditions
    print("\nExample: Step function with matching boundary conditions")
    def step_initial_condition(x):
        return np.where(x < 0.5, 1.0, 0.0)
    
    # First case: Matching boundary conditions
    print("Case 1: Matching boundary conditions")
    solve_and_plot(
        initial_condition=step_initial_condition,
        bc_left=lambda t: 1.0,   # Matches left side of step
        bc_right=lambda t: 0.0,  # Matches right side of step
        alpha=0.1,               # Reduced diffusivity for clearer visualization
        nx=100                   # More spatial points for better resolution
    )
    
    # Second case: Non-matching boundary conditions
    print("Case 2: Non-matching boundary conditions")
    solve_and_plot(
        initial_condition=step_initial_condition,
        bc_left=lambda t: 0.0,   # Does NOT match left side of step
        bc_right=lambda t: 0.0,  # Matches right side of step
        alpha=0.1,
        nx=100
    )


if __name__ == "__main__":
    main() 