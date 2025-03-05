"""
Implementation of finite difference solvers for the 1D heat equation.
"""

import numpy as np
from enum import Enum
from typing import Optional, Callable, Tuple


class SchemeType(Enum):
    """Enumeration of available numerical schemes."""
    FTCS = "ftcs"  # Forward-Time Central-Space (explicit)
    BTCS = "btcs"  # Backward-Time Central-Space (implicit)
    CN = "cn"      # Crank-Nicolson (implicit)


class HeatEquationSolver:
    """
    A class for solving the 1D heat equation using finite difference methods.
    
    The heat equation solved is:
        ∂u/∂t = α * ∂²u/∂x²
    
    Attributes:
        alpha (float): Thermal diffusivity coefficient
        length (float): Length of the domain
        nx (int): Number of spatial points
        nt (int): Number of time steps
        dx (float): Spatial step size
        dt (float): Time step size
        scheme (SchemeType): Numerical scheme to use
        x (np.ndarray): Spatial grid points
        r (float): Stability parameter (α*dt/dx²)
    """
    
    def __init__(
        self,
        alpha: float,
        length: float,
        nx: int,
        nt: int,
        dt: float,
        scheme: SchemeType = SchemeType.FTCS,
        bc_left: Optional[Callable[[float], float]] = None,
        bc_right: Optional[Callable[[float], float]] = None
    ):
        """
        Initialize the heat equation solver.
        
        Args:
            alpha: Thermal diffusivity coefficient
            length: Length of the domain
            nx: Number of spatial points
            nt: Number of time steps
            dt: Time step size
            scheme: Numerical scheme to use (default: FTCS)
            bc_left: Left boundary condition function of time (default: zero temperature)
            bc_right: Right boundary condition function of time (default: zero temperature)
        
        The boundary condition functions should take a time value and return the temperature:
            def bc(t: float) -> float
        
        If boundary conditions are not provided, zero temperature (u = 0) is assumed.
        
        Raises:
            ValueError: If the FTCS scheme parameters violate the stability condition
        """
        self.alpha = alpha
        self.length = length
        self.nx = nx
        self.nt = nt
        self.dt = dt
        self.dx = length / (nx - 1)
        self.scheme = scheme
        
        # Compute stability parameter
        self.r = alpha * dt / (self.dx ** 2)
        
        # Check stability for FTCS
        if scheme == SchemeType.FTCS and self.r > 0.5:
            raise ValueError(
                f"FTCS scheme is unstable for given parameters. "
                f"Current r = {self.r:.3f}, must be ≤ 0.5. "
                f"Reduce dt or increase dx."
            )
        
        # Set boundary conditions
        self.bc_left = bc_left if bc_left is not None else lambda t: 0.0
        self.bc_right = bc_right if bc_right is not None else lambda t: 0.0
        
        # Initialize grid
        self.x = np.linspace(0, length, nx)
        
    def solve(self, initial_condition: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the heat equation.
        
        Args:
            initial_condition: Function defining the initial temperature distribution
            
        Returns:
            Tuple containing:
                - Time points array (shape: (nt+1,))
                - Solution array (shape: (nt+1, nx))
        """
        # Initialize solution array
        u = np.zeros((self.nt + 1, self.nx))
        t = np.linspace(0, self.nt * self.dt, self.nt + 1)
        
        # Set initial condition
        u[0] = initial_condition(self.x)
        
        # Apply boundary conditions
        for n in range(self.nt + 1):
            u[n, 0] = self.bc_left(t[n])
            u[n, -1] = self.bc_right(t[n])
        
        if self.scheme == SchemeType.FTCS:
            self._solve_ftcs(u)
        elif self.scheme == SchemeType.BTCS:
            self._solve_btcs(u)
        else:  # Crank-Nicolson
            self._solve_cn(u)
            
        return t, u
    
    def _solve_ftcs(self, u: np.ndarray) -> None:
        """
        Solve using the Forward-Time Central-Space (explicit) scheme.
        
        Args:
            u: Solution array to be updated in-place
        """
        for n in range(self.nt):
            u[n+1, 1:-1] = (u[n, 1:-1] + 
                           self.r * (u[n, 2:] - 2 * u[n, 1:-1] + u[n, :-2]))
    
    def _solve_btcs(self, u: np.ndarray) -> None:
        """
        Solve using the Backward-Time Central-Space (implicit) scheme.
        
        Args:
            u: Solution array to be updated in-place
        """
        # Create tridiagonal matrix
        main_diag = np.ones(self.nx) * (1 + 2 * self.r)
        off_diag = np.ones(self.nx - 1) * (-self.r)
        
        # Create tridiagonal matrix
        matrix = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)
        
        # Time stepping
        for n in range(self.nt):
            # Right hand side
            b = u[n].copy()
            
            # Modify matrix and RHS for boundary conditions
            matrix[0] = matrix[-1] = np.zeros(self.nx)
            matrix[0, 0] = matrix[-1, -1] = 1.0
            b[0] = self.bc_left((n + 1) * self.dt)
            b[-1] = self.bc_right((n + 1) * self.dt)
            
            # Solve system
            u[n+1] = np.linalg.solve(matrix, b)

    def _solve_cn(self, u: np.ndarray) -> None:
        """
        Solve using the Crank-Nicolson (implicit) scheme.
        
        Args:
            u: Solution array to be updated in-place
        """
        r = self.r
        
        # Implicit part (left side)
        main_diag_i = np.ones(self.nx) * (1 + r)
        off_diag_i = np.ones(self.nx - 1) * (-r/2)
        
        # Explicit part (right side)
        main_diag_e = np.ones(self.nx) * (1 - r)
        off_diag_e = np.ones(self.nx - 1) * (r/2)
        
        # Create matrices
        matrix_i = np.diag(main_diag_i) + np.diag(off_diag_i, k=1) + np.diag(off_diag_i, k=-1)
        matrix_e = np.diag(main_diag_e) + np.diag(off_diag_e, k=1) + np.diag(off_diag_e, k=-1)
        
        # Time stepping
        for n in range(self.nt):
            # Modify matrices for boundary conditions
            matrix_i[0] = matrix_i[-1] = np.zeros(self.nx)
            matrix_i[0, 0] = matrix_i[-1, -1] = 1.0
            
            # Compute right-hand side
            b = matrix_e @ u[n]
            b[0] = self.bc_left((n + 1) * self.dt)
            b[-1] = self.bc_right((n + 1) * self.dt)
            
            # Solve system
            u[n+1] = np.linalg.solve(matrix_i, b)
