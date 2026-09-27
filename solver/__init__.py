"""
INDRA (Indigenous Numerical Decision-making and Resource Allocator)
Optimization Solver Core
Version: 0.1.0-prototype

Developed as a sovereign Indian optimization solver for industrial applications.
Supports: LP, MILP, QP
"""

from .simplex import RevisedSimplexSolver
from .interior_point import InteriorPointSolver
from .branch_and_bound import BranchAndBoundSolver
from .qp_solver import QPSolver
from .presolve import Presolve
from .problem import OptimizationProblem, SolverResult, SolveStatus

__version__ = "0.1.0-prototype"
__name__ = "INDRA"
__fullname__ = "Indigenous Numerical Decision-making and Resource Allocator"

__all__ = [
    "RevisedSimplexSolver",
    "InteriorPointSolver",
    "BranchAndBoundSolver",
    "QPSolver",
    "Presolve",
    "OptimizationProblem",
    "SolverResult",
    "SolveStatus",
]
