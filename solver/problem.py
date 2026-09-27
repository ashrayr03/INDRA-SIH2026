"""
OptimizationProblem and SolverResult data structures.
Supports LP, MILP, and QP problem formulations.
"""

import time
import numpy as np
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


class SolveStatus(Enum):
    OPTIMAL = "OPTIMAL"
    INFEASIBLE = "INFEASIBLE"
    UNBOUNDED = "UNBOUNDED"
    TIME_LIMIT = "TIME_LIMIT"
    ITERATION_LIMIT = "ITERATION_LIMIT"
    UNKNOWN = "UNKNOWN"
    NOT_SOLVED = "NOT_SOLVED"


class ProblemType(Enum):
    LP = "LP"       # Linear Programming
    MILP = "MILP"   # Mixed-Integer Linear Programming
    QP = "QP"       # Quadratic Programming
    MIQP = "MIQP"   # Mixed-Integer Quadratic Programming


@dataclass
class OptimizationProblem:
    """
    Standard form optimization problem.
    
    LP/MILP:
        minimize    c^T x
        subject to  A_ub x <= b_ub
                    A_eq x  = b_eq
                    lb <= x <= ub
                    x_i in Z  for i in integer_vars
    
    QP:
        minimize    (1/2) x^T Q x + c^T x
        subject to  A_ub x <= b_ub
                    A_eq x  = b_eq
                    lb <= x <= ub
    """
    # Objective
    c: np.ndarray                           # Cost vector (n,)
    Q: Optional[np.ndarray] = None         # Quadratic cost matrix (n,n) for QP

    # Constraints
    A_ub: Optional[np.ndarray] = None      # Inequality constraint matrix (m_ub, n)
    b_ub: Optional[np.ndarray] = None      # Inequality RHS (m_ub,)
    A_eq: Optional[np.ndarray] = None      # Equality constraint matrix (m_eq, n)
    b_eq: Optional[np.ndarray] = None      # Equality RHS (m_eq,)

    # Bounds
    lb: Optional[np.ndarray] = None        # Lower bounds (n,)
    ub: Optional[np.ndarray] = None        # Upper bounds (n,)

    # Integer variables
    integer_vars: Optional[List[int]] = None  # Indices of integer variables

    # Metadata
    name: str = "unnamed"
    var_names: Optional[List[str]] = None
    con_names: Optional[List[str]] = None
    maximize: bool = False                  # If True, maximize instead of minimize

    def __post_init__(self):
        self.c = np.array(self.c, dtype=np.float64)
        n = len(self.c)

        if self.lb is None:
            self.lb = np.zeros(n)
        else:
            self.lb = np.array(self.lb, dtype=np.float64)

        if self.ub is None:
            self.ub = np.full(n, np.inf)
        else:
            self.ub = np.array(self.ub, dtype=np.float64)

        if self.A_ub is not None:
            self.A_ub = np.array(self.A_ub, dtype=np.float64)
        if self.b_ub is not None:
            self.b_ub = np.array(self.b_ub, dtype=np.float64)
        if self.A_eq is not None:
            self.A_eq = np.array(self.A_eq, dtype=np.float64)
        if self.b_eq is not None:
            self.b_eq = np.array(self.b_eq, dtype=np.float64)
        if self.Q is not None:
            self.Q = np.array(self.Q, dtype=np.float64)

        if self.var_names is None:
            self.var_names = [f"x{i}" for i in range(n)]

    @property
    def n_vars(self) -> int:
        return len(self.c)

    @property
    def n_ineq(self) -> int:
        return self.A_ub.shape[0] if self.A_ub is not None else 0

    @property
    def n_eq(self) -> int:
        return self.A_eq.shape[0] if self.A_eq is not None else 0

    @property
    def problem_type(self) -> ProblemType:
        if self.Q is not None:
            if self.integer_vars:
                return ProblemType.MIQP
            return ProblemType.QP
        if self.integer_vars:
            return ProblemType.MILP
        return ProblemType.LP

    def is_integer_var(self, idx: int) -> bool:
        if self.integer_vars is None:
            return False
        return idx in self.integer_vars

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": self.problem_type.value,
            "n_vars": self.n_vars,
            "n_ineq": self.n_ineq,
            "n_eq": self.n_eq,
            "n_integer": len(self.integer_vars) if self.integer_vars else 0,
            "has_quadratic": self.Q is not None,
        }


@dataclass
class SolverResult:
    """Container for solver output."""
    status: SolveStatus = SolveStatus.NOT_SOLVED
    objective_value: Optional[float] = None
    x: Optional[np.ndarray] = None
    dual_values: Optional[np.ndarray] = None
    reduced_costs: Optional[np.ndarray] = None

    # Performance metrics
    solve_time: float = 0.0
    iterations: int = 0
    nodes_explored: int = 0          # For B&B
    primal_residual: float = 0.0
    dual_residual: float = 0.0
    optimality_gap: float = 0.0

    # Solver info
    solver_name: str = "INDRA"
    solver_version: str = "0.1.0"
    algorithm: str = ""
    log: List[str] = field(default_factory=list)

    def is_optimal(self) -> bool:
        return self.status == SolveStatus.OPTIMAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "objective_value": self.objective_value,
            "x": self.x.tolist() if self.x is not None else None,
            "solve_time": round(self.solve_time, 6),
            "iterations": self.iterations,
            "nodes_explored": self.nodes_explored,
            "primal_residual": self.primal_residual,
            "dual_residual": self.dual_residual,
            "optimality_gap": self.optimality_gap,
            "solver_name": self.solver_name,
            "algorithm": self.algorithm,
        }
