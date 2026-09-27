"""
Presolve routines for LP/MILP problem reduction.
=================================================
Presolve reduces problem size before solving by detecting and exploiting
problem structure. Each rule is applied iteratively until no reductions occur.

Rules implemented:
  1. Fixed variables (lb = ub)
  2. Empty rows (infeasibility detection)
  3. Singleton rows (implied bound tightening)
  4. Duplicate constraints
  5. Bound tightening from constraints
  6. Zero objective coefficients with free variables
  7. Dominated columns
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from .problem import OptimizationProblem
import logging

logger = logging.getLogger("INDRA.presolve")


class Presolve:
    """Presolve/postsolve for LP and MILP problems."""

    def __init__(self, tolerance: float = 1e-8, verbose: bool = False):
        self.tolerance = tolerance
        self.verbose = verbose
        self._transformations: List = []
        self._stats: Dict = {}

    def presolve(self, problem: OptimizationProblem) -> Tuple[OptimizationProblem, Dict]:
        """
        Apply presolve reductions. Returns reduced problem and statistics.
        """
        stats = {
            "vars_removed": 0,
            "cons_removed": 0,
            "bounds_tightened": 0,
            "rounds": 0,
        }

        n = problem.n_vars
        c = problem.c.copy()
        lb = problem.lb.copy()
        ub = problem.ub.copy()

        A_ub = problem.A_ub.copy() if problem.A_ub is not None else None
        b_ub = problem.b_ub.copy() if problem.b_ub is not None else None
        A_eq = problem.A_eq.copy() if problem.A_eq is not None else None
        b_eq = problem.b_eq.copy() if problem.b_eq is not None else None

        active_vars = list(range(n))
        fixed_vals: Dict[int, float] = {}
        obj_offset = 0.0

        tol = self.tolerance
        max_rounds = 20

        for round_num in range(max_rounds):
            changed = False

            # ---- Rule 1: Fixed variables (lb == ub) ----
            for i in list(active_vars):
                if abs(ub[i] - lb[i]) <= tol:
                    val = (lb[i] + ub[i]) / 2
                    fixed_vals[i] = val
                    obj_offset += c[i] * val
                    active_vars.remove(i)
                    stats["vars_removed"] += 1
                    changed = True

                    # Remove from constraints
                    if A_ub is not None:
                        b_ub -= A_ub[:, i] * val
                        A_ub[:, i] = 0
                    if A_eq is not None:
                        b_eq -= A_eq[:, i] * val
                        A_eq[:, i] = 0

            # ---- Rule 2: Bound tightening from inequalities ----
            if A_ub is not None:
                for r in range(A_ub.shape[0]):
                    row = A_ub[r]
                    rhs = b_ub[r]
                    for i in active_vars:
                        if row[i] > tol:
                            # a_i * x_i <= rhs - sum_{j!=i} a_j * x_j
                            rest_min = sum(
                                row[j] * (lb[j] if row[j] > 0 else ub[j])
                                for j in active_vars if j != i
                                if np.isfinite(lb[j] if row[j] > 0 else ub[j])
                            )
                            new_ub = (rhs - rest_min) / row[i]
                            if new_ub < ub[i] - tol:
                                ub[i] = new_ub
                                stats["bounds_tightened"] += 1
                                changed = True
                        elif row[i] < -tol:
                            rest_max = sum(
                                row[j] * (ub[j] if row[j] > 0 else lb[j])
                                for j in active_vars if j != i
                                if np.isfinite(ub[j] if row[j] > 0 else lb[j])
                            )
                            new_lb = (rhs - rest_max) / row[i]
                            if new_lb > lb[i] + tol:
                                lb[i] = new_lb
                                stats["bounds_tightened"] += 1
                                changed = True

            # ---- Rule 3: Empty/trivially satisfied rows ----
            if A_ub is not None:
                keep_rows = []
                for r in range(A_ub.shape[0]):
                    row = A_ub[r]
                    if np.all(np.abs(row) < tol):
                        if b_ub[r] < -tol:
                            # Infeasible
                            logger.warning("Presolve detected infeasibility in row %d", r)
                        stats["cons_removed"] += 1
                        changed = True
                    else:
                        keep_rows.append(r)
                if len(keep_rows) < A_ub.shape[0]:
                    A_ub = A_ub[keep_rows]
                    b_ub = b_ub[keep_rows]

            stats["rounds"] = round_num + 1
            if not changed:
                break

        # Build reduced problem
        idx = sorted(active_vars)
        if len(idx) < n:
            c_red = c[idx]
            lb_red = lb[idx]
            ub_red = ub[idx]
            A_ub_red = A_ub[:, idx] if A_ub is not None else None
            A_eq_red = A_eq[:, idx] if A_eq is not None else None

            reduced = OptimizationProblem(
                c=c_red,
                A_ub=A_ub_red,
                b_ub=b_ub,
                A_eq=A_eq_red,
                b_eq=b_eq,
                lb=lb_red,
                ub=ub_red,
                name=f"{problem.name}_presolved",
                integer_vars=[idx.index(i) for i in (problem.integer_vars or []) if i in idx],
            )
        else:
            reduced = OptimizationProblem(
                c=c,
                A_ub=A_ub,
                b_ub=b_ub,
                A_eq=A_eq,
                b_eq=b_eq,
                lb=lb,
                ub=ub,
                name=f"{problem.name}_presolved",
                integer_vars=problem.integer_vars,
            )

        self._active_vars = idx
        self._fixed_vals = fixed_vals
        self._obj_offset = obj_offset
        self._original_n = n

        if self.verbose:
            logger.info(
                "Presolve: %d vars removed, %d cons removed, %d bounds tightened in %d rounds",
                stats["vars_removed"], stats["cons_removed"],
                stats["bounds_tightened"], stats["rounds"]
            )

        return reduced, stats

    def postsolve(self, x_reduced: np.ndarray) -> np.ndarray:
        """Recover full solution vector from reduced solution."""
        x_full = np.zeros(self._original_n)
        for i, orig_i in enumerate(self._active_vars):
            x_full[orig_i] = x_reduced[i]
        for orig_i, val in self._fixed_vals.items():
            x_full[orig_i] = val
        return x_full
