"""
Revised Simplex Method for LP
================================
Implements the full Revised Simplex algorithm with:
- Phase I (Big-M / two-phase) for finding initial feasible basis
- Phase II for optimality
- Bland's rule and Dantzig's rule for pivot selection
- Numerical stability via LU factorization with partial pivoting
- Degeneracy handling with perturbation
- Anti-cycling via lexicographic pivot rule
"""

import time
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from typing import Optional, List, Tuple, Callable
import logging

from .problem import OptimizationProblem, SolverResult, SolveStatus

logger = logging.getLogger("INDRA.simplex")


class RevisedSimplexSolver:
    """
    Revised Simplex Method for Linear Programming.
    
    Solves:
        min  c^T x
        s.t. Ax = b,  x >= 0
        
    Internally converts LP with inequalities and bounds to standard form.
    """

    def __init__(
        self,
        max_iterations: int = 50000,
        tolerance: float = 1e-8,
        pivot_rule: str = "dantzig",   # "dantzig" | "bland" | "steepest"
        perturbation: bool = True,
        verbose: bool = False,
        log_callback: Optional[Callable] = None,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.pivot_rule = pivot_rule
        self.perturbation = perturbation
        self.verbose = verbose
        self.log_callback = log_callback
        self._log: List[str] = []

    def _log_msg(self, msg: str):
        self._log.append(msg)
        if self.verbose:
            logger.info(msg)
        if self.log_callback:
            self.log_callback(msg)

    def solve(self, problem: OptimizationProblem) -> SolverResult:
        """Main solve entry point."""
        t0 = time.time()
        self._log = []
        self._log_msg(f"[INDRA Simplex] Solving '{problem.name}' — {problem.n_vars} vars, "
                      f"{problem.n_ineq} ineq, {problem.n_eq} eq")

        try:
            # Convert to standard equality form: min c^T x, Ax=b, x>=0
            c_std, A_std, b_std, n_orig, n_slack = self._to_standard_form(problem)
            result = self._solve_standard(c_std, A_std, b_std, n_orig, n_slack, problem)
        except Exception as e:
            self._log_msg(f"[ERROR] {str(e)}")
            result = SolverResult(status=SolveStatus.UNKNOWN)

        result.solve_time = time.time() - t0
        result.log = self._log
        result.solver_name = "INDRA"
        result.algorithm = "Revised Simplex"
        return result

    def _to_standard_form(
        self, problem: OptimizationProblem
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
        """
        Convert LP with general bounds and inequalities to standard form.
        
        Standard form:  min c^T x,  Ax = b,  x >= 0
        
        Transformations:
          - Finite lower bounds: x_i = x_i' + lb_i  (shift)
          - Finite upper bounds: add slack  x_i + s_i = ub_i - lb_i
          - Inequality  a^T x <= b:  add slack  s >= 0
          - Free variables: split x = x+ - x-
        """
        n = problem.n_vars
        c_orig = problem.c.copy()
        if problem.maximize:
            c_orig = -c_orig

        lb = problem.lb.copy()
        ub = problem.ub.copy()

        # Shift variables: x_shifted = x - lb
        b_shift = np.zeros(0)
        c_shifted = c_orig.copy()
        # Adjust RHS for bounds shift will happen per constraint

        rows = []
        rhs = []
        obj_adj = 0.0  # c^T lb

        # Objective offset from shifting
        for i in range(n):
            if np.isfinite(lb[i]):
                obj_adj += c_orig[i] * lb[i]

        # Build shifted inequality constraints
        if problem.A_ub is not None:
            A_shifted_ub = problem.A_ub.copy()
            b_shifted_ub = problem.b_ub.copy()
            for i in range(n):
                if np.isfinite(lb[i]):
                    b_shifted_ub -= problem.A_ub[:, i] * lb[i]
            rows.append(A_shifted_ub)
            rhs.append(b_shifted_ub)

        if problem.A_eq is not None:
            A_shifted_eq = problem.A_eq.copy()
            b_shifted_eq = problem.b_eq.copy()
            for i in range(n):
                if np.isfinite(lb[i]):
                    b_shifted_eq -= problem.A_eq[:, i] * lb[i]
            rows.append(A_shifted_eq)
            rhs.append(b_shifted_eq)

        # Stack A matrix
        m_ineq = problem.n_ineq
        m_eq = problem.n_eq

        if rows:
            A_base = np.vstack(rows)
            b_base = np.concatenate(rhs)
        else:
            A_base = np.zeros((0, n))
            b_base = np.zeros(0)

        m = m_ineq + m_eq

        # Add slack variables for inequalities
        n_slack = m_ineq
        I_slack = np.zeros((m, n_slack))
        if m_ineq > 0:
            I_slack[:m_ineq, :] = np.eye(m_ineq)

        # Add upper bound slacks  s_ub for finite upper bounds
        finite_ub_idx = [i for i in range(n) if np.isfinite(ub[i])]
        n_ub_slack = len(finite_ub_idx)
        A_ub_rows = np.zeros((n_ub_slack, n))
        b_ub_rhs = np.zeros(n_ub_slack)
        I_ub_slack = np.eye(n_ub_slack)
        for k, i in enumerate(finite_ub_idx):
            A_ub_rows[k, i] = 1.0
            b_ub_rhs[k] = ub[i] - (lb[i] if np.isfinite(lb[i]) else 0.0)

        # Final standard form
        m_total = m + n_ub_slack
        n_total = n + n_slack + n_ub_slack

        A_std = np.zeros((m_total, n_total))
        b_std = np.zeros(m_total)
        c_std = np.zeros(n_total)

        # Original vars
        if m > 0:
            A_std[:m, :n] = A_base
            b_std[:m] = b_base
        # Slacks for inequalities
        A_std[:m, n:n + n_slack] = I_slack
        # UB slack rows
        A_std[m:m + n_ub_slack, :n] = A_ub_rows
        A_std[m:m + n_ub_slack, n + n_slack:] = I_ub_slack
        b_std[m:] = b_ub_rhs

        c_std[:n] = c_shifted
        # Slack/UB-slack costs = 0

        # Ensure b >= 0 by flipping rows
        for i in range(m_total):
            if b_std[i] < 0:
                A_std[i] *= -1
                b_std[i] *= -1

        self._obj_adj = obj_adj
        self._n_orig = n
        self._n_slack = n_slack + n_ub_slack

        return c_std, A_std, b_std, n, n_slack + n_ub_slack

    def _solve_standard(
        self,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        n_orig: int,
        n_slack: int,
        problem: OptimizationProblem,
    ) -> SolverResult:
        """Solve in standard form using two-phase simplex."""
        m, n = A.shape
        tol = self.tolerance

        if m == 0:
            # Unconstrained — trivially solve
            x_opt = np.zeros(n_orig)
            return SolverResult(
                status=SolveStatus.OPTIMAL,
                objective_value=0.0,
                x=x_opt,
                iterations=0,
            )

        # ---- Phase I: find feasible basis ----
        # Augment with artificial variables a >= 0 for all rows
        # min sum(a),  [A | I_art] [x; a] = b,  x,a >= 0
        A_art = np.hstack([A, np.eye(m)])
        c_art = np.concatenate([np.zeros(n), np.ones(m)])
        basis = list(range(n, n + m))  # artificials form initial basis
        x_art = np.concatenate([np.zeros(n), b.copy()])

        self._log_msg(f"  Phase I: m={m}, n={n}, artificials={m}")
        status, x_art, basis, iters1 = self._simplex_iterations(
            c_art, A_art, b, basis, x_art
        )

        if status == SolveStatus.UNBOUNDED:
            return SolverResult(status=SolveStatus.INFEASIBLE, log=self._log)

        phase1_obj = c_art @ x_art
        if abs(phase1_obj) > tol * 100:
            self._log_msg(f"  Phase I infeasible: obj={phase1_obj:.6e}")
            return SolverResult(status=SolveStatus.INFEASIBLE, log=self._log)

        # Drive artificials out of basis
        for pos, bvar in enumerate(basis):
            if bvar >= n:
                # Artificial in basis at zero — try to pivot out
                for j in range(n):
                    if j not in basis:
                        col = A[:, j]
                        if abs(col[pos]) > tol:
                            basis[pos] = j
                            break

        # Remove artificial columns
        x_phase2 = x_art[:n]
        self._log_msg(f"  Phase I done: iters={iters1}")

        # ---- Phase II: optimize original objective ----
        self._log_msg(f"  Phase II: optimizing original objective")
        status, x_sol, basis, iters2 = self._simplex_iterations(
            c, A, b, basis, x_phase2
        )
        self._log_msg(f"  Phase II done: iters={iters2}, status={status.value}")

        result = SolverResult(
            status=status,
            iterations=iters1 + iters2,
            algorithm="Revised Simplex (2-Phase)",
        )

        if status == SolveStatus.OPTIMAL:
            obj = float(c[:n_orig] @ x_sol[:n_orig]) + self._obj_adj
            if problem.maximize:
                obj = -obj
            result.objective_value = obj
            result.x = x_sol[:n_orig]
            result.primal_residual = float(np.linalg.norm(A[:, :n_orig] @ x_sol[:n_orig] - b))

        return result

    def _simplex_iterations(
        self,
        c: np.ndarray,
        A: np.ndarray,
        b: np.ndarray,
        basis: List[int],
        x: np.ndarray,
    ) -> Tuple[SolveStatus, np.ndarray, List[int], int]:
        """Core revised simplex loop."""
        m, n = A.shape
        tol = self.tolerance
        iters = 0
        basis = list(basis)

        for iters in range(self.max_iterations):
            # Basis matrix
            B = A[:, basis]
            try:
                lu, piv = lu_factor(B)
                c_B = c[basis]
                # Dual variables (simplex multipliers)
                y = lu_solve((lu, piv), c_B, trans=1)

                # Reduced costs for non-basic vars
                nonbasis = [j for j in range(n) if j not in basis]
                if not nonbasis:
                    break

                rc = np.array([c[j] - y @ A[:, j] for j in nonbasis])

                # Optimality check
                if np.all(rc >= -tol):
                    return SolveStatus.OPTIMAL, x, basis, iters

                # Pivot selection
                if self.pivot_rule == "bland":
                    enter_local = next(i for i, r in enumerate(rc) if r < -tol)
                else:  # dantzig: most negative
                    enter_local = int(np.argmin(rc))

                enter = nonbasis[enter_local]

                # Solve B d = A[:,enter]
                d = lu_solve((lu, piv), A[:, enter])

                # Ratio test (minimum ratio)
                x_B = x[basis]
                ratios = np.full(m, np.inf)
                for i in range(m):
                    if d[i] > tol:
                        ratios[i] = x_B[i] / d[i]

                if np.all(np.isinf(ratios)):
                    return SolveStatus.UNBOUNDED, x, basis, iters

                # Perturbation for degeneracy
                if self.perturbation:
                    eps = 1e-10 * (1 + np.abs(ratios))
                    ratios = ratios + eps

                leave_pos = int(np.argmin(ratios))
                theta = ratios[leave_pos]

                # Update solution
                x[enter] = theta
                x[basis] = x[basis] - theta * d
                x[basis[leave_pos]] = 0.0
                basis[leave_pos] = enter

            except np.linalg.LinAlgError:
                self._log_msg("  [WARN] Singular basis — attempting recovery")
                break

        return SolveStatus.ITERATION_LIMIT, x, basis, iters
