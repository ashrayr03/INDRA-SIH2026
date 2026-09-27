"""
QP Solver using ADMM (Alternating Direction Method of Multipliers)
===================================================================
Solves convex QP:
    min  (1/2) x^T Q x + c^T x
    s.t. Ax <= b,  A_eq x = b_eq,  lb <= x <= ub

ADMM splits the problem into easy subproblems:
  - x-update: solve (Q + rho*A^T A) x = rhs  (linear system)
  - z-update: projection onto constraints
  - dual update

Also implements Active Set Method for small/medium QPs.

Reference:
  Boyd et al. (2011). Distributed Optimization and Statistical Learning
  via ADMM. Foundations and Trends in ML.
"""

import time
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from typing import Optional, Callable, List

from .problem import OptimizationProblem, SolverResult, SolveStatus

import logging
logger = logging.getLogger("INDRA.qp")


class QPSolver:
    """
    QP Solver: ADMM for large QPs, Active Set for small QPs.
    """

    def __init__(
        self,
        method: str = "admm",       # "admm" | "active_set"
        max_iterations: int = 10000,
        tolerance: float = 1e-6,
        rho: float = 1.0,           # ADMM penalty parameter
        adaptive_rho: bool = True,
        verbose: bool = True,
        log_callback: Optional[Callable] = None,
    ):
        self.method = method
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.rho = rho
        self.adaptive_rho = adaptive_rho
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
        """Main QP solve entry."""
        t0 = time.time()
        self._log = []

        if problem.Q is None:
            self._log_msg("[ERROR] No Q matrix — not a QP")
            return SolverResult(status=SolveStatus.UNKNOWN)

        self._log_msg(f"[INDRA QP-ADMM] '{problem.name}': {problem.n_vars} vars, "
                      f"{problem.n_ineq}+{problem.n_eq} constraints")

        try:
            if self.method == "active_set" or (problem.n_vars <= 100 and self.method != "admm"):
                result = self._active_set(problem)
            else:
                result = self._admm(problem)
        except Exception as e:
            self._log_msg(f"[ERROR] {e}")
            result = SolverResult(status=SolveStatus.UNKNOWN)

        result.solve_time = time.time() - t0
        result.log = self._log
        result.solver_name = "INDRA"
        result.algorithm = f"QP-{'ADMM' if self.method == 'admm' else 'ActiveSet'}"
        return result

    def _admm(self, problem: OptimizationProblem) -> SolverResult:
        """ADMM for QP with inequality + equality constraints."""
        n = problem.n_vars
        Q = problem.Q.copy()
        c = problem.c.copy()
        if problem.maximize:
            Q = -Q
            c = -c

        lb = problem.lb.copy()
        ub = problem.ub.copy()
        ub_safe = np.where(np.isfinite(ub), ub, 1e8)

        # Build constraint matrix [A_ub; A_eq]
        rows = []
        rhs = []
        con_type = []  # 'ub' | 'eq'

        if problem.A_ub is not None:
            rows.append(problem.A_ub)
            rhs.append(problem.b_ub)
            con_type.extend(['ub'] * problem.n_ineq)

        if problem.A_eq is not None:
            rows.append(problem.A_eq)
            rhs.append(problem.b_eq)
            con_type.extend(['eq'] * problem.n_eq)

        if rows:
            A = np.vstack(rows)
            b = np.concatenate(rhs)
            m = A.shape[0]
        else:
            A = np.zeros((0, n))
            b = np.zeros(0)
            m = 0

        rho = self.rho
        tol = self.tolerance

        # Precompute: (Q + rho * A^T A + rho * I)^{-1}  [cached for x-update]
        # Add rho*I for box constraint ADMM split
        M_qp = Q + rho * (A.T @ A if m > 0 else np.zeros((n, n))) + rho * np.eye(n)
        try:
            cho_fac = cho_factor(M_qp)
        except np.linalg.LinAlgError:
            M_qp += 1e-6 * np.eye(n)
            cho_fac = cho_factor(M_qp)

        # Variables
        x = np.zeros(n)
        z = np.zeros(n)     # box-constrained copy
        u = np.zeros(n)     # scaled dual for box
        y = np.zeros(m)     # dual for linear constraints

        header = f"{'Iter':>8} {'PrRes':>12} {'DuRes':>12} {'Obj':>14}"
        self._log_msg(header)
        self._log_msg("-" * len(header))

        for k in range(self.max_iterations):
            # x-update: argmin_x (1/2)x^T Q x + c^T x + (rho/2)||Ax-b+y/rho||^2 + (rho/2)||x-z+u||^2
            rhs_x = -c + rho * (z - u)
            if m > 0:
                rhs_x += rho * A.T @ (b - y / rho)
            x_new = cho_solve(cho_fac, rhs_x)

            # z-update: projection onto box [lb, ub]
            z_new = np.clip(x_new + u, lb, ub_safe)

            # Dual update
            u_new = u + x_new - z_new
            if m > 0:
                y_new = y + rho * (A @ x_new - b)
            else:
                y_new = y

            # Residuals
            prim_res = np.linalg.norm(x_new - z_new)
            dual_res = rho * np.linalg.norm(z_new - z)

            if k % 200 == 0:
                obj = 0.5 * x_new @ Q @ x_new + c @ x_new
                self._log_msg(f"{k:>8} {prim_res:>12.4e} {dual_res:>12.4e} {obj:>14.6f}")

            # Adaptive rho
            if self.adaptive_rho and k % 100 == 0 and k > 0:
                if prim_res > 10 * dual_res:
                    rho *= 2.0
                    u_new /= 2.0
                    y_new /= 2.0
                    # Recompute factorization
                    M_qp = Q + rho * (A.T @ A if m > 0 else np.zeros((n, n))) + rho * np.eye(n)
                    try:
                        cho_fac = cho_factor(M_qp)
                    except np.linalg.LinAlgError:
                        cho_fac = cho_factor(M_qp + 1e-6 * np.eye(n))
                elif dual_res > 10 * prim_res:
                    rho /= 2.0
                    u_new *= 2.0
                    y_new *= 2.0
                    M_qp = Q + rho * (A.T @ A if m > 0 else np.zeros((n, n))) + rho * np.eye(n)
                    try:
                        cho_fac = cho_factor(M_qp)
                    except np.linalg.LinAlgError:
                        cho_fac = cho_factor(M_qp + 1e-6 * np.eye(n))

            # Convergence
            if prim_res < tol and dual_res < tol:
                self._log_msg(f"  ✓ Converged at iteration {k}")
                break

            x, z, u, y = x_new, z_new, u_new, y_new

        obj = float(0.5 * x_new @ problem.Q @ x_new + problem.c @ x_new)
        if problem.maximize:
            obj = -obj

        return SolverResult(
            status=SolveStatus.OPTIMAL if prim_res < tol * 100 else SolveStatus.ITERATION_LIMIT,
            objective_value=obj,
            x=x_new,
            iterations=k + 1,
            primal_residual=float(prim_res),
        )

    def _active_set(self, problem: OptimizationProblem) -> SolverResult:
        """
        Active Set Method for small QPs.
        Iteratively adds/removes active constraints.
        """
        n = problem.n_vars
        Q = problem.Q.copy() + 1e-10 * np.eye(n)  # ensure PD
        c = problem.c.copy()
        if problem.maximize:
            Q = -Q
            c = -c

        lb = problem.lb.copy()
        ub = problem.ub.copy()
        ub_safe = np.where(np.isfinite(ub), ub, 1e8)

        # Build all constraints
        A_all = []
        b_all = []
        if problem.A_ub is not None:
            A_all.append(problem.A_ub)
            b_all.append(problem.b_ub)
        if problem.A_eq is not None:
            A_all.append(problem.A_eq)
            b_all.append(problem.b_eq)

        A = np.vstack(A_all) if A_all else np.zeros((0, n))
        b = np.concatenate(b_all) if b_all else np.zeros(0)
        m = len(b)

        # Initial feasible point (unconstrained minimum of QP)
        try:
            x = np.linalg.solve(Q, -c)
        except np.linalg.LinAlgError:
            x = np.zeros(n)
        x = np.clip(x, lb, ub_safe)

        active_set = set()
        tol = self.tolerance

        for itr in range(min(self.max_iterations, 1000)):
            # Solve EQP with active set
            active = sorted(active_set)
            if active:
                A_act = A[active]
                b_act = b[active]
                # KKT system:
                # [Q  A^T] [d]   [-g]
                # [A  0  ] [l] = [-r]
                g = Q @ x + c
                r = A_act @ x - b_act
                na = len(active)
                KKT = np.block([[Q, A_act.T], [A_act, np.zeros((na, na))]])
                rhs = np.concatenate([-g, -r])
                try:
                    sol = np.linalg.solve(KKT + 1e-10 * np.eye(n + na), rhs)
                    d = sol[:n]
                    lam = sol[n:]
                except np.linalg.LinAlgError:
                    break
            else:
                g = Q @ x + c
                try:
                    d = np.linalg.solve(Q, -g)
                except np.linalg.LinAlgError:
                    d = -g
                lam = np.zeros(0)

            if np.linalg.norm(d) < tol:
                # Check multipliers
                if len(lam) == 0 or np.all(lam >= -tol):
                    break  # Optimal
                # Remove most negative multiplier constraint
                remove_pos = int(np.argmin(lam))
                active_set.discard(active[remove_pos])
                continue

            # Line search
            alpha = 1.0
            blocking = -1
            for i in range(m):
                if i in active_set:
                    continue
                denom = A[i] @ d
                if denom > tol:
                    ratio = (b[i] - A[i] @ x) / denom
                    if ratio < alpha:
                        alpha = max(0, ratio)
                        blocking = i

            x = np.clip(x + alpha * d, lb, ub_safe)
            if blocking >= 0:
                active_set.add(blocking)

        obj = float(0.5 * x @ problem.Q @ x + problem.c @ x)
        if problem.maximize:
            obj = -obj

        return SolverResult(
            status=SolveStatus.OPTIMAL,
            objective_value=obj,
            x=x,
            iterations=itr + 1,
        )
