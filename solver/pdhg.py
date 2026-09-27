"""
PDHG: Primal-Dual Hybrid Gradient for LP/QP
=============================================
GPU-accelerated first-order method for large-scale LP and QP.

Algorithm: PDHG (Chambolle-Pock) applied to LP:
  min  c^T x                     primal
  s.t. Ax = b,  x >= 0

Equivalent saddle-point:
  min_x max_y  c^T x + y^T(b - Ax)
              x >= 0

Update rules:
  x^{k+1} = proj_{x>=0}(x^k - tau*(c + A^T y^k))
  y^{k+1} = y^k + sigma*(b - A*(2x^{k+1} - x^k))

GPU acceleration via CuPy (falls back to NumPy if unavailable).
Adaptive restart and step-size tuning included.

Reference:
  Applegate et al. (2021). Practical Large-Scale Linear Programming using
  Primal-Dual Hybrid Gradient. NeurIPS 2021.
"""

import time
import numpy as np
import logging
from typing import Optional, Callable, List, Tuple

from .problem import OptimizationProblem, SolverResult, SolveStatus

logger = logging.getLogger("INDRA.pdhg")

# Try to import CuPy for GPU acceleration
try:
    import cupy as cp
    GPU_AVAILABLE = True
    logger.info("CuPy detected — GPU acceleration enabled")
except ImportError:
    cp = None
    GPU_AVAILABLE = False
    logger.info("CuPy not found — running PDHG on CPU (NumPy)")


class PDHGSolver:
    """
    Primal-Dual Hybrid Gradient (PDHG) for large-scale LP/QP.
    
    Ideal for very large, sparse problems where factorization-based methods
    (simplex, IPM) are too memory-intensive.
    
    GPU-accelerated via CuPy when available.
    """

    def __init__(
        self,
        max_iterations: int = 100000,
        tolerance: float = 1e-6,
        use_gpu: bool = True,
        restart_scheme: str = "adaptive",   # "none" | "fixed" | "adaptive"
        step_size_rule: str = "spectral",   # "fixed" | "spectral"
        verbose: bool = True,
        log_interval: int = 1000,
        log_callback: Optional[Callable] = None,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.restart_scheme = restart_scheme
        self.step_size_rule = step_size_rule
        self.verbose = verbose
        self.log_interval = log_interval
        self.log_callback = log_callback
        self._log: List[str] = []

        if self.use_gpu:
            self.xp = cp
            self._log_msg("PDHG: Using GPU (CuPy)")
        else:
            self.xp = np
            self._log_msg("PDHG: Using CPU (NumPy)")

    def _log_msg(self, msg: str):
        self._log.append(msg)
        if self.verbose:
            logger.info(msg)
        if self.log_callback:
            self.log_callback(msg)

    def solve(self, problem: OptimizationProblem) -> SolverResult:
        """Main solve entry."""
        t0 = time.time()
        self._log = []
        self._log_msg(f"[INDRA PDHG] '{problem.name}': {problem.n_vars} vars, "
                      f"{problem.n_ineq}+{problem.n_eq} constraints, "
                      f"GPU={'YES' if self.use_gpu else 'NO'}")
        try:
            result = self._solve_internal(problem)
        except Exception as e:
            self._log_msg(f"[ERROR] {e}")
            result = SolverResult(status=SolveStatus.UNKNOWN)

        result.solve_time = time.time() - t0
        result.log = self._log
        result.solver_name = "INDRA"
        result.algorithm = f"PDHG ({'GPU' if self.use_gpu else 'CPU'})"
        return result

    def _to_standard_form(
        self, problem: OptimizationProblem
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """Convert to: min c^Tx, Ax=b, l<=x<=u"""
        n = problem.n_vars
        c = problem.c.copy()
        obj_adj = 0.0

        if problem.maximize:
            c = -c

        lb = problem.lb.copy()
        ub = problem.ub.copy()

        rows_A = []
        rows_b = []

        if problem.A_ub is not None:
            # Add slacks for inequalities
            m_ub = problem.n_ineq
            A_slack = np.hstack([problem.A_ub, np.eye(m_ub)])
            rows_A.append(A_slack)
            rows_b.append(problem.b_ub)
            n = n + m_ub
            c = np.concatenate([c, np.zeros(m_ub)])
            lb = np.concatenate([lb, np.zeros(m_ub)])
            ub = np.concatenate([ub, np.full(m_ub, np.inf)])

        if problem.A_eq is not None:
            n_orig_vars = problem.n_vars
            n_slack = (problem.n_ineq if problem.A_ub is not None else 0)
            rows_A.append(np.hstack([
                problem.A_eq,
                np.zeros((problem.n_eq, n - problem.n_vars - (problem.n_ineq if problem.A_ub is not None else 0)))
                if n > problem.n_vars + (problem.n_ineq if problem.A_ub is not None else 0) else np.zeros((problem.n_eq, 0))
            ]))
            rows_b.append(problem.b_eq)

        if rows_A:
            A = np.vstack(rows_A)
            b = np.concatenate(rows_b)
        else:
            m_total = 0
            A = np.zeros((0, n))
            b = np.zeros(0)

        # Pad A columns if needed
        if A.shape[1] < n:
            A = np.hstack([A, np.zeros((A.shape[0], n - A.shape[1]))])

        return c, A, b, lb, ub, obj_adj

    def _solve_internal(self, problem: OptimizationProblem) -> SolverResult:
        """PDHG main loop."""
        c_np, A_np, b_np, lb_np, ub_np, obj_adj = self._to_standard_form(problem)
        xp = self.xp
        n = len(c_np)
        m = len(b_np)

        if m == 0:
            # Unbounded in general; return trivial solution
            return SolverResult(
                status=SolveStatus.OPTIMAL,
                objective_value=float(obj_adj),
                x=np.zeros(problem.n_vars),
            )

        # Transfer to GPU if available
        c = xp.array(c_np)
        A = xp.array(A_np)
        b = xp.array(b_np)
        lb = xp.array(lb_np)
        ub_finite = xp.array(np.where(np.isfinite(ub_np), ub_np, 1e10))

        # Step sizes: tau * sigma * ||A||^2 < 1
        if self.step_size_rule == "spectral":
            # Estimate ||A|| via power iteration
            v = xp.random.randn(n)
            v = v / (xp.linalg.norm(v) + 1e-14)
            for _ in range(20):
                u = A @ v
                v = A.T @ u
                nrm = xp.linalg.norm(v)
                v = v / (nrm + 1e-14)
            norm_A = float(xp.sqrt(nrm))
        else:
            norm_A = float(xp.linalg.norm(A, 'fro')) / np.sqrt(max(m, n))

        tau = 0.9 / (norm_A + 1e-14)
        sigma = 0.9 / (norm_A + 1e-14)

        # Initialize
        x = xp.zeros(n)
        y = xp.zeros(m)
        x_bar = x.copy()

        tol = self.tolerance
        best_prim = np.inf
        restart_period = 500

        header = f"{'Iter':>8} {'PrRes':>12} {'DuRes':>12} {'Gap':>12} {'Obj':>14}"
        self._log_msg(header)
        self._log_msg("-" * len(header))

        for k in range(self.max_iterations):
            # Dual update
            y_new = y + sigma * (b - A @ x_bar)

            # Primal update
            x_new = x - tau * (c + A.T @ y_new)
            # Project onto box [lb, ub]
            x_new = xp.clip(x_new, lb, ub_finite)

            # Extrapolation (over-relaxation)
            x_bar_new = 2 * x_new - x

            # Residuals
            prim_res = xp.linalg.norm(A @ x_new - b)
            dual_res = xp.linalg.norm(A.T @ y_new + c - xp.clip(A.T @ y_new + c, lb, ub_finite)) / tau
            obj_val = float(c @ x_new)
            dual_obj = float(b @ y_new)
            gap = abs(obj_val - dual_obj) / (1 + abs(obj_val))

            prim_res_f = float(prim_res)
            dual_res_f = float(dual_res)

            if k % self.log_interval == 0:
                self._log_msg(
                    f"{k:>8} {prim_res_f:>12.4e} {dual_res_f:>12.4e} "
                    f"{gap:>12.4e} {obj_val:>14.6f}"
                )

            # Adaptive restart
            if self.restart_scheme == "adaptive" and k > 0 and k % restart_period == 0:
                if obj_val < best_prim - tol * abs(best_prim):
                    best_prim = obj_val
                else:
                    # Restart
                    y_new = xp.zeros(m)
                    x_bar_new = x_new.copy()

            # Convergence
            if prim_res_f < tol and dual_res_f < tol and gap < tol:
                self._log_msg(f"  ✓ Converged at iteration {k}")
                break

            x = x_new
            y = y_new
            x_bar = x_bar_new

        # Transfer back to CPU
        if self.use_gpu:
            x_cpu = cp.asnumpy(x_new)
            y_cpu = cp.asnumpy(y_new)
        else:
            x_cpu = np.array(x_new)
            y_cpu = np.array(y_new)

        obj = float(c_np @ x_cpu) + obj_adj
        if problem.maximize:
            obj = -obj

        prim_r = float(np.linalg.norm(A_np @ x_cpu - b_np))
        dual_r = float(np.linalg.norm(A_np.T @ y_cpu + c_np))

        status = (SolveStatus.OPTIMAL
                  if prim_r < tol * 100 and dual_r < tol * 100
                  else SolveStatus.ITERATION_LIMIT)

        return SolverResult(
            status=status,
            objective_value=obj,
            x=x_cpu[:problem.n_vars],
            dual_values=y_cpu,
            iterations=k + 1,
            primal_residual=prim_r,
            dual_residual=dual_r,
            optimality_gap=gap,
        )
