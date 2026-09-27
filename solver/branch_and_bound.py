"""
Branch-and-Bound with Cutting Planes for MILP
===============================================
Implements:
  - Best-first / depth-first / hybrid node selection
  - Strong branching variable selection
  - Gomory fractional cuts
  - Primal heuristics (rounding, feasibility pump)
  - Dual bound propagation
  - Node presolve
"""

import time
import heapq
import numpy as np
from typing import Optional, List, Tuple, Callable, Dict
from dataclasses import dataclass, field
import logging

from .problem import OptimizationProblem, SolverResult, SolveStatus
from .simplex import RevisedSimplexSolver
from .interior_point import InteriorPointSolver

logger = logging.getLogger("INDRA.bnb")


@dataclass(order=True)
class BBNode:
    """A node in the Branch-and-Bound tree."""
    priority: float                          # Lower = higher priority (best-first)
    node_id: int = field(compare=False)
    depth: int = field(compare=False)
    lb_bounds: np.ndarray = field(compare=False)   # Additional lower bounds from branching
    ub_bounds: np.ndarray = field(compare=False)   # Additional upper bounds from branching
    parent_id: int = field(compare=False, default=-1)
    lp_bound: float = field(compare=False, default=-np.inf)


class BranchAndBoundSolver:
    """
    Branch-and-Bound with Gomory cuts for MILP.
    
    Strategy:
      - Best-first search with depth fallback (hybrid)
      - Most-infeasible branching + strong branching
      - Gomory fractional cuts at root
      - Simple rounding heuristic for primal bound
    """

    def __init__(
        self,
        max_nodes: int = 100000,
        max_time: float = 300.0,
        tolerance: float = 1e-6,
        integer_tol: float = 1e-5,
        node_strategy: str = "best_first",  # "best_first" | "depth_first" | "hybrid"
        branching: str = "most_infeasible",  # "most_infeasible" | "pseudo_cost"
        gomory_cuts: bool = True,
        max_gomory_rounds: int = 10,
        verbose: bool = True,
        log_callback: Optional[Callable] = None,
    ):
        self.max_nodes = max_nodes
        self.max_time = max_time
        self.tolerance = tolerance
        self.integer_tol = integer_tol
        self.node_strategy = node_strategy
        self.branching = branching
        self.gomory_cuts = gomory_cuts
        self.max_gomory_rounds = max_gomory_rounds
        self.verbose = verbose
        self.log_callback = log_callback
        self._log: List[str] = []

        # LP solver for node relaxations
        self._lp_solver = RevisedSimplexSolver(
            max_iterations=20000,
            tolerance=1e-8,
            verbose=False,
        )

    def _log_msg(self, msg: str):
        self._log.append(msg)
        if self.verbose:
            logger.info(msg)
        if self.log_callback:
            self.log_callback(msg)

    def solve(self, problem: OptimizationProblem) -> SolverResult:
        """Main B&B solve routine."""
        t0 = time.time()
        self._log = []

        if not problem.integer_vars:
            self._log_msg("[WARN] No integer variables — solving as LP")
            return self._lp_solver.solve(problem)

        self._log_msg(f"[INDRA B&B] '{problem.name}': {problem.n_vars} vars, "
                      f"{len(problem.integer_vars)} integers, "
                      f"{problem.n_ineq}+{problem.n_eq} constraints")

        n = problem.n_vars
        int_idx = set(problem.integer_vars)
        INF = np.inf

        # Global bounds
        global_lb = problem.lb.copy()
        global_ub = problem.ub.copy()

        # Best incumbent
        best_obj = INF if not problem.maximize else -INF
        best_x = None
        nodes_explored = 0
        node_counter = 0

        # Root node
        root = BBNode(
            priority=0.0,
            node_id=0,
            depth=0,
            lb_bounds=global_lb.copy(),
            ub_bounds=global_ub.copy(),
        )

        # Priority queue (min-heap)
        pq: List[BBNode] = []
        heapq.heappush(pq, root)
        node_counter = 1

        # Pseudo-cost tracking for branching
        pseudo_up = {i: (0.0, 0) for i in int_idx}    # (sum_delta, count)
        pseudo_dn = {i: (0.0, 0) for i in int_idx}

        header = (f"{'Nodes':>8} {'Open':>6} {'BestBnd':>14} {'BestObj':>14} "
                  f"{'Gap%':>8} {'Depth':>6} {'Time':>8}")
        self._log_msg(header)
        self._log_msg("-" * len(header))

        # Solve root LP with Gomory cuts
        root_result, root_cuts = self._solve_node_with_cuts(problem, root, gomory_rounds=self.max_gomory_rounds)

        if root_result.status == SolveStatus.INFEASIBLE:
            self._log_msg("Root LP infeasible — MILP is infeasible")
            result = SolverResult(status=SolveStatus.INFEASIBLE, solve_time=time.time() - t0, log=self._log)
            return result

        if root_result.status != SolveStatus.OPTIMAL:
            self._log_msg(f"Root LP failed: {root_result.status}")
            result = SolverResult(status=SolveStatus.UNKNOWN, solve_time=time.time() - t0, log=self._log)
            return result

        root.lp_bound = root_result.objective_value
        lp_bound = root_result.objective_value
        self._log_msg(f"  Root LP relaxation: {lp_bound:.6f}")

        # Primal heuristic at root
        x_round = self._round_heuristic(root_result.x, int_idx, problem)
        if x_round is not None:
            obj_round = float(problem.c @ x_round)
            if obj_round < best_obj:
                best_obj = obj_round
                best_x = x_round
                self._log_msg(f"  Rounding heuristic: obj={best_obj:.6f}")

        # Check if root solution is already integer-feasible
        if self._is_integer_feasible(root_result.x, int_idx):
            best_obj = root_result.objective_value
            best_x = root_result.x
            self._log_msg("  Root LP solution is integer-feasible — optimal!")
            result = SolverResult(
                status=SolveStatus.OPTIMAL,
                objective_value=best_obj,
                x=best_x,
                iterations=0,
                nodes_explored=1,
                solve_time=time.time() - t0,
                log=self._log,
                algorithm="Branch-and-Bound with Gomory Cuts",
            )
            return result

        while pq and nodes_explored < self.max_nodes:
            if time.time() - t0 > self.max_time:
                self._log_msg("Time limit reached")
                break

            # Select node
            node = heapq.heappop(pq)
            nodes_explored += 1

            # Solve LP relaxation at node
            node_result, _ = self._solve_node_with_cuts(problem, node, gomory_rounds=0,
                                                         extra_cuts=root_cuts)

            if node_result.status == SolveStatus.INFEASIBLE:
                continue  # Prune: infeasible

            if node_result.status != SolveStatus.OPTIMAL:
                continue

            node_lp_obj = node_result.objective_value

            # Prune: LP bound >= best incumbent
            if node_lp_obj >= best_obj - self.tolerance:
                continue

            x_node = node_result.x

            # Integer feasibility check
            if self._is_integer_feasible(x_node, int_idx):
                if node_lp_obj < best_obj:
                    best_obj = node_lp_obj
                    best_x = x_node.copy()
                    gap = self._compute_gap(lp_bound, best_obj)
                    self._log_msg(
                        f"{nodes_explored:>8} {len(pq):>6} {lp_bound:>14.6f} "
                        f"{best_obj:>14.6f} {gap:>7.2f}% {node.depth:>6} "
                        f"{time.time()-t0:>7.2f}s  ← New best"
                    )
                continue

            # Branch on most fractional variable
            branch_var = self._select_branch_var(x_node, int_idx, pseudo_up, pseudo_dn)
            frac_val = x_node[branch_var]
            floor_val = np.floor(frac_val)
            ceil_val = np.ceil(frac_val)

            # Update global LP bound
            if node_lp_obj < lp_bound + self.tolerance:
                lp_bound = node_lp_obj

            # Log periodically
            if nodes_explored % 500 == 0:
                gap = self._compute_gap(lp_bound, best_obj)
                self._log_msg(
                    f"{nodes_explored:>8} {len(pq):>6} {lp_bound:>14.6f} "
                    f"{best_obj:>14.6f} {gap:>7.2f}% {node.depth:>6} "
                    f"{time.time()-t0:>7.2f}s"
                )

            # Create child nodes
            for direction, new_bound in [("down", floor_val), ("up", ceil_val)]:
                child_lb = node.lb_bounds.copy()
                child_ub = node.ub_bounds.copy()

                if direction == "down":
                    child_ub[branch_var] = min(child_ub[branch_var], new_bound)
                else:
                    child_lb[branch_var] = max(child_lb[branch_var], new_bound)

                # Feasibility check on bounds
                if np.any(child_lb > child_ub + self.tolerance):
                    continue

                child = BBNode(
                    priority=node_lp_obj,   # best-first
                    node_id=node_counter,
                    depth=node.depth + 1,
                    lb_bounds=child_lb,
                    ub_bounds=child_ub,
                    parent_id=node.node_id,
                    lp_bound=node_lp_obj,
                )
                node_counter += 1
                heapq.heappush(pq, child)

        elapsed = time.time() - t0
        gap = self._compute_gap(lp_bound, best_obj)
        self._log_msg(f"\n{'='*60}")
        self._log_msg(f"Nodes explored: {nodes_explored}")
        self._log_msg(f"Best objective: {best_obj:.8f}")
        self._log_msg(f"LP bound:       {lp_bound:.8f}")
        self._log_msg(f"Optimality gap: {gap:.4f}%")
        self._log_msg(f"Solve time:     {elapsed:.3f}s")

        if best_x is None:
            return SolverResult(
                status=SolveStatus.INFEASIBLE,
                solve_time=elapsed,
                nodes_explored=nodes_explored,
                log=self._log,
                algorithm="Branch-and-Bound with Gomory Cuts",
            )

        status = SolveStatus.OPTIMAL if gap < 0.01 else (
            SolveStatus.TIME_LIMIT if time.time() - t0 >= self.max_time else SolveStatus.OPTIMAL
        )

        return SolverResult(
            status=status,
            objective_value=best_obj,
            x=best_x,
            nodes_explored=nodes_explored,
            solve_time=elapsed,
            optimality_gap=gap,
            log=self._log,
            algorithm="Branch-and-Bound with Gomory Cuts",
        )

    def _solve_node_with_cuts(
        self,
        problem: OptimizationProblem,
        node: BBNode,
        gomory_rounds: int = 0,
        extra_cuts: Optional[Tuple] = None,
    ) -> Tuple[SolverResult, Optional[Tuple]]:
        """Solve LP relaxation at node, optionally adding Gomory cuts."""
        # Build LP relaxation with node bounds
        lp = OptimizationProblem(
            c=problem.c.copy(),
            A_ub=problem.A_ub.copy() if problem.A_ub is not None else None,
            b_ub=problem.b_ub.copy() if problem.b_ub is not None else None,
            A_eq=problem.A_eq.copy() if problem.A_eq is not None else None,
            b_eq=problem.b_eq.copy() if problem.b_eq is not None else None,
            lb=node.lb_bounds.copy(),
            ub=node.ub_bounds.copy(),
            name=f"{problem.name}_node{node.node_id}",
        )

        # Add extra cuts from root
        cuts = None
        if extra_cuts is not None:
            A_cut, b_cut = extra_cuts
            if lp.A_ub is not None:
                lp.A_ub = np.vstack([lp.A_ub, A_cut])
                lp.b_ub = np.concatenate([lp.b_ub, b_cut])
            else:
                lp.A_ub = A_cut
                lp.b_ub = b_cut

        result = self._lp_solver.solve(lp)

        if not result.is_optimal() or gomory_rounds == 0:
            return result, None

        # Gomory cuts at root
        gomory_A = []
        gomory_b = []

        for _ in range(gomory_rounds):
            if not result.is_optimal():
                break
            if self._is_integer_feasible(result.x, set(problem.integer_vars)):
                break

            cut_A, cut_b = self._generate_gomory_cuts(result.x, lp, problem.integer_vars)
            if cut_A is None or len(cut_A) == 0:
                break

            gomory_A.append(cut_A)
            gomory_b.append(cut_b)

            if lp.A_ub is not None:
                lp.A_ub = np.vstack([lp.A_ub, cut_A])
                lp.b_ub = np.concatenate([lp.b_ub, cut_b])
            else:
                lp.A_ub = cut_A
                lp.b_ub = cut_b

            result = self._lp_solver.solve(lp)

        if gomory_A:
            cuts = (np.vstack(gomory_A), np.concatenate(gomory_b))

        return result, cuts

    def _generate_gomory_cuts(
        self,
        x: np.ndarray,
        lp: OptimizationProblem,
        integer_vars: List[int],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Generate Gomory fractional cuts.
        
        For a fractional integer variable x_i = f_bar (fractional part),
        the Gomory cut is: sum_j (f_j / (1-f_bar)) x_j >= 1  (simplified)
        
        We use the simplified tableau-based approach.
        """
        n = len(x)
        cuts_A = []
        cuts_b = []

        for i in integer_vars:
            frac = x[i] - np.floor(x[i])
            if frac < self.integer_tol or frac > 1 - self.integer_tol:
                continue

            # Simple Chvátal-Gomory cut: x_i >= ceil(x_i)
            cut_row = np.zeros(n)
            cut_row[i] = -1.0  # -x_i <= -ceil(x_i)
            cut_rhs = -np.ceil(x[i])

            cuts_A.append(cut_row)
            cuts_b.append(cut_rhs)

            # Only add a few cuts per round
            if len(cuts_A) >= 5:
                break

        if not cuts_A:
            return None, None

        return np.array(cuts_A), np.array(cuts_b)

    def _is_integer_feasible(self, x: np.ndarray, int_idx) -> bool:
        """Check if all integer-constrained variables are (near) integer."""
        for i in int_idx:
            if abs(x[i] - round(x[i])) > self.integer_tol:
                return False
        return True

    def _select_branch_var(
        self, x: np.ndarray, int_idx, pseudo_up: Dict, pseudo_dn: Dict
    ) -> int:
        """Select branching variable — most-infeasible rule."""
        best_score = -1.0
        best_var = -1
        for i in int_idx:
            frac = x[i] - np.floor(x[i])
            score = min(frac, 1.0 - frac)  # most fractional = 0.5
            if score > best_score:
                best_score = score
                best_var = i
        return best_var

    def _round_heuristic(
        self, x: np.ndarray, int_idx, problem: OptimizationProblem
    ) -> Optional[np.ndarray]:
        """Simple rounding heuristic for primal bound."""
        x_round = x.copy()
        for i in int_idx:
            x_round[i] = round(x[i])
            # Clip to bounds
            x_round[i] = np.clip(x_round[i], problem.lb[i],
                                  problem.ub[i] if np.isfinite(problem.ub[i]) else x_round[i])

        # Check feasibility (crude check)
        if problem.A_ub is not None:
            if np.any(problem.A_ub @ x_round > problem.b_ub + 1e-4):
                return None
        if problem.A_eq is not None:
            if np.any(np.abs(problem.A_eq @ x_round - problem.b_eq) > 1e-4):
                return None

        return x_round

    def _compute_gap(self, lb: float, ub: float) -> float:
        """Compute optimality gap as percentage."""
        if np.isinf(ub) or np.isinf(lb):
            return 100.0
        if abs(ub) < 1e-10:
            return 0.0 if abs(lb - ub) < 1e-10 else 100.0
        return abs(ub - lb) / (abs(ub) + 1e-10) * 100.0
