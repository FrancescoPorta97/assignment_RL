from scipy.optimize import linear_sum_assignment
from environment_instance import Environment
import math
import numpy as np

def get_optimal_solution_cost(capacity, cost_matrix):
    C = np.repeat(cost_matrix, repeats=capacity, axis =0)
    rows, cols = linear_sum_assignment(C)
    cost = C[rows, cols].sum()
    return cost

def get_greedy_solution_cost(capacity: int, cost_matrix: np.ndarray) -> float:
    """
    Greedy capacitated assignment:
    For each resource, pick its cheapest remaining task, up to `capacity` tasks per resource.
    Returns the total cost.
    """
    C = np.asarray(cost_matrix, dtype=float)
    W = C.copy()

    n_resources, n_tasks = W.shape
    if n_resources * capacity > n_tasks:
        raise ValueError(
            f"Infeasible: total capacity {n_resources * capacity} exceeds number of tasks {n_tasks}"
        )
    chosen_tasks = set()
    solution = []

    for r in range(n_resources):
        taken = 0
        while taken < capacity:
            row = W[r, :]

            if not np.isfinite(row).any():
                # Nothing left this row can take (shouldn't happen if totals match, but safe)
                break

            t = int(np.argmin(row))        
            if t in chosen_tasks:

                W[:, t] = np.inf
                continue

            chosen_tasks.add(t)
            solution.append((r, t))
            # Make this task unavailable for all other resources
            W[:, t] = np.inf
            taken += 1

    total_cost = float(sum(C[r, t] for r, t in solution))
    return total_cost

if __name__ == "__main__":

    capacity = 8
    n_resources = 4
    cost_std_dev = math.sqrt((1/12))*0.5 
    environment = Environment(resource_capacity= capacity,
                               num_resources= n_resources,
                                 cost_std_dev= cost_std_dev)
    cost_opt = get_optimal_solution_cost(capacity, environment.cost_task_resource)
    cost_greedy = get_greedy_solution_cost(capacity, environment.cost_task_resource)
    print(cost_opt)
    print(cost_greedy)
