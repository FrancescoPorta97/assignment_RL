from scipy.optimize import linear_sum_assignment
from environment_instance import Environment
import math
import numpy as np
import os
from actor_critic_models import RlModel, ActorConfig
from helpers import (
    get_agent_status,
    get_eligible_logits,
    get_assignment_cost
)
import torch


def get_optimal_solution_cost(capacity, cost_matrix):
    C = np.repeat(cost_matrix, repeats=capacity, axis=0)
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

    #dev initializations
    model_version = "20251105_135350"
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    model_folder = os.path.join(CURRENT_DIR,"models", model_version)
    NUM_RUNS = 1000

    #ml initializations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_size = 33  # +1 rl token

    #rl initializations
    capacity = 8
    n_resources = 4
    cost_std_dev = math.sqrt((1 / 12)) * 0.5
    actor_model = RlModel(ActorConfig()).to(device)
    actor_model.load_state_dict(torch.load(os.path.join(model_folder, "actor.pt"), map_location=device))

    for run_idx in range(NUM_RUNS):
        
        environment = Environment(
            resource_capacity=capacity, num_resources=n_resources, cost_std_dev=cost_std_dev
        )

        #get rl solution cost
        for task in range(environment.get_num_tasks()):

            if not environment.is_resource_available():
                environment.update_resource()

            eligible_actions = environment.get_eligible_tasks()
            agent_input, mask = get_agent_status(
                        tasks_to_mask=environment.tasks_to_mask,
                        resource_to_fill=environment.resource_to_fill,
                        num_tasks=environment.get_num_tasks(),
                        cost_task_resource=environment.cost_task_resource,
                        block_size=block_size,
                        device=device,
                    )
            with torch.no_grad():
                policy_logits = actor_model(agent_input, mask)[0].squeeze(1)
            masked_logits = get_eligible_logits(policy_logits, eligible_actions)
            action = int(masked_logits.argmax(dim=-1))
            environment.take_action(action)
        
        cost_rl = get_assignment_cost(environment.cost_task_resource,
                                    environment.assignment_solution)
        cost_opt = get_optimal_solution_cost(capacity, environment.cost_task_resource)
        cost_greedy = get_greedy_solution_cost(capacity, environment.cost_task_resource)
        print("Run idx:", run_idx)
