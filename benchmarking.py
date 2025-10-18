from scipy.optimize import linear_sum_assignment
from environment_instance import Environment
import math
import numpy as np

def get_optimal_solution_cost(capacity, cost_matrix):
    C = np.repeat(cost_matrix, repeats=capacity, axis =0)
    rows, cols = linear_sum_assignment(C)
    cost = C[rows, cols].sum()
    return cost

if __name__ == "__main__":

    capacity = 8
    n_resources = 4
    cost_std_dev = math.sqrt((1/12))*0.5 
    environment = Environment(resource_capacity= capacity, num_resources= n_resources, cost_std_dev= cost_std_dev)
    cost_opt = get_optimal_solution_cost(capacity, environment.cost_task_resource)
    print(cost_opt)
