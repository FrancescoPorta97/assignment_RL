import torch
import numpy as np

def get_agent_status (tasks_to_mask: list, resource_to_fill: np.array,
                       num_tasks: int, cost_task_resource: np.array,
                       block_size: int = 127, slack: int = 1) -> tuple :  #to debug

    token_to_masks_task = [task +slack for task in tasks_to_mask]
    token_to_masks_pad = list(range(num_tasks+slack, block_size))
    token_to_masks = token_to_masks_task + token_to_masks_pad

    boolean_mask = torch.ones(block_size, dtype=torch.bool)
    if token_to_masks:  # guard against empty
        idx = torch.as_tensor(token_to_masks, dtype=torch.long)
        idx = idx[(idx >= 0) & (idx < block_size)]  # safety clamp
        boolean_mask[idx] = False
        boolean_mask = boolean_mask.view(1,block_size)

    cost_task_resource = np.hstack([resource_to_fill.reshape(resource_to_fill.size,1), cost_task_resource])
    cost_task_resource = torch.tensor(cost_task_resource)
    dim_1 = cost_task_resource.size()[0]
    dim_2 = cost_task_resource.size()[1]
    cost_task_resource = torch.transpose(cost_task_resource, 1, 0)
    cost_task_resource = cost_task_resource.view(1, dim_2, dim_1)
    rows_to_add = block_size - dim_2
    
    if rows_to_add > 0:
        zeros = torch.zeros((1, rows_to_add, dim_1), dtype=cost_task_resource.dtype)
        tokens = torch.cat([cost_task_resource, zeros], dim=1)
    else :
        tokens = cost_task_resource
    
    return tokens.to(torch.float32), boolean_mask.view(1, block_size)

def get_assignment_cost(cost_task_resource: np.array, assignment_soultion: dict) -> int:

    total_cost =0
    for resource_idx, tasks_assigned in assignment_soultion.items():

        costs_for_resource = [ cost_task_resource[int(resource_idx), task_idx] for task_idx in tasks_assigned]
        total_cost = total_cost + sum(costs_for_resource)
    
    return total_cost



        




