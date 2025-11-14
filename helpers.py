import torch
import numpy as np
from losses import gae


def get_agent_status(
    tasks_to_mask: list,
    resource_to_fill: np.array,
    num_tasks: int,
    cost_task_resource: np.array,
    block_size: int = 127,
    slack: int = 1,
    device=None,
) -> tuple:

    token_to_masks_task = [task + slack for task in tasks_to_mask]
    token_to_masks_pad = list(range(num_tasks + slack, block_size))
    token_to_masks = token_to_masks_task + token_to_masks_pad

    boolean_mask = torch.ones(block_size, dtype=torch.bool, device=device)
    if token_to_masks:  # guard against empty
        idx = torch.as_tensor(token_to_masks, dtype=torch.long, device=device)
        idx = idx[(idx >= 0) & (idx < block_size)]  # safety clamp
        boolean_mask[idx] = False
        boolean_mask = boolean_mask.view(1, block_size)

    cost_task_resource = np.hstack(
        [resource_to_fill.reshape(resource_to_fill.size, 1), cost_task_resource]
    )
    cost_task_resource = torch.tensor(cost_task_resource, device=device)
    dim_1 = cost_task_resource.size()[0]
    dim_2 = cost_task_resource.size()[1]
    cost_task_resource = torch.transpose(cost_task_resource, 1, 0)
    cost_task_resource = cost_task_resource.view(1, dim_2, dim_1)
    rows_to_add = block_size - dim_2

    if rows_to_add > 0:
        zeros = torch.zeros((1, rows_to_add, dim_1), dtype=cost_task_resource.dtype, device=device)
        tokens = torch.cat([cost_task_resource, zeros], dim=1)
    else:
        tokens = cost_task_resource

    return tokens.to(torch.float32), boolean_mask.view(1, block_size)


def get_assignment_cost(cost_task_resource: np.array, assignment_soultion: dict) -> int:

    total_cost = 0
    for resource_idx, tasks_assigned in assignment_soultion.items():

        costs_for_resource = [
            cost_task_resource[int(resource_idx), task_idx] for task_idx in tasks_assigned
        ]
        total_cost = total_cost + sum(costs_for_resource)

    return total_cost


def gae_advantages(advantages: list, env_ids: list, gamma, lam):

    gae_advantages = [None] * len(advantages)
    episode_to_gae_advantages = {}
    env_iterable = range(len(set(env_ids)))

    for env_idx in env_iterable:

        episode_to_gae_advantages[env_idx] = gae(
            [x for x, mask in zip(advantages, env_ids) if mask == env_idx], gamma, lam
        )

    for env_idx in env_iterable:

        i = 0
        for adv_idx in range(len(advantages)):

            if env_ids[adv_idx] == env_idx:
                gae_advantages[adv_idx] = episode_to_gae_advantages[env_idx][i]
                i = i + 1

    return gae_advantages


def gae_td_targets(advantages: list, values: list):

    return [advantage + value for advantage, value in zip(advantages, values)]

def get_eligible_logits(policy_logits, eligible_actions):
    masked_logits = torch.full_like(policy_logits, float("-inf"))
    if isinstance(eligible_actions, (list, tuple)) and eligible_actions and isinstance(
        eligible_actions[0], (list, tuple)
    ):
        for idx, actions in enumerate(eligible_actions):
            if len(actions) == 0:
                continue
            masked_logits[idx, actions] = policy_logits[idx, actions]
    else:
        if isinstance(eligible_actions, (list, tuple)) and len(eligible_actions) == 0:
            return masked_logits
        masked_logits[..., eligible_actions] = policy_logits[..., eligible_actions]
    return masked_logits
