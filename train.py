from actor_critic_models import (
    RlModel,
    ActorConfig,
    CriticConfig,
)

from helpers import get_agent_status, get_assignment_cost
from losses import *
from environment_instance import Environment

# third party library imports
import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim
import random
import numpy as np
import wandb
import math
import os

def sample_log_prob_action(logits):
    action_prob = F.softmax(logits, dim=-1)
    dist = distributions.Categorical(action_prob)
    action_idx = dist.sample()
    log_prob_action = dist.log_prob(action_idx)
    return action_idx, log_prob_action

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

def get_normalized_entropy(logits):
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)
    num_actions = probs.size(-1)
    max_entropy = torch.log(
        torch.tensor(float(num_actions), device=logits.device, dtype=logits.dtype)
    )
    return entropy / max_entropy

if __name__ == "__main__":

    # rl initializations
    n_resources = 4
    cost_std_dev = math.sqrt((1/12))*0.5  #about half of uniform variance
    capacity = 8
    gamma = 0.99
    CYCLES = 5000
    EPISODES = 16  # episodes for training steps

    # deep learning initializations
    LEARNING_RATE_ACTOR = 1e-2
    LEARNING_RATE_CRITIC = 1e-3
    BATCH_SIZE = 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    block_size = 33  # +1 rl token
    actor_model = RlModel(ActorConfig()).to(device)
    critic_model = RlModel(CriticConfig()).to(device)

    if torch.cuda.is_available():
        print("Using GPU...")
        actor_model = torch.compile(actor_model)
        critic_model = torch.compile(critic_model)
    else:
        print("Using CPU...")

    optimizer_actor = optim.Adam(actor_model.parameters(), lr=LEARNING_RATE_ACTOR)
    optimizer_critic = optim.Adam(critic_model.parameters(), lr=LEARNING_RATE_CRITIC)

    # wandb initializations
    wandb_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    wandb_config = dict(
        env="Assignment_RL",
        gamma=gamma,
        actor_lr=LEARNING_RATE_ACTOR,
        critic_lr=LEARNING_RATE_CRITIC,
        episodes_per_update=EPISODES,
        cost_std_dev = cost_std_dev
    )

    wandb.init(project="Assignment_RL", config=wandb_config)

    for cycle_idx in range(CYCLES):

        log_prob_actions, td_targets, advantages, values, episode_costs, entropies = (
            [],
            [],
            [],
            [],
            [],
            [],
        )

        environments = [
            Environment(resource_capacity=capacity, num_resources=n_resources, cost_std_dev= cost_std_dev)
            for _ in range(EPISODES)
        ]
        done_flags = [False] * EPISODES

        while not all(done_flags):
            agent_inputs = []
            masks = []
            eligible_actions_batch = []
            env_indices = []

            for env_idx, env in enumerate(environments):
                if done_flags[env_idx]:
                    continue

                if not env.is_resource_available():
                    env.update_resource()

                agent_input, mask = get_agent_status(
                    tasks_to_mask=env.tasks_to_mask,
                    resource_to_fill=env.resource_to_fill,
                    num_tasks=env.get_num_tasks(),
                    cost_task_resource=env.cost_task_resource,
                    block_size=block_size,
                )
                agent_inputs.append(agent_input)
                masks.append(mask)
                eligible_actions_batch.append(env.get_eligible_tasks())
                env_indices.append(env_idx)

            if not env_indices:
                break

            agent_batch = torch.cat(agent_inputs, dim=0).to(device)
            mask_batch = torch.cat(masks, dim=0).to(device)

            value_batch = critic_model(agent_batch, mask_batch)[0].view(-1)
            policy_logits_batch = actor_model(agent_batch, mask_batch)[0].squeeze(1)

            entropy_values = get_normalized_entropy(policy_logits_batch).detach().cpu()
            masked_logits_batch = get_eligible_logits(
                policy_logits_batch, eligible_actions_batch
            )

            action_batch, log_prob_batch = sample_log_prob_action(masked_logits_batch)

            records = []
            next_inputs = []
            next_masks = []
            next_record_indices = []

            for local_idx, env_idx in enumerate(env_indices):
                env = environments[env_idx]
                action = int(action_batch[local_idx].item())
                reward = env.take_action(action)

                record = {
                    "value": value_batch[local_idx],
                    "log_prob": log_prob_batch[local_idx],
                    "reward": reward,
                    "entropy": float(entropy_values[local_idx].item()),
                    "next_value": torch.zeros_like(value_batch[local_idx]),
                }

                if len(env.tasks_to_mask) == env.get_num_tasks():
                    episode_cost = get_assignment_cost(
                        env.cost_task_resource, env.assignment_solution
                    )
                    episode_costs.append(episode_cost)
                    done_flags[env_idx] = True
                else:
                    agent_input_next, mask_next = get_agent_status(
                        tasks_to_mask=env.tasks_to_mask,
                        resource_to_fill=env.resource_to_fill,
                        num_tasks=env.get_num_tasks(),
                        cost_task_resource=env.cost_task_resource,
                        block_size=block_size,
                    )
                    next_inputs.append(agent_input_next)
                    next_masks.append(mask_next)
                    next_record_indices.append(len(records))

                records.append(record)

            if next_inputs:
                next_batch_input = torch.cat(next_inputs, dim=0).to(device)
                next_batch_mask = torch.cat(next_masks, dim=0).to(device)
                next_value_batch = (
                    critic_model(next_batch_input, next_batch_mask)[0].view(-1)
                )

                for idx, record_idx in enumerate(next_record_indices):
                    records[record_idx]["next_value"] = next_value_batch[idx]

            for record in records:
                reward_tensor = torch.tensor(
                    record["reward"], dtype=torch.float32, device=device
                )
                td_target = compute_td_target(reward_tensor, gamma, record["next_value"])
                advantage = compute_advantage(td_target, record["value"])

                log_prob_actions.append(record["log_prob"])
                td_targets.append(td_target)
                advantages.append(advantage)
                values.append(record["value"])
                entropies.append(record["entropy"])

        N = len(values)
        assert len(log_prob_actions) == len(advantages) == len(td_targets) == N

        indices = list(range(N))
        random.shuffle(indices)

        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()

        print(f"Learning from {EPISODES * (cycle_idx + 1)} episodes of experience in total")

        retain_graph = True
        for start in range(0, N, BATCH_SIZE):
            batch_idx = indices[start : start + BATCH_SIZE]

            b_logp = torch.stack([log_prob_actions[i] for i in batch_idx])
            b_adv = torch.stack([advantages[i] for i in batch_idx])
            b_td = torch.stack([td_targets[i] for i in batch_idx])
            b_val = torch.stack([values[i] for i in batch_idx])

            a_loss = actor_loss(b_logp, b_adv)
            c_loss = critic_loss(b_td, b_val)
            if start == N-BATCH_SIZE:
                retain_graph= False
            a_loss.backward(retain_graph=retain_graph)
            c_loss.backward(retain_graph=retain_graph)

        optimizer_actor.step()
        optimizer_critic.step()

        print(f"The critic loss is {c_loss}")
        print(f"The actor loss is {a_loss}")
        print(
            f"The average value for {EPISODES} is {torch.stack([v.detach() for v in values]).mean().item()}"
        )
        print(
            f"The average entropy of the policy for {EPISODES} is {np.array(entropies).mean()}"
        )
        print(
            f"The average episode cost for {EPISODES} episodes was {np.array(episode_costs).mean()}"
        )

        wandb.log(
            {
                "experience_episodes": EPISODES * (cycle_idx + 1),
                "avg_solution_cost": np.array(episode_costs).mean(),
                "avg_entropy": np.array(entropies).mean(),
                "avg_value": torch.stack([v.detach() for v in values]).mean().item(),
                "actor_loss": a_loss,
                "critic_loss": c_loss,
            }
        )
