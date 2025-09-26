from actor_critic_models import (
RlModel,
ActorConfig,
 CriticConfig
 )

from helpers import get_agent_status, get_assignment_cost
from losses import *
from environment_instance import Environment

#third party library imports
import torch
import torch.nn.functional as F
import torch.distributions as distributions
import torch.optim as optim
import random
import numpy as np
import wandb
import os

def sample_log_prob_action(logits):

    action_prob = F.softmax(logits, dim=-1)
    dist = distributions.Categorical(action_prob)
    action_idx = dist.sample()
    log_prob_action = dist.log_prob(action_idx)

    return action_idx, log_prob_action

def get_eligible_logits(policy_logits, eligible_actions):

    mask = torch.full_like(policy_logits, float("-inf"))
    mask[:, :, eligible_actions] = 0.0
    masked_logits = policy_logits + mask

    return masked_logits

def get_normalized_entropy(logits):

    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    
    # Shannon entropy per sample
    entropy = -(probs * log_probs).sum(dim=-1)
    
    # Maximum possible entropy = log(num_actions)
    num_actions = probs.size(-1)
    max_entropy = torch.log(torch.tensor(float(num_actions)))
    
    norm_entropy = entropy / max_entropy
    return norm_entropy.mean().item()

if __name__ == "__main__":

    #rl initializations
    n_resources = 4
    capacity = 8
    gamma = 0.99 
    CYCLES = 5000
    EPISODES = 16  # episodes for training steps
    TERMINATING_CONDITION = False
    
    #deep learning initializations
    LEARNING_RATE_ACTOR = 0.01
    LEARNING_RATE_CRITIC = 0.01
    BATCH_SIZE = 32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    block_size = 33   #+1 rl token 
    actor_model = RlModel(ActorConfig())
    critic_model = RlModel(CriticConfig())
    optimizer_actor = optim.Adam(actor_model.parameters(), lr = LEARNING_RATE_ACTOR)
    optimizer_critic = optim.Adam(critic_model.parameters(), lr = LEARNING_RATE_CRITIC)
    
    #wandb initializations
    wandb_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_key)
    wandb_config = dict(
    env="Assignment_RL",
    gamma=gamma,
    actor_lr=LEARNING_RATE_ACTOR,
    critic_lr=LEARNING_RATE_CRITIC,
    episodes_per_update=EPISODES,
    )

    wandb.init(project= "Assignment_RL", config= wandb_config)

    if torch.cuda.is_available():
        actor_model = torch.compile(actor_model)
        critic_model = torch.compile(critic_model)

    for cycle_idx in range(CYCLES):
        
        log_prob_actions, td_targets, advantages, values, episode_costs, entropies = [], [], [], [], [], []

        for episode_idx in range(EPISODES):   

            episode = Environment(resource_capacity=capacity, num_resources= n_resources)
            TERMINATING_CONDITION = False

            while (TERMINATING_CONDITION == False):

                if not episode.is_resource_available():
                    episode.update_resource()

                agent_input, mask = get_agent_status(tasks_to_mask = episode.tasks_to_mask,
                                                    resource_to_fill = episode.resource_to_fill,
                                                    num_tasks = episode.get_num_tasks(),
                                                    cost_task_resource= episode.cost_task_resource,
                                                    block_size= block_size)
                agent_input = agent_input.to(device)
                mask = mask.to(device)
                
                value = critic_model(agent_input, mask)[0]

                eligible_actions = episode.get_eligible_tasks()
                policy_logits = actor_model(agent_input, mask)[0]
                normalized_entropy = get_normalized_entropy(policy_logits)
                policy_logits = get_eligible_logits(policy_logits, eligible_actions)
                action_idx, log_prob_action = sample_log_prob_action(policy_logits)
                
                reward = episode.take_action(int(action_idx))  #environment update
                agent_input, mask = get_agent_status(tasks_to_mask = episode.tasks_to_mask,
                                                     resource_to_fill = episode.resource_to_fill,
                                                    num_tasks = episode.get_num_tasks(),
                                                    cost_task_resource= episode.cost_task_resource,
                                                    block_size= block_size)
                agent_input = agent_input.to(device)
                mask = mask.to(device)
                next_value = critic_model(agent_input, mask)[0]

                td_target = compute_td_target(torch.tensor(reward), gamma, next_value)
                advantage = compute_advantage(td_target, value)

                log_prob_actions.append(log_prob_action)
                td_targets.append(td_target)
                advantages.append(advantage)
                values.append(value)
                entropies.append(normalized_entropy)

                if len(episode.tasks_to_mask) == episode.get_num_tasks():
                    episode_cost = get_assignment_cost (episode.cost_task_resource, episode.assignment_solution)
                    episode_costs.append(episode_cost)
                    TERMINATING_CONDITION = True

        #update actor-critic networks
        N = len(values)
        assert len(log_prob_actions) == len(advantages) == len(td_targets) == N

        indices = list(range(N))
        random.shuffle(indices)

        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()

        print(f"Learning from {EPISODES*(cycle_idx+1)} episodes of experience in total")
        for start in range(0, N, BATCH_SIZE):
            batch_idx = indices[start:start + BATCH_SIZE]

            # slice each list consistently
            b_logp = [log_prob_actions[i] for i in batch_idx]
            b_adv  = [advantages[i]       for i in batch_idx]
            b_td   = [td_targets[i]       for i in batch_idx]
            b_val  = [values[i]           for i in batch_idx]

            a_loss = actor_loss(torch.stack(b_logp), torch.stack(b_adv))
            c_loss = critic_loss(torch.stack(b_td), torch.stack(b_val))
            a_loss.backward()
            c_loss.backward()
        
        optimizer_actor.step()
        optimizer_critic.step()

        print (f"The critic loss is {c_loss}")
        print (f"The actor loss is {a_loss}")
        print(f"The average value for {EPISODES} is {torch.stack([v.detach() for v in values]).mean().item()}")
        print(f"The average entropy of the policy for {EPISODES} is {np.array(entropies).mean()}")
        print(f"The average episode cost for {EPISODES} episodes was {np.array(episode_costs).mean()}")

        wandb.log({
        "experience_episodes": EPISODES * (cycle_idx +1),
        "avg_solution_cost": np.array(episode_costs).mean(),
        "avg_entropy": np.array(entropies).mean(),
        "avg_value": torch.stack([v.detach() for v in values]).mean().item(),
        "actor_loss": a_loss,
        "critic_loss": c_loss,
         })


            
            





