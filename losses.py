import torch
import torch.nn.functional as F

def compute_td_target(reward, gamma, next_value):
    # next_value should not backprop into the critic's target
    return reward + gamma * next_value.detach()

def compute_advantage(td_target, value):
    return td_target - value  # no detach here; reused for both losses

def actor_loss(log_prob_actions, advantages):
    # stop grads through the advantage
    return -(log_prob_actions * advantages.detach()).mean()

def critic_loss(td_targets, values):
    # 1/2 * (TD error)^2
    return 0.5 * (td_targets - values).pow(2).mean()

