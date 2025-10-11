import torch 

def compute_td_target(reward, gamma, next_value):
    # next_value should not backprop into the critic's target
    return reward + gamma * next_value.detach()

def compute_advantage(td_target, value):
    return td_target - value  # no detach here; reused for both losses

def actor_loss(log_prob_actions, advantages):
    # stop grads through the advantage
    advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
    return -(log_prob_actions * advantages.detach()).mean()

def critic_loss(v_targets, values):
    # 1/2 * (TD error)^2
    return 0.5 * (v_targets.detach() - values).pow(2).mean()

def gae(episode_advantages: list, gamma, lam) -> list :

    T = len(episode_advantages)
    gae_advs = []
    for t in reversed(range(T)):
        if t == len(episode_advantages) -1:
            gae = episode_advantages[t] 
        else:
            gae = episode_advantages[t] + gamma * lam * gae
        
        gae_advs.append(gae)
    gae_advs.reverse()

    return gae_advs