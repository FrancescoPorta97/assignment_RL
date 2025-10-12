"""
Actor critic models
"""

import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
    
    def _build_mask(self, tasks_to_mask, B, T, device):
        """
        Build a boolean mask for attention of shape [B, 1, T, T].
        True  = allowed (not masked)
        False = masked (blocked)

        tasks_to_mask: BoolTensor [B, T], where True means "keep token",
                    False means "mask this token completely".
        """
        if tasks_to_mask is None:
            return None

        if tasks_to_mask.dtype == torch.bool:
            # tasks_to_mask: [B, T]
            rl_mask = torch.ones(B,1, dtype=torch.bool).to(device)
            tasks_to_mask = torch.cat([rl_mask, tasks_to_mask],dim=1)
            keep = tasks_to_mask.to(device)

            # start with all allowed
            mask = torch.ones((B, 1, T, T), dtype=torch.bool, device=device)

            for b in range(B):
                bad = (~keep[b]).nonzero(as_tuple=False).squeeze(-1)  # indices to eliminate
                if bad.numel() > 0:
                    mask[b, :, bad, :] = False  # eliminate as queries (rows)
                    mask[b, :, :, bad] = False  # eliminate as keys    (cols)

            return mask
        else:
            raise ValueError("tasks_to_mask must be a BoolTensor of shape [B, T]")

    def forward(self, x, tasks_to_mask):

        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        mask = self._build_mask(tasks_to_mask, B, T, x.device)  # True = masked
  
        y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask= mask, dropout_p=self.dropout if self.training else 0, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = SelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, tasks_to_mask = None):
        x = x + self.attn(self.ln_1(x), tasks_to_mask)
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class ActorConfig:
    block_size: int = 33
    output_size: int = 32
    resource_token_idx: int = 0
    n_layer: int = 12
    n_head: int = 1
    n_embd: int = 4
    dropout: float = 0.0
    bias: bool = False # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

@dataclass
class CriticConfig:
    block_size: int = 33
    output_size: int = 1
    resource_token_idx: int = 0
    n_layer: int = 12
    n_head: int = 1
    n_embd: int = 4
    dropout: float = 0.0
    bias: bool = False 

class RlModel(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.output_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.resource_vec = nn.Parameter(torch.zeros(config.n_embd))
        self.rl_token = nn.Parameter(torch.zeros(1, 1, config.n_embd))
        # Heads: policy scores are produced per token (n_embd->1),
        # value is predicted from the RL token (n_embd->1)
        self.policy_head = nn.Linear(config.n_embd, 1, bias=False)
        self.value_head = nn.Linear(config.n_embd, 1, bias=False)
    
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2f" % (self.get_num_params(),))

    def get_num_params(self):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())

        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, tasks_emb, tasks_to_mask=None):
        device = tasks_emb.device
        b, t, n = tasks_emb.size()  # tasks embeddings of shape (b, t, n_embd)

        # forward the GPT model itself
        idx = self.config.resource_token_idx
        is_special = (torch.arange(t, device=device) == idx).float().view(1, t, 1)  # [1,T,1]
        x = tasks_emb + is_special* self.resource_vec.to(device) # broadcast [1,1,d]

        rl = self.rl_token.expand(b, 1, n).to(device)
        x = torch.cat([rl, x], dim=1)
        x = self.transformer.drop(x)

        for block in self.transformer.h:
            x = block(x, tasks_to_mask)
        x = self.transformer.ln_f(x)

        # Actor returns per-token logits over the first `output_size` task tokens
        # (skip RL token at 0 and resource token at 1).
        if self.config.output_size > 1:
            start = 2
            end = 2 + self.config.output_size
            task_tokens = x[:, start:end, :]                      # [B, output_size, n_embd]
            logits = self.policy_head(task_tokens).squeeze(-1)    # [B, output_size]
            loss = None
            return logits, loss
        # Critic returns a scalar value from the RL token
        value = self.value_head(x[:, [0], :])                     # [B, 1, 1]
        loss = None
        return value, loss
    
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu



    

