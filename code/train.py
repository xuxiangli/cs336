import torch
import math
import os
import typing
import numpy as np

from collections.abc import Callable
from typing import Optional


def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    assert logits.shape[:-1] == target.shape
    target = target.unsqueeze(dim=-1)

    logits_max = torch.amax(logits, dim=-1, keepdim=True)
    logsumexp = torch.log(
        torch.sum(torch.exp(logits - logits_max), dim=-1, keepdim=True)
    )

    loss = logsumexp - (torch.gather(logits, dim=-1, index=target) - logits_max)
    return loss.mean()


class AdamW(torch.optim.Optimizer):
    """
    """

    def __init__(self, params, lr, betas, weight_decay, eps=1e-8):
        beta1, beta2 = betas
        defaults = {
            "lr": lr,
            "beta1": beta1,
            "beta2": beta2,
            "lam": weight_decay,
            "eps": eps
        }
        super().__init__(params, defaults)


    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lam = group["lam"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 1)
                if t == 1:
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)

                grad = p.grad.data
                state["m"] = beta1*state["m"] + (1 - beta1)*grad
                state["v"] = beta2*state["v"] + (1 - beta2)*grad**2
                
                lr_t = lr * math.sqrt(1 - beta2**t)/(1 - beta1**t)

                p.data -= lr*lam*p.data
                p.data -= lr_t*state["m"]/(torch.sqrt(state["v"]) + eps)

                state["t"] = t + 1

        return loss


def cosine_annealing_learning_rate_schedule(
    t, alpha_max, alpha_min, t_w, t_c
):
    if t < t_w:
        return alpha_max*t/t_w
    elif t >= t_c:
        return alpha_min

    return alpha_min + 1/2 * (1 + math.cos((t-t_w)/(t_c-t_w)*math.pi)) * (alpha_max - alpha_min)


def gradient_clipping_l2(params, max_norm, eps=1e-6):
    norm = 0.
    params_list = []

    for p in params:
        if p.grad is None:
            continue

        norm += torch.sum(p.grad.data**2)
        params_list.append(p)

    norm = math.sqrt(norm)
    
    if norm > max_norm:
        factor = max_norm / (norm + eps)
        for p in params_list:
            p.grad.mul_(factor)


def data_loading(dataset, batch_size, context_length, device=None):
    assert len(dataset.shape) == 1
    length = dataset.shape[0]
    assert length > context_length

    start = np.random.randint(0, length - context_length, (batch_size,)).reshape(-1, 1)
    index = start + np.arange(context_length)

    inputs = torch.as_tensor(dataset[index], device=device, dtype=torch.int64)
    targets = torch.as_tensor(dataset[index+1], device=device, dtype=torch.int64)

    return inputs, targets


def save_checkpoint(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, 
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
    state = {}
    state["model"] = model.state_dict()
    state["optimizer"] = optimizer.state_dict()
    state["it"] = iteration

    if type(out) is str:
        torch.save(state, out + ".tmp")
        os.replace(out + ".tmp", out)
    else:
        torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], 
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
):
    state = torch.load(src)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    iteration = state["it"]
    return iteration