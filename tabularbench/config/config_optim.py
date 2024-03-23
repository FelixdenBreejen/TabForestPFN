from dataclasses import dataclass


@dataclass
class ConfigOptim():
    max_steps: int
    log_every_n_steps: int
    eval_every_n_steps: int
    batch_size: int
    gradient_accumulation_steps: int
    lr: float
    weight_decay: float
    beta1: float
    beta2: float
    warmup_steps: int
    cosine_scheduler: bool
    max_grad_norm: float
    use_pretrained_weights: bool
    path_to_weights: str