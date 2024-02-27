from dataclasses import dataclass
from typing import Optional

@dataclass
class GuidanceConfig:
    """ Config for score distillation guidance """
    # Stage I model name
    model_name: str = "DeepFloyd/IF-I-XL-v1.0"
    # Stage II model name
    stage_II_model_name: str = "DeepFloyd/IF-II-L-v1.0"
    # CFG guidance scale
    guidance_scale: float = 20.0
    # Whether or not to use half precision weights
    half_precision_weights: bool = True
    # Gradient clipping value
    grad_clip_val: Optional[float] = None
    # Whether or not to use cpu offloading
    cpu_offload: bool = False
