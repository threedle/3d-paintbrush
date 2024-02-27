from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class MeshConfig:
    """ Parameters for the mesh """
    path: str = "./data/hand.obj"
    dy: float = 0.25
    shape_scale: float = 0.6

@dataclass
class GuidanceConfig:
    """ Parameters for the score distillation guidance """
    object_name: str = "hand"
    style: str = "fancy gold"
    edit: str = "watch"
    prefix: str = ""
    localization_prompt: Optional[str] = None # "a 3d render of a gray hand with a yellow watch"
    negative_prompt: Optional[str] = None
    style_prompt: Optional[str] = None # "a 3d render of a gray hand with a fancy gold watch"
    background_prompt: Optional[str] = None # "a 3d render of a hand with a yellow watch"
    append_direction: bool = False
    cascaded: bool = True
    stage_I_weight: float = 1.0
    stage_II_weight: float = 0.1
    stage_II_weight_range: Tuple[float] = (0, 0.8)
    no_lerp_stage_II: bool = False
    anneal_t: bool = False
    batched_sd: bool = False
    third_loss: bool = True

@dataclass
class TextureConfig:
    """ Parameters for the texture mapping and texture """
    texture_map: bool = False
    texture_resolution: int = 1024
    inference_texture_resolution: int = 1024
    mlp_batch_size: int = 400000
    sample_points: bool = False
    anti_aliasing: bool = True
    bake_anti_aliasing: bool = False
    aa_scale: int = 5
    bake_aa_scale: int = 5
    overwrite_inverse_map: bool = False
    global_map_cache: bool = True
    primary_col: Tuple[float] = (0.8, 1.0, 0.0)
    secondary_col: Tuple[float] = (0.71, 0.71, 0.71)

@dataclass
class RenderConfig:
    render_size: int = 1024
    eval_grid_size: int = 512
    radius_range: Tuple[float] = (1.0, 1.5)
    angle_overhead: int = 30 # Set [0,angle_overhead] as the overhead region
    angle_front: int = 70
    elev_range: Tuple[int] = (0, 150)
    azim_range: Tuple[int] = (0, 360)
    init_elev: int = 0
    init_azim: int = 0
    white_background: bool = True
    background_aug: bool = True
    lighting: bool = True
    viz_elev: Tuple[int] = (30, 30, 30, 30, 30, 30, 90, -90)
    viz_azim: Tuple[int] = (-45, 0, 45, -90, 90, 180, 0, 180)
    viz_lights: Tuple[Tuple[float]] = (
        (1.5, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1., 1., 1., 1., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1.2, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1.2, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1., 1., 1., 1., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 0., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 0., 0., 0., 0., 0., 0.)
    )
    viz_radius: float = 1.5
    viz_style: bool = True
    eval_size: int = 10
    full_eval_size: int = 10
    inference_azim: int = None
    inference_elev: int = 30
    inference_texture_lighting: bool = True
    inference_viz_lights: Tuple[Tuple[float]] = (
        (1.5, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 1., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 0., 0., 0., 0., 0., 0.),
        (1.5, 0., 1., 0., 0., 0., 0., 0., 0.)
    )

@dataclass
class OptimizationConfig:
    lr: float = 1.e-4
    epochs: int = 5000
    optimizer: str = "adam"
    batch_size: int = 4
    seed: int = 0

@dataclass
class NetworkConfig:
    localization_depth: int = 4
    texture_depth: int = 4
    width: int = 256
    positional_encoding: bool = True
    localization_sigma: float = 12.0
    texture_sigma: float = 24.0
    clamp: str = "sigmoid"
    background_mlp: bool = True

@dataclass
class LogConfig:
    log_interval: int = 100
    log_interval_viz: int = 150
    log_interval_real_renders: int = 500
    exp_dir: str = "./results/demo"
    inference: bool = False
    inference_threshold: float = 0.5
    dataloader_size: int = 100
    num_workers: int = 0
    model_path: str = None
    log_all_intermediate_results: bool = False

@dataclass
class TrainConfig:
    mesh: MeshConfig = field(default_factory = MeshConfig)
    guidance: GuidanceConfig = field(default_factory = GuidanceConfig)
    texture: TextureConfig = field(default_factory = TextureConfig)
    render: RenderConfig = field(default_factory = RenderConfig)
    optim: OptimizationConfig = field(default_factory = OptimizationConfig)
    network: NetworkConfig = field(default_factory = NetworkConfig)
    log: LogConfig = field(default_factory = LogConfig)
