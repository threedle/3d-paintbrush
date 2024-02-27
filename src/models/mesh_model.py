import os

import kaolin as kal
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from loguru import logger

from .mesh import Mesh
from .render import Renderer
from pathlib import Path
from models.neural_highlighter import NeuralHighlighter
from models.neural_style_field import NeuralStyleField
from utils import inverse_map_batched

class MeshModel(nn.Module):
    def __init__(
        self,
        cfg: dict,
        render_grid_size: int = 64,
        texture_resolution: int = 1024,
        device=torch.device("cuda"),
    ):
        super().__init__()
        self.device = device
        self.cfg = cfg
        self.dy = self.cfg.mesh.dy
        self.mesh_scale = self.cfg.mesh.shape_scale
        self.texture_resolution = texture_resolution
        self.renderer = Renderer(device=self.device, dim=(render_grid_size, render_grid_size))
        self.mesh = self.init_mesh()

        # Initialize neural textures
        self.texture_img = torch.randn(1, 3, self.texture_resolution, self.texture_resolution).to(self.device)
        self.background_img = torch.randn(1, 3, self.texture_resolution, self.texture_resolution).to(self.device)
        self.neural_style_field = self.init_neural_style_field()
        if self.cfg.network.background_mlp:
            self.bg_nsf = self.init_neural_style_field()
        self.texture_prob = torch.zeros(self.texture_resolution, self.texture_resolution).to(self.device)
        self.neural_highlighter = self.init_neural_highlighter()

        # Initialize texture map
        self.surface_points, self.texel_indices = self.init_texture_map()
        self.face_attributes = kal.ops.mesh.index_vertices_by_faces(
            self.vt.unsqueeze(0),
            self.ft.long()).detach()

        self.backgrounds = torch.FloatTensor([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [1, 1, 1]]).to(self.device)

    def init_mesh(self):
        mesh = Mesh(self.cfg.mesh.path, self.device)
        mesh.normalize_mesh(inplace=True, target_scale=self.mesh_scale, dy=self.dy)

        return mesh

    def init_neural_highlighter(self):
        nh = NeuralHighlighter(
                self.cfg.network.localization_depth,
                self.cfg.network.width,
                2, # output dim
                input_dim=3,
                positional_encoding=self.cfg.network.positional_encoding,
                sigma=self.cfg.network.localization_sigma,
        )
        return nh
    
    def init_neural_style_field(self):
        nsf = NeuralStyleField(
                self.cfg.network.texture_depth,
                self.cfg.network.width,
                3, # output dim
                input_dim=3,
                positional_encoding=self.cfg.network.positional_encoding,
                sigma=self.cfg.network.texture_sigma,
                clamp=self.cfg.network.clamp,
        )
        return nsf
    
    def init_texture_map(self):
        logger.info("Using xatlas texture map")
        self.vt, self.ft = self.init_xatlas_texture_map()
        
        if self.cfg.texture.global_map_cache:
            mesh_path = Path(self.cfg.mesh.path)
            cache_path = Path('data/inverse_map_cache') / mesh_path.stem / Path(f"{self.cfg.texture.aa_scale}@{self.cfg.texture.texture_resolution}")
            if self.cfg.texture.bake_anti_aliasing:
                cache_path = cache_path + '_bake_aa'
        else:
            cache_path = Path(self.cfg.log.exp_dir)
        sp_cache, ti_cache = cache_path / 'surface_points.pth', cache_path / 'texel_indices.pth'
        if (not self.cfg.texture.overwrite_inverse_map) and sp_cache.exists() and ti_cache.exists():
            logger.info("Loading inverse map from cache")
            surface_points = torch.load(sp_cache).to(self.device)
            texel_indices = torch.load(ti_cache).to(self.device)
        else:
            logger.info("Computing inverse map and caching for later. This may take a few minutes, but is only required once per mesh. Consider downloading our precomputed inverse maps as specified in the README to skip this step.")
            triangles = self.vt[self.ft.long()]
            tolerance = 1e-6 # Hard coded for now
            pooling_op = torch.nn.MaxPool2d(kernel_size=self.cfg.texture.aa_scale, stride=self.cfg.texture.aa_scale, return_indices=True)
            surface_points, texel_indices = inverse_map_batched(
                triangles,
                self.texture_resolution,
                self.mesh,
                tolerance=tolerance,
                anti_aliasing=self.cfg.texture.anti_aliasing,
                aa_scale=self.cfg.texture.aa_scale,
                bake_aa_scale=self.cfg.texture.bake_aa_scale,
                pooling_operation=pooling_op,
                device=self.device,
            )
            os.makedirs(cache_path, exist_ok=True)
            torch.save(surface_points.cpu(), sp_cache)
            torch.save(texel_indices.cpu(), ti_cache)
            logger.info("Finished computing and caching inverse map")

        logger.info("Initialized texture map")
        return surface_points.detach(), texel_indices.detach()

    def init_xatlas_texture_map(self):
        cache_path = Path(self.cfg.log.exp_dir) / 'uv_map'
        Path(cache_path).mkdir(parents=True, exist_ok=True)
        vt_cache, ft_cache = cache_path / 'vt.pth', cache_path / 'ft.pth'
        if vt_cache.exists() and ft_cache.exists():
            logger.info("Loading xatlas texture map from cache")
            vt = torch.load(vt_cache).cuda()
            ft = torch.load(ft_cache).cuda()
        else:
            # unwrap uvs
            import xatlas
            v_np = self.mesh.vertices.cpu().numpy()
            f_np = self.mesh.faces.int().cpu().numpy()
            logger.info(f'Running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')
            atlas = xatlas.Atlas()
            atlas.add_mesh(v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 4
            atlas.generate(chart_options=chart_options)
            vmapping, ft_np, vt_np = atlas[0]  # [N], [M, 3], [N, 2]
            vt = torch.from_numpy(vt_np.astype(np.float32)).float().cuda()
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().cuda()
            os.makedirs(cache_path, exist_ok=True)
            torch.save(vt.cpu(), vt_cache)
            torch.save(ft.cpu(), ft_cache)
        return vt, ft

    def forward(self, x):
        return_dict = {
            "pred_rgb": self.neural_style_field(x),
            "pred_prob": self.neural_highlighter(x)[:, 0]
        }
        if self.cfg.network.background_mlp:
            return_dict["pred_rgb_bg"] = self.bg_nsf(x)
        return return_dict

    def get_params(self):
        params = [*self.neural_style_field.parameters()]
        if self.cfg.network.background_mlp:
            params += [*self.bg_nsf.parameters()]
        params += [*self.neural_highlighter.parameters()]
        return params

    # adapted from https://github.com/eladrich/latent-nerf
    @torch.no_grad()
    def export_mesh(self, path, texture, file_name=''):
        v, f = self.mesh.vertices, self.mesh.faces.int()
        h0, w0 = 256, 256

        # v, f: torch Tensor
        v_np = v.cpu().numpy()  # [N, 3]
        f_np = f.cpu().numpy()  # [M, 3]

        vt_np = self.vt.detach().cpu().numpy()
        ft_np = self.ft.detach().cpu().numpy()

        # save the texture image
        torchvision.utils.save_image(texture, os.path.join(path, f'{file_name}_albedo.png'))

        # save obj (v, vt, f /)
        obj_file = os.path.join(path, f'{file_name}_mesh.obj')
        mtl_file = os.path.join(path, f'{file_name}_mesh.mtl')

        logger.info(f'writing obj mesh to {obj_file}')
        with open(obj_file, "w") as fp:
            fp.write(f'mtllib {file_name}_mesh.mtl \n')

            for v in v_np:
                fp.write(f'v {v[0]} {v[1]} {v[2]} \n')

            for v in vt_np:
                fp.write(f'vt {v[0]} {v[1]} \n')

            fp.write(f'usemtl mat0 \n')
            for i in range(len(f_np)):
                fp.write(
                    f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

        with open(mtl_file, "w") as fp:
            fp.write(f'newmtl mat0 \n')
            fp.write(f'Ka 1.000000 1.000000 1.000000 \n')
            fp.write(f'Kd 1.000000 1.000000 1.000000 \n')
            fp.write(f'Ks 0.000000 0.000000 0.000000 \n')
            fp.write(f'Tr 1.000000 \n')
            fp.write(f'illum 1 \n')
            fp.write(f'Ns 0.000000 \n')
            fp.write(f'map_Kd {file_name}_albedo.png \n')

    def render(self, elev, azim, radius, lights=None, test=False, dims=None, texture_lighting=False):
        rtn = []
        rtn.append(self.render_texture(elev, azim, radius, self.binary_texture_img, lighting=self.cfg.render.lighting, lights=lights))
        if texture_lighting:
            rtn.append(self.render_texture(elev, azim, radius, self.masked_texture_img, lighting=self.cfg.render.lighting, lights=lights))
        else:
            rtn.append(self.render_texture(elev, azim, radius, self.masked_texture_img))
        rtn.append(self.render_texture(elev, azim, radius, self.masked_background_texture_img))
        return rtn

    def render_vertices(self, elev, azim, radius, colors):
        pred_features, mask = self.renderer.render_mesh(self.mesh, colors, elev, azim, radius)
        mask = mask.detach()
        pred_map = (pred_features * mask) + (1 * (1 - mask))
        rtn = {'image': pred_map, 'mask': mask, 'foreground': pred_features}
        return rtn

    def render_texture(self, elev, azim, radius, texture, background_aug=False, lighting=False, lights=None):
        pred_features, mask = self.renderer.render_texture(
                                        self.mesh.vertices,
                                        self.mesh.faces,
                                        self.face_attributes,
                                        texture,
                                        elev=elev,
                                        azim=azim,
                                        radius=radius,
                                        look_at_height=self.dy,
                                        lighting=lighting,
                                        lights=lights,
                                    )
        mask = mask.detach()
        if self.cfg.render.white_background:
            pred_map = (pred_features * mask) + (1 * (1 - mask))
            rtn = {'image': pred_map, 'mask': mask, 'foreground': pred_features}
        else:
            background = self.backgrounds[torch.randint(len(self.backgrounds), (1,))]
            pred_map = (pred_features * mask) + (background[:, :, None, None].repeat(1, 1, 256, 256) * (1 - mask))
            rtn = {'image': pred_map, 'mask': mask, 'foreground': pred_features}
        return rtn