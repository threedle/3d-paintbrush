import random
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


anti_alias_avgpool = torch.nn.AvgPool2d(kernel_size=3, stride=3)
anti_alias_maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=3, return_indices=True)

def get_view_direction(elev, azim, overhead, front):
    # adapted from https://github.com/eladrich/latent-nerf
    #                   azim [B,];          elev: [B,]
    # front = 0         [0, front)
    # side (left) = 1   [front, 180)
    # back = 2          [180, 180+front)
    # side (right) = 3  [180+front, 360)
    # top = 4                               [0, overhead]
    # bottom = 5                            [180-overhead, 180]
    res = torch.zeros(elev.shape[0], dtype=torch.long)
    # first determine by phis

    # Convert elevation and azimuth to polar coordinates
    thetas = (np.pi / 2) - elev
    phis = azim

    # res[(phis < front)] = 0
    res[(phis >= (2 * np.pi - front / 2)) & (phis < front / 2)] = 0

    # res[(phis >= front) & (phis < np.pi)] = 1
    res[(phis >= front / 2) & (phis < (np.pi - front / 2))] = 1

    # res[(phis >= np.pi) & (phis < (np.pi + front))] = 2
    res[(phis >= (np.pi - front / 2)) & (phis < (np.pi + front / 2))] = 2

    # res[(phis >= (np.pi + front))] = 3
    res[(phis >= (np.pi + front / 2)) & (phis < (2 * np.pi - front / 2))] = 3
    # override by thetas
    res[thetas <= overhead] = 4
    res[thetas >= (np.pi - overhead)] = 5
    return res

def make_path(path: Path) -> Path:
    path.mkdir(exist_ok=True,parents=True)
    return path

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True

def inverse_map_batched(
    triangles: torch.FloatTensor,
    n: int,
    mesh,
    tolerance: float = 1e-6,
    batch_size: int = 5000,
    anti_aliasing: bool = True,
    bake_anti_aliasing: bool = False,
    aa_scale: int = 3,
    bake_aa_scale: int = 3,
    pooling_operation=anti_alias_maxpool,
    device: torch.device = torch.device("cpu"),
):
    from tqdm import tqdm
    from loguru import logger

    if anti_aliasing:
        n *= aa_scale
    if bake_anti_aliasing:
        n *= bake_aa_scale

    # Get texel points
    p = get_texels(n, device=device)

    # scale triangles to pixel coordinates
    scaled_triangles = triangles * n

    with torch.no_grad():
        surface_points_full = []
        pt_idx_full = []
        tri_idx_full = []
        logger.info(f"Computing map from texels to surface points... Caching for future runs.")
        for i in tqdm(range(0, p.shape[0], batch_size)):
            batch_p = p[i:i+batch_size]
            batch_barycentric_coords = compute_barycentric_coords(scaled_triangles, batch_p)

            # Determinte which points are inside a triangle
            valid_points = torch.all((batch_barycentric_coords >= -tolerance) & (batch_barycentric_coords <= 1+tolerance), dim=2)

            # Get indices of valid points and corresponding triangles
            pt_idx, tri_idx = valid_points.nonzero(as_tuple=True)

            # Select valid barycentric coordinates
            valid_barycentric_coords = batch_barycentric_coords[pt_idx, tri_idx]

            # Select corresponding triangle vertices for each valid point
            vt_idx = mesh.faces[tri_idx].long()

            # Compute 3D coordinates for valid points
            surface_points = torch.einsum("ij,ijk->ik", valid_barycentric_coords, mesh.vertices[vt_idx])

            surface_points_full.append(surface_points)
            pt_idx_full.append(pt_idx+i)
            if anti_aliasing: # and not bake_anti_aliasing:
                # save triangle indices
                tri_idx_full.append(tri_idx)
        surface_points = torch.cat(surface_points_full, dim=0)
        pt_idx = torch.cat(pt_idx_full, dim=0)

        if anti_aliasing: # and not bake_anti_aliasing:
            # get anti-aliased valid points
            tri_idx = torch.cat(tri_idx_full, dim=0)
            texture_flat = torch.zeros((n**2), dtype=torch.float32).to(device)
            texture_flat[pt_idx] = 1
            texture = texture_flat.reshape(n, n)
            aa_texture, max_indices = pooling_operation(texture.unsqueeze(0).unsqueeze(0))
            flat_aa_texture = aa_texture.flatten()
            aa_pt_idx = torch.where(flat_aa_texture)[0]

            triangles_flat = -1 * torch.ones((n**2), dtype=torch.long).to(device)
            triangles_flat[pt_idx] = tri_idx

            # get maxpooled indices
            max_indices = max_indices.flatten()
            flat_aa_triangles = triangles_flat[max_indices]

            # get triangles
            aa_triangle_idx = flat_aa_triangles[torch.where(flat_aa_triangles!=-1)[0]]

            # rescale n
            n = n//aa_scale
            # Get texel points
            p = get_texels(n, device=device)
            # scale triangles to pixel coordinates
            scaled_triangles = triangles * n

            # get aa surface points
            aa_valid_points = p[aa_pt_idx]
            aa_surface_points_full = []

            for i in tqdm(range(0, aa_valid_points.shape[0], batch_size)):
                batch_points = aa_valid_points[i:i+batch_size]
                batch_tri_idx = aa_triangle_idx[i:i+batch_size]
                batch_tris = scaled_triangles[batch_tri_idx]

                # Compute the barycentric coordinates for valid points+triangles
                batch_barycentric_coords = compute_barycentric_coords(batch_tris, batch_points)

                # Select valid barycentric coordinates
                first_dim_idx = torch.arange(batch_barycentric_coords.shape[0])
                second_dim_idx = torch.arange(batch_barycentric_coords.shape[1])
                aa_valid_barycentric_coords = batch_barycentric_coords[first_dim_idx, second_dim_idx]

                # Select corresponding triangle vertices for each valid point
                vt_idx = mesh.faces[batch_tri_idx].long()

                # Compute 3D coordinates for valid points
                aa_surface_points = torch.einsum("ij,ijk->ik", aa_valid_barycentric_coords, mesh.vertices[vt_idx])

                aa_surface_points_full.append(aa_surface_points)
            aa_surface_points = torch.cat(aa_surface_points_full, dim=0)
            surface_points = aa_surface_points
            pt_idx = aa_pt_idx


    return surface_points, pt_idx

def bake_surface_features(
    features,
    texel_indices,
    texture_image,
    anti_aliasing=False,
    aa_scale=3,
    pooling_operation=anti_alias_avgpool,
    relative_init=None,
):
    if anti_aliasing:
        texture_image = F.interpolate(
            texture_image.unsqueeze(0).unsqueeze(0),
            scale_factor=aa_scale,
            mode="nearest",
        ).squeeze(0).squeeze(0)
    flat_texture = texture_image.flatten()
    flat_texture[texel_indices] = features
    if relative_init is not None:
        flat_relative_init = relative_init.flatten()
        flat_texture[texel_indices] += flat_relative_init[texel_indices]
    texture = flat_texture.reshape(texture_image.shape[0], texture_image.shape[1])
    if anti_aliasing:
        texture = texture.unsqueeze(0).unsqueeze(0)
        texture = pooling_operation(texture)
        texture = texture.squeeze(0).squeeze(0)
    return texture

def get_texels(n, device=torch.device("cpu")):
    # Create a grid of pixel coordinates
    x = torch.linspace(0, n-1, steps=n).to(device)
    y = torch.linspace(0, n-1, steps=n).to(device)
    px, py = torch.meshgrid(x, y)

    # Create pixel point tensor
    p = torch.stack((px.flatten(), py.flatten()), dim=1)

    return p

def compute_barycentric_coords(triangles, points):
    a, b, c = triangles[:, 0, :], triangles[:, 1, :], triangles[:, 2, :]
    v0, v1 = b - a, c - a

    v2 = points[:, None, :] - a[None, :, :]
    
    d00 = torch.sum(v0 * v0, dim=-1)
    d01 = torch.sum(v0 * v1, dim=-1)
    d11 = torch.sum(v1 * v1, dim=-1)
    d20 = torch.sum(v2 * v0, dim=-1)
    d21 = torch.sum(v2 * v1, dim=-1)
    
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / denom
    w = (d00 * d21 - d01 * d20) / denom
    u = 1 - (v + w)
    
    return torch.stack([u, v, w], dim=-1)

def save_model(mlp, path):
    torch.save(mlp.state_dict(), path)

def load_model(mlp, path, strict=True):
    mlp.load_state_dict(torch.load(path), strict=strict)
    return mlp
