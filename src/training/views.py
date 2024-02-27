import numpy as np
import torch
from torch.utils.data import DataLoader
from utils import get_view_direction
from models.render import Renderer


# Code adapted from https://github.com/eladrich/latent-nerf

def rand_poses(size, device, radius_range=(1.0, 1.5), elev_range=(0.0, 150.0), azim_range=(0.0, 360.0),
               angle_overhead=30.0, angle_front=60.0, init_elev=0.0, init_azim=0.0, rtn_dir=False,
               cam_mtx=True):
    elev_range = np.deg2rad(elev_range)
    azim_range = np.deg2rad(azim_range)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)
    init_elev = np.deg2rad(init_elev)
    init_azim = np.deg2rad(init_azim)

    radius = torch.rand(size, device=device) * (radius_range[1] - radius_range[0]) + radius_range[0]
    elevs = torch.rand(size, device=device) * (elev_range[1] - elev_range[0]) + elev_range[0]
    azims = torch.rand(size, device=device) * (azim_range[1] - azim_range[0]) + azim_range[0]

    if rtn_dir:
        dirs = get_view_direction(elevs, azims, angle_overhead, angle_front)
    else:
        dirs = None

    elevs += init_elev
    azims += init_azim

    if cam_mtx:
        camera_mtx = Renderer.get_camera_from_view(elevs, azims, radius)
    else:
        camera_mtx = None

    return dirs, elevs, azims, radius, camera_mtx


def circle_poses(device, radius=1.25, elev=60.0, azim=0.0, angle_overhead=30.0, angle_front=60.0):
    elev = np.deg2rad(elev)
    azim = np.deg2rad(azim)
    angle_overhead = np.deg2rad(angle_overhead)
    angle_front = np.deg2rad(angle_front)

    elevs = torch.FloatTensor([elev]).to(device)
    azims = torch.FloatTensor([azim]).to(device)
    dirs = get_view_direction(elevs, azims, angle_overhead, angle_front)

    return dirs, elevs.item(), azims.item(), radius


class ViewsDataset:
    def __init__(self, cfg: dict, device, type='train', size=100):
        super().__init__()

        self.cfg = cfg
        self.device = device
        self.type = type  # train, val, test
        self.size = size

        self.training = self.type in ['train', 'all']

    def collate(self, index):

        B = len(index)  # always 1
        if self.training:
            # random pose on the fly
            dirs, elevs, azims, radius, camera_mtx = rand_poses(
                B, self.device, radius_range=self.cfg.render.radius_range,
                angle_overhead=self.cfg.render.angle_overhead, elev_range=self.cfg.render.elev_range, azim_range=self.cfg.render.azim_range,
                angle_front=self.cfg.render.angle_front, init_azim=self.cfg.render.init_azim, init_elev=self.cfg.render.init_elev
            )

        else:
            # circle pose
            azim = (index[0] / self.size) * 360
            dirs, elevs, azims, radius = circle_poses(self.device, radius=self.cfg.render.radius_range[1] * 1.2, elev=60,
                                                      azim=azim,
                                                      angle_overhead=self.cfg.render.angle_overhead,
                                                      angle_front=self.cfg.render.angle_front)
            camera_mtx = None

        data = {
            'dir': dirs,
            'elev': elevs,
            'azim': azims,
            'radius': radius,
            'camera_mtx': camera_mtx,
        }

        return data

    def dataloader(self):
        loader = DataLoader(list(range(self.size)), batch_size=self.cfg.optim.batch_size, collate_fn=self.collate, shuffle=self.training,
                            num_workers=self.cfg.log.num_workers)
        loader._data = self  # an ugly fix... we need to access dataset in trainer.
        return loader