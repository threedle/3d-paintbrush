import kaolin as kal
import torch
import numpy as np

class Renderer:
    # adapted from https://github.com/eladrich/latent-nerf

    def __init__(
        self,
        device,
        dim=(224, 224),
        interpolation_mode='nearest',
        # Light Tensor (positive first): [ambient, right/left, front/back, top/bottom, ...]
        lights=torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ):
        assert interpolation_mode in ['nearest', 'bilinear', 'bicubic'], f'no interpolation mode {interpolation_mode}'

        camera = kal.render.camera.generate_perspective_projection(np.pi / 3).to(device)

        self.device = device
        self.interpolation_mode = interpolation_mode
        self.camera_projection = camera
        self.dim = dim
        self.background = torch.ones(dim).to(device).float()
        self.lights = lights.unsqueeze(0).to(device)

    @staticmethod
    def get_camera_from_view(elev, azim, r=3.0, look_at_height=0.0, device=torch.device('cuda')):
        """
        Convert tensor elevation/azimuth values into camera projections 

        Args:
            elev (torch.Tensor): elevation
            azim (torch.Tensor): azimuth
            r (float, optional): radius. Defaults to 3.0.

        Returns:
            Camera projection matrix (B x 4 x 3)
        """
        x = r * torch.cos(elev) * torch.cos(azim)
        y = r * torch.sin(elev)
        z = r * torch.cos(elev) * torch.sin(azim)
        B = elev.shape[0]

        if len(x.shape) == 0:
            pos = torch.tensor([x,y,z]).unsqueeze(0).to(device)
        else:
            pos = torch.stack([x, y, z], dim=1)
        look_at = torch.zeros_like(pos)
        look_at[:, 1] = look_at_height

        up = torch.tensor([0.0, 1.0, 0.0]).unsqueeze(0).repeat(B, 1).to(device)
        camera_proj = kal.render.camera.generate_transformation_matrix(pos, look_at, up).to(device)
        return camera_proj

    def render_texture(
        self, verts, faces, uv_face_attr, texture_map, elev=0, azim=0, radius=2,
        look_at_height=0.0, dims=None, white_background=False, lighting=False, lights=None
    ):
        dims = self.dim if dims is None else dims
        B = elev.shape[0]
        lights = self.lights.repeat(B, 1) if lights is None else lights

        camera_transform = self.get_camera_from_view(elev, azim, r=radius, look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            verts.to(self.device), faces.to(self.device), self.camera_projection, camera_transform=camera_transform)

        uv_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
            face_vertices_image, uv_face_attr.repeat(B, 1, 1, 1))
        uv_features = uv_features.detach()

        mask = (face_idx > -1).float()[..., None]
        image_features = kal.render.mesh.texture_mapping(uv_features, texture_map.repeat(B, 1, 1, 1), mode=self.interpolation_mode)
        image_features = image_features * mask

        if lighting:
            image_features = torch.clamp(image_features, 0.0, 1.0)
            # image_normals = face_normals[:, face_idx]#.squeeze(0) # TODO: might need to fix this when add lighting back
            # TODO: batch normals
            image_normals = []
            for i in range(len(face_normals)):
                image_normals.append(face_normals[i, face_idx[i], :])
            image_normals = torch.stack(image_normals)
            image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, lights)#.unsqueeze(0)
            image_features = image_features * image_lighting.unsqueeze(-1).repeat(1, 1, 1, 3)
            image_features = torch.clamp(image_features, 0.0, 1.0)

        if white_background:
            image_features += 1 * (1 - mask)

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)
    
    def render_mesh(self, mesh, colors, elev=0, azim=0, radius=2, look_at_height=0.0, dims=None, white_background=False, lighting=False, lights=None):
        dims = self.dim if dims is None else dims

        import kaolin.ops.mesh
        face_attributes = kaolin.ops.mesh.index_vertices_by_faces(
                                colors.unsqueeze(0).repeat(elev.shape[0], 1, 1),
                                mesh.faces.long()
                            )
        camera_transform = self.get_camera_from_view(torch.tensor(elev), torch.tensor(azim), r=radius,
                                                look_at_height=look_at_height).to(self.device)
        face_vertices_camera, face_vertices_image, face_normals = kal.render.mesh.prepare_vertices(
            mesh.vertices.to(self.device), mesh.faces.to(self.device).long(), self.camera_projection, camera_transform=camera_transform)


        image_features, face_idx = kal.render.mesh.rasterize(dims[1], dims[0], face_vertices_camera[:, :, :, -1],
                                                              face_vertices_image, face_attributes)

        if lighting:
            image_features = torch.clamp(image_features, 0.0, 1.0)
            # image_normals = face_normals[:, face_idx]#.squeeze(0) # TODO: might need to fix this when add lighting back
            # TODO: batch normals
            image_normals = []
            for i in range(len(face_normals)):
                image_normals.append(face_normals[i, face_idx[i], :])
            image_normals = torch.stack(image_normals)
            image_lighting = kal.render.mesh.spherical_harmonic_lighting(image_normals, lights)#.unsqueeze(0)
            image_features = image_features * image_lighting.unsqueeze(-1).repeat(1, 1, 1, 3)
            image_features = torch.clamp(image_features, 0.0, 1.0)

        mask = (face_idx > -1).float()[..., None]
        if white_background:
            image_features += 1 * (1 - mask)

        return image_features.permute(0, 3, 1, 2), mask.permute(0, 3, 1, 2)
