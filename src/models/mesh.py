import kaolin as kal
import torch
import copy

class Mesh:
    def __init__(self,obj_path, device):
        # from https://github.com/threedle/text2mesh and https://github.com/eladrich/latent-nerf

        if ".obj" in obj_path:
            try:
                mesh = kal.io.obj.import_mesh(obj_path, with_normals=True, with_materials=True)
            except:
                mesh = kal.io.obj.import_mesh(obj_path, with_normals=True, with_materials=False)
        elif ".off" in obj_path:
            mesh = kal.io.off.import_mesh(obj_path)
        else:
            raise ValueError(f"{obj_path} extension not implemented in mesh reader.")

        self.vertices = mesh.vertices.to(device)
        self.faces = mesh.faces.to(device)
        self.ft = mesh.face_uvs_idx
        self.vt = mesh.uvs

    def standardize_mesh(self,inplace=False):
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices
        center = verts.mean(dim=0)
        verts -= center
        scale = torch.std(torch.norm(verts, p=2, dim=1))
        verts /= scale
        mesh.vertices = verts
        return mesh

    def normalize_mesh(self,inplace=False, target_scale=1, dy=0):
        mesh = self if inplace else copy.deepcopy(self)

        verts = mesh.vertices
        center = verts.mean(dim=0)
        verts -= center
        scale = torch.max(torch.norm(verts, p=2, dim=1))
        verts /= scale
        verts *= target_scale
        verts[:, 1] += dy
        mesh.vertices = verts
        return mesh
