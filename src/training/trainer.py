import os
import torch
import torchvision
from pathlib import Path
from torchvision import transforms
from tqdm import tqdm
from models.mesh_model import MeshModel
from training.views import ViewsDataset
from typing import Dict, Union, List
from loguru import logger
from torch.utils.data import DataLoader
from utils import bake_surface_features, load_model, save_model, seed_everything
from configs.train_config import TrainConfig
from guidance.csd import CSD


class Trainer:
    def __init__(self, cfg: TrainConfig):
        seed_everything(cfg.optim.seed)

        # Set up config
        self.cfg = cfg
        self.exp_path = Path(self.cfg.log.exp_dir)
        self.exp_path.mkdir(parents=True, exist_ok=True)

        # Save config
        import yaml
        with open(cfg.log.exp_dir + '/config.yml', 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # Create output paths
        render_path = os.path.join(self.cfg.log.exp_dir, "renders")
        Path(render_path).mkdir(parents=True, exist_ok=True)
        texture_path = os.path.join(self.cfg.log.exp_dir, "textures")
        Path(texture_path).mkdir(parents=True, exist_ok=True)
        mesh_path = os.path.join(self.cfg.log.exp_dir, "meshes")
        Path(mesh_path).mkdir(parents=True, exist_ok=True)

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.epoch = 0
        self.init_logger()
        self.mesh_model = self.init_mesh()
        self.diffusion = self.init_diffusion()

        # Initialize text embeddings
        self.preprocess_text()
        self.text_z, self.text_z_neg = self.calc_text_embeddings(self.cfg.guidance.localization_prompt)
        self.style_text_z, self.style_text_z_neg = self.calc_text_embeddings(self.cfg.guidance.style_prompt)
        self.background_text_z, self.background_text_z_neg = self.calc_text_embeddings(self.cfg.guidance.background_prompt)
        del self.diffusion.pipe.text_encoder # delete text encoder to save memory
        torch.cuda.empty_cache()

        # Initialize optimizer and dataloader
        self.optim = self.init_optimizer()
        self.dataloaders = self.init_dataloaders()

        # Initialize logging
        self.losses = []
        self.style_losses = []
        self.prob_losses = []
        self.background_losses = []

        # Highlighter colors
        full_colors = [self.cfg.texture.primary_col, self.cfg.texture.secondary_col]
        self.colors = torch.tensor(full_colors).to(self.device)
        c1, c2 = self.colors[0], self.colors[1]
        self.c1 = c1.unsqueeze(0).repeat(1, self.mesh_model.texture_resolution, self.mesh_model.texture_resolution, 1).permute(0, 3, 1, 2)
        self.c2 = c2.unsqueeze(0).repeat(1, self.mesh_model.texture_resolution, self.mesh_model.texture_resolution, 1).permute(0, 3, 1, 2)

        # Initialize visualization
        self.cfg.render.viz_elev = torch.deg2rad(torch.Tensor(self.cfg.render.viz_elev).to(self.device) + self.cfg.render.init_elev)
        self.cfg.render.viz_azim = torch.deg2rad(torch.Tensor(self.cfg.render.viz_azim).to(self.device) + self.cfg.render.init_azim)
        self.cfg.render.viz_radius = torch.Tensor([self.cfg.render.viz_radius]).to(self.device).repeat(self.cfg.render.viz_elev.shape[0])
        self.cfg.render.viz_lights = torch.Tensor(self.cfg.render.viz_lights).to(self.device)

    def init_mesh(self):
        return MeshModel(
            self.cfg,
            render_grid_size=self.cfg.render.render_size,
            texture_resolution=self.cfg.texture.texture_resolution,
            device=self.device,
        ).to(self.device)

    def init_diffusion(self):
        diffusion = CSD()
        if self.cfg.guidance.cascaded:
            self.diffusion_II = CSD(stage=2)
            del self.diffusion_II.pipe.text_encoder
            torch.cuda.empty_cache()

        return diffusion

    def preprocess_text(self):
        if self.cfg.guidance.prefix != "":
            self.cfg.guidance.prefix += " "
        if self.cfg.guidance.localization_prompt is None:
            self.cfg.guidance.localization_prompt = f"a 3d render of a gray {self.cfg.guidance.object_name} with {self.cfg.guidance.prefix}yellow {self.cfg.guidance.edit}"
        if self.cfg.guidance.style_prompt is None:
            if self.cfg.guidance.style == '':
                self.cfg.guidance.style_prompt = f"a 3d render of a gray {self.cfg.guidance.object_name} with {self.cfg.guidance.prefix}{self.cfg.guidance.edit}"    
            else:
                self.cfg.guidance.style_prompt = f"a 3d render of a gray {self.cfg.guidance.object_name} with {self.cfg.guidance.prefix}{self.cfg.guidance.style} {self.cfg.guidance.edit}"
        if self.cfg.guidance.background_prompt is None:
            self.cfg.guidance.background_prompt = f"a 3d render of a {self.cfg.guidance.object_name} with {self.cfg.guidance.prefix}yellow {self.cfg.guidance.edit}"
        logger.info(f"Localization prompt: {self.cfg.guidance.localization_prompt}")
        logger.info(f"Style prompt: {self.cfg.guidance.style_prompt}")
        logger.info(f"Background prompt: {self.cfg.guidance.background_prompt}")

    def calc_text_embeddings(self, text) -> Union[torch.Tensor, List[torch.Tensor]]:
        text_z, text_z_neg = self.diffusion.encode_prompt(text)
        return text_z, text_z_neg

    def init_optimizer(self):
        optim = torch.optim.Adam(self.mesh_model.get_params(), self.cfg.optim.lr)
        return optim

    # TODO: clean this up
    def init_dataloaders(self) -> Dict[str, DataLoader]:
        train_dataloader = ViewsDataset(self.cfg, device=self.device, type='train', size=self.cfg.log.dataloader_size).dataloader()
        val_loader = ViewsDataset(self.cfg, device=self.device, type='val',
                                  size=self.cfg.render.eval_size).dataloader()
        # Will be used for creating the final video
        val_large_loader = ViewsDataset(self.cfg, device=self.device, type='val',
                                        size=self.cfg.render.full_eval_size).dataloader()
        dataloaders = {'train': train_dataloader, 'val': val_loader, 'val_large': val_large_loader}
        return dataloaders

    def init_logger(self):
        logger.remove()  # Remove default logger
        log_format = "<cyan>{time:YYYY-MM-DD HH:mm:ss}</cyan> <level>{message}</level>"
        logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, format=log_format)
        logger.add(Path(self.exp_path) / 'log.txt', colorize=False, format=log_format)

    def train(self):
        self.mesh_model.train()
        pbar = tqdm(total=self.cfg.optim.epochs, initial=self.epoch)
        logger.info('Starting optimization')

        while self.epoch < self.cfg.optim.epochs:
            # Keep going over dataloader until finished the required number of iterations
            for data in self.dataloaders['train']:
                pbar.update(1)

                # Update stage II weight
                # (only if cascaded and stage_II_weight_range is provided, otherwise use a constant stage II weight)
                if self.cfg.guidance.cascaded and (self.cfg.guidance.stage_II_weight_range is not None) and (not self.cfg.guidance.no_lerp_stage_II):
                    alpha = self.epoch / self.cfg.optim.epochs
                    self.cfg.guidance.stage_II_weight = self.cfg.guidance.stage_II_weight_range[0] * (1 - alpha) + self.cfg.guidance.stage_II_weight_range[1] * alpha

                # Reset gradients
                self.optim.zero_grad()

                # Annealed timestep sampling
                if self.cfg.guidance.anneal_t and (self.epoch > (self.cfg.optim.epochs / 2)):
                    self.diffusion.update_step(min_step_percent=0.02, max_step_percent=0.5)
                    self.diffusion_II.update_step(min_step_percent=0.02, max_step_percent=0.5)

                # Sample points for MLP to reduce memory usage
                if self.cfg.texture.sample_points:
                    sampled_texel_indices = torch.randint(0, self.mesh_model.texel_indices.shape[0], (self.cfg.texture.mlp_batch_size,))
                    network_input = self.mesh_model.surface_points[sampled_texel_indices]
                    texel_indices = self.mesh_model.texel_indices[sampled_texel_indices]
                else:
                    network_input = self.mesh_model.surface_points
                    texel_indices = self.mesh_model.texel_indices

                # Get MLP predictions
                preds = self.mesh_model(network_input)

                # Localization branch
                pred_probs = preds['pred_prob']
                self.mesh_model.texture_prob = bake_surface_features(
                    pred_probs,
                    texel_indices,
                    self.mesh_model.texture_prob.clone().detach(),
                    anti_aliasing=self.cfg.texture.bake_anti_aliasing,
                )
                probs = self.mesh_model.texture_prob.transpose(0, 1).flip(0).repeat(1, 3, 1, 1)

                # Texture branch
                pred_rgbs = preds['pred_rgb']
                all_channels = []
                for rgb_idx in range(3):
                    single_channel = bake_surface_features(
                        pred_rgbs[:, rgb_idx],
                        texel_indices,
                        self.mesh_model.texture_img[0, rgb_idx, :, :].clone().detach(),
                        anti_aliasing=self.cfg.texture.bake_anti_aliasing,
                    )
                    all_channels.append(single_channel)
                self.mesh_model.texture_img = torch.stack(all_channels, dim=0).unsqueeze(0)
                texture_img = self.mesh_model.texture_img.transpose(2, 3).flip(2)

                # Background texture branch
                if self.cfg.network.background_mlp:
                    pred_rgbs_bg = preds['pred_rgb_bg']
                    all_channels = []
                    for rgb_idx in range(3):
                        single_channel = bake_surface_features(
                            pred_rgbs_bg[:, rgb_idx],
                            texel_indices,
                            self.mesh_model.background_img[0, rgb_idx, :, :].clone().detach(),
                            anti_aliasing=self.cfg.texture.bake_anti_aliasing,
                        )
                        all_channels.append(single_channel)
                    self.mesh_model.background_img = torch.stack(all_channels, dim=0).unsqueeze(0)
                    background_img = self.mesh_model.background_img.transpose(2, 3).flip(2)
                else:
                    background_img = texture_img.clone()

                # Perform masks with the predicted localization
                self.mesh_model.binary_texture_img = probs * self.c1 + (1 - probs) * self.c2
                self.mesh_model.masked_texture_img = probs * texture_img + (1 - probs) * self.c2
                self.mesh_model.masked_background_texture_img = probs * self.c1 + (1 - probs) * background_img

                # View info
                elev = data['elev']
                azim = data['azim']
                radius = data['radius']
                B = elev.shape[0]

                # Render mesh
                outputs = self.mesh_model.render(elev=elev, azim=azim, radius=radius)
                pred_prob_rgb = outputs[0]['image']
                pred_style_rgb = outputs[1]['image']
                pred_background_rgb = outputs[2]['image']

                # Adjust text embeddings for batch size
                text_z = self.text_z.repeat(B, 1, 1) # [B, 77, 4096]
                text_z_neg = self.text_z_neg.repeat(B, 1, 1) # [B, 77, 4096]
                style_text_z = self.style_text_z.repeat(B, 1, 1) # [B, 77, 4096]
                style_text_z_neg = self.style_text_z_neg.repeat(B, 1, 1) # [B, 77, 4096]
                background_text_z = self.background_text_z.repeat(B, 1, 1) # [B, 77, 4096]
                background_text_z_neg = self.background_text_z_neg.repeat(B, 1, 1) # [B, 77, 4096]

                # Compute the score distillation loss
                if self.cfg.guidance.batched_sd:
                    rgb_inputs = torch.cat([pred_prob_rgb, pred_style_rgb, pred_background_rgb])
                    text_z_inputs = torch.cat([text_z, style_text_z, background_text_z])
                    text_z_neg_inputs = torch.cat([text_z_neg, style_text_z_neg, background_text_z_neg])
                    sds = self.diffusion(rgb_inputs, text_z_inputs, text_z_neg_inputs)
                    loss = sds['loss']
                    if self.cfg.guidance.cascaded:
                        stage_I_loss = loss.clone()
                        sds_II = self.diffusion_II(rgb_inputs, text_z_inputs, text_z_neg_inputs)
                        stage_II_loss = sds_II['loss']
                        loss = stage_I_loss * self.cfg.guidance.stage_I_weight + stage_II_loss * self.cfg.guidance.stage_II_weight
                    # Rescale loss to match unbatched score distillation scale
                    loss = loss * 3
                else:
                    prob_sds = self.diffusion(pred_prob_rgb, text_z, text_z_neg)
                    style_sds = self.diffusion(pred_style_rgb, style_text_z, style_text_z_neg)
                    loss = prob_sds['loss'] + style_sds['loss']
                    if self.cfg.guidance.third_loss:
                        background_sds = self.diffusion(pred_background_rgb, background_text_z, background_text_z_neg)
                        loss += background_sds['loss']

                    if self.cfg.guidance.cascaded:
                        stage_I_loss = loss.clone()
                        prob_sds_II = self.diffusion_II(pred_prob_rgb, text_z, text_z_neg)
                        style_sds_II = self.diffusion_II(pred_style_rgb, style_text_z, style_text_z_neg)
                        stage_II_loss = prob_sds_II['loss'] + style_sds_II['loss']
                        if self.cfg.guidance.third_loss:
                            background_sds_II = self.diffusion_II(pred_background_rgb, background_text_z, background_text_z_neg)
                            stage_II_loss += background_sds_II['loss']
                        loss = stage_I_loss * self.cfg.guidance.stage_I_weight + stage_II_loss * self.cfg.guidance.stage_II_weight

                # Backpropagate and update weights
                loss.backward()
                self.optim.step()

                # Log results
                if self.epoch % self.cfg.log.log_interval == 0:
                    logger.info(f"Epoch: {self.epoch}")
                    logger.info(f"Loss: {loss.item()}")
                    if self.cfg.guidance.cascaded:
                        logger.info(f"Stage I loss: {stage_I_loss.item()}, Stage II loss: {stage_II_loss.item()}")
                        logger.info(f"Stage I weight: {self.cfg.guidance.stage_I_weight}")
                        logger.info(f"Stage II weight: {self.cfg.guidance.stage_II_weight}")
                if self.epoch % self.cfg.log.log_interval_viz == 0:
                    # Visualize mesh and textures
                    self.visualize_mesh(self.cfg, self.mesh_model, output_path=self.exp_path, name=f"iter_{self.epoch}")
                    self.save_renders(self.exp_path, self.mesh_model.masked_texture_img, name=f"textures/local_edit_texture_{self.epoch}.png")
                    self.save_renders(self.exp_path, self.mesh_model.binary_texture_img, name=f"textures/localization_texture_{self.epoch}.png")

                    # Save model
                    save_model(self.mesh_model, os.path.join(self.exp_path, "model.pth"))
                if self.cfg.log.log_all_intermediate_results and self.epoch % self.cfg.log.log_interval_real_renders == 0:
                    self.save_renders(self.exp_path, pred_prob_rgb, name=f"renders/localization_{self.epoch}.png")
                    self.save_renders(self.exp_path, pred_style_rgb, name=f"renders/local_edit_{self.epoch}.png")
                    self.save_renders(self.exp_path, pred_background_rgb, name=f"renders/background_{self.epoch}.png")

                    self.save_renders(self.exp_path, self.mesh_model.texture_prob, name=f"textures/localization_probabilities_{self.epoch}.png")
                    self.save_renders(self.exp_path, self.mesh_model.texture_img, name=f"textures/local_edit_unmasked_texture_{self.epoch}.png")
                    self.save_renders(self.exp_path, self.mesh_model.background_img, name=f"textures/background_unmasked_texture_{self.epoch}.png")
                    self.save_renders(self.exp_path, self.mesh_model.masked_background_texture_img, name=f"textures/background_masked_texture{self.epoch}.png")
                self.epoch += 1

        logger.info('Finished training')

        # Export mesh
        # Visualize mesh and textures
        self.visualize_mesh(self.cfg, self.mesh_model, output_path=self.exp_path, name=f"final_renders")
        self.save_renders(self.exp_path, self.mesh_model.masked_texture_img, name=f"textures/final_local_edit_texture.png")
        self.save_renders(self.exp_path, self.mesh_model.binary_texture_img, name=f"textures/final_localization_texture.png")
        self.mesh_model.export_mesh(self.exp_path, texture=self.mesh_model.binary_texture_img, file_name="meshes/localization")
        self.mesh_model.export_mesh(self.exp_path, texture=self.mesh_model.masked_texture_img, file_name="meshes/local_edit")
        if self.cfg.log.log_all_intermediate_results:
            self.mesh_model.export_mesh(self.exp_path, texture=self.mesh_model.masked_background_texture_img, file_name="meshes/background_edit")

        logger.info('\tDone!')

    @staticmethod
    def save_renders(dir, rendered_images, name):
        torchvision.utils.save_image(rendered_images, os.path.join(dir, name))

    @staticmethod
    def visualize_mesh(cfg, mesh_model, output_path, name, texture_lighting=False):
        render = mesh_model.render(cfg.render.viz_elev, cfg.render.viz_azim, cfg.render.viz_radius, lights=cfg.render.viz_lights, texture_lighting=texture_lighting)
        localization_render = render[0]['image']
        style_render = render[1]['image']
        background_render = render[2]['image']
        renders = [localization_render]
        if cfg.render.viz_style:
            renders.append(style_render)
        renders = torch.cat(renders, dim=0)
        Trainer.save_renders(output_path, renders, name=f"renders/{name}.png")

    def load_model(self, model_path, strict=True):
        logger.info(f"Loading model from {model_path}")
        self.mesh_model = load_model(self.mesh_model, model_path, strict=strict)

    @staticmethod
    def inference(cfg, output_path=None, model_path=None, threshold=0.5):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create output paths
        if output_path is None:
            output_path = cfg.log.exp_dir
        render_path = os.path.join(output_path, "renders")
        Path(render_path).mkdir(parents=True, exist_ok=True)
        mesh_path = os.path.join(output_path, "meshes")
        Path(mesh_path).mkdir(parents=True, exist_ok=True)

        # Load pre-trained model
        mesh_model = MeshModel(
            cfg,
            render_grid_size=cfg.render.render_size,
            texture_resolution=cfg.texture.texture_resolution,
            device=device,
        ).to(device)
        if model_path is None:
            model_path = output_path + "/model.pth"
        load_model(mesh_model, model_path, strict=False)
        mesh_model.eval()

        # Set colors
        c1 = torch.Tensor([0.8, 1.0, 0.0]).to(device)
        c2 = torch.Tensor([0.71, 0.71, 0.71]).to(device)
        c1 = c1.unsqueeze(0).repeat(1, mesh_model.texture_resolution, mesh_model.texture_resolution, 1).permute(0, 3, 1, 2)
        c2 = c2.unsqueeze(0).repeat(1, mesh_model.texture_resolution, mesh_model.texture_resolution, 1).permute(0, 3, 1, 2)

        # Initialize visualization
        if cfg.render.inference_azim is None:
            cfg.render.viz_elev = torch.deg2rad(torch.Tensor(cfg.render.viz_elev).to(device) + cfg.render.init_elev)
            cfg.render.viz_azim = torch.deg2rad(torch.Tensor(cfg.render.viz_azim).to(device) + cfg.render.init_azim)
            cfg.render.viz_radius = torch.Tensor([cfg.render.viz_radius]).to(device).repeat(cfg.render.viz_elev.shape[0])
            cfg.render.viz_lights = torch.Tensor(cfg.render.inference_viz_lights).to(device)
        else:
            cfg.render.viz_elev = torch.deg2rad(torch.Tensor([cfg.render.inference_elev]).to(device) + cfg.render.init_elev)
            cfg.render.viz_azim = torch.deg2rad(torch.Tensor([cfg.render.inference_azim]).to(device) + cfg.render.init_azim)
            cfg.render.viz_radius = torch.Tensor([cfg.render.viz_radius]).to(device)
            cfg.render.viz_lights = torch.Tensor((1.5, 0., 1., 1., 0., 0., 0., 0., 0.)).to(device)

        # Predict textures with pre-trained model
        with torch.no_grad():
            preds = mesh_model(mesh_model.surface_points)
            texel_indices = mesh_model.texel_indices

            # Localization branch
            pred_probs = preds['pred_prob']
            mesh_model.texture_prob = bake_surface_features(
                pred_probs,
                texel_indices,
                mesh_model.texture_prob.clone().detach(),
                anti_aliasing=cfg.texture.bake_anti_aliasing,
            )
            probs = mesh_model.texture_prob.transpose(0, 1).flip(0).repeat(1, 3, 1, 1)

            # Texture branch
            pred_rgbs = preds['pred_rgb']
            all_channels = []
            for rgb_idx in range(3):
                single_channel = bake_surface_features(
                    pred_rgbs[:, rgb_idx],
                    texel_indices,
                    mesh_model.texture_img[0, rgb_idx, :, :].clone().detach(),
                    anti_aliasing=cfg.texture.bake_anti_aliasing,
                )
                all_channels.append(single_channel)
            mesh_model.texture_img = torch.stack(all_channels, dim=0).unsqueeze(0)
            texture_img = mesh_model.texture_img.transpose(2, 3).flip(2)

            # Background texture branch
            pred_rgbs_bg = preds['pred_rgb_bg']
            all_channels = []
            for rgb_idx in range(3):
                single_channel = bake_surface_features(
                    pred_rgbs_bg[:, rgb_idx],
                    texel_indices,
                    mesh_model.background_img[0, rgb_idx, :, :].clone().detach(),
                    anti_aliasing=cfg.texture.bake_anti_aliasing,
                )
                all_channels.append(single_channel)
            mesh_model.background_img = torch.stack(all_channels, dim=0).unsqueeze(0)
            background_img = mesh_model.background_img.transpose(2, 3).flip(2)

            # Perform masks with the predicted localization
            thresholded_probs = torch.where(probs > threshold, probs, torch.zeros_like(probs))
            mesh_model.binary_texture_img = thresholded_probs * c1 + (1 - thresholded_probs) * c2
            mesh_model.masked_texture_img = thresholded_probs * texture_img + (1 - thresholded_probs) * c2
            mesh_model.masked_background_texture_img = thresholded_probs * c1 + (1 - thresholded_probs) * background_img

            # Save renders
            Trainer.visualize_mesh(cfg, mesh_model, output_path=output_path, name="inference", texture_lighting=cfg.render.inference_texture_lighting)

            # Export mesh
            mesh_model.export_mesh(output_path, texture=mesh_model.binary_texture_img, file_name="meshes/localization")
            mesh_model.export_mesh(output_path, texture=mesh_model.masked_texture_img, file_name="meshes/local_edit")
            if cfg.log.log_all_intermediate_results:
                mesh_model.export_mesh(output_path, texture=mesh_model.masked_background_texture_img, file_name="meshes/background_edit")

        logger.info('Finished inference!')
