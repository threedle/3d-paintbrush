import torch
import torch.nn.functional as F
from configs.guidance_config import GuidanceConfig
from diffusers import IFPipeline, UNet2DConditionModel
from typing import Optional
from contextlib import contextmanager
from loguru import logger

class CSD:
    def __init__(self, cfg: GuidanceConfig = None, stage: int = 1) -> None:
        # Set device (cuda or cpu)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # If no config passed, use default config
        if cfg is None:
            cfg = GuidanceConfig()
        self.cfg = cfg

        # Determine stage of diffusion model
        self.stage = stage
        if stage == 2:
            self.cfg.model_name = self.cfg.stage_II_model_name

        # Load model
        self.weights_dtype = (
            torch.float16 if self.cfg.half_precision_weights else torch.float32
        )
        self.load_model()

    def load_model(self):
        self.pipe = IFPipeline.from_pretrained(
            self.cfg.model_name,
            safety_checker=None,
            watermarker=None,
            feature_extractor=None,
            requires_safety_checker=False,
            variant="fp16" if self.cfg.half_precision_weights else None,
            torch_dtype=self.weights_dtype,
        ).to(self.device)
        self.unet = self.pipe.unet.eval()

        for p in self.unet.parameters():
            p.requires_grad_(False)

        self.scheduler = self.pipe.scheduler

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.set_min_max_steps()  # set to default value (0.02-0.98)

        self.alphas: torch.FloatTensor = self.scheduler.alphas_cumprod.to(
            self.device
        )

        self.grad_clip_val: Optional[float] = self.cfg.grad_clip_val

        if self.cfg.cpu_offload:
            self.pipe.enable_sequential_cpu_offload()

        logger.info("Loaded DeepFloyd IF")

    @torch.cuda.amp.autocast(enabled=False)
    def set_min_max_steps(self, min_step_percent=0.02, max_step_percent=0.98):
        self.min_step = int(self.num_train_timesteps * min_step_percent)
        self.max_step = int(self.num_train_timesteps * max_step_percent)

    def encode_prompt(self, prompt: str, negative_prompt: str = None):
        prompt_embeds, negative_embeds = self.pipe.encode_prompt(prompt, negative_prompt=negative_prompt)
        return prompt_embeds, negative_embeds

    def classifier_free_guidance(self, noise_pred: torch.FloatTensor, guidance_scale: float, output_channels=3):
        noise_pred_text, noise_pred_uncond = noise_pred.chunk(2)
        noise_pred_uncond, _ = noise_pred_uncond.split(output_channels, dim=1)
        noise_pred_text, _ = noise_pred_text.split(output_channels, dim=1)
        noise_pred = noise_pred_text + guidance_scale * (
            noise_pred_text - noise_pred_uncond
        )
        return noise_pred

    @torch.cuda.amp.autocast(enabled=False)
    def forward_unet(
        self,
        unet: UNet2DConditionModel,
        latents: torch.FloatTensor,
        t: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        class_labels: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        input_dtype = latents.dtype
        return unet(
            latents.to(self.weights_dtype),
            t.to(self.weights_dtype),
            encoder_hidden_states=encoder_hidden_states.to(self.weights_dtype),
            class_labels=class_labels.to(self.weights_dtype) if class_labels is not None else None,
        ).sample.to(input_dtype)

    @contextmanager
    def disable_unet_class_embedding(self, unet: UNet2DConditionModel):
        class_embedding = unet.class_embedding
        try:
            unet.class_embedding = None
            yield unet
        finally:
            unet.class_embedding = class_embedding

    def compute_grad_base(
        self,
        z_0: torch.FloatTensor, # B, 3, H, W
        text_embeddings: torch.FloatTensor, # 2B, 77, 4096
    ):
        B = z_0.shape[0]

        with torch.no_grad():
            # timestep ~ U(min_step, max_step)
            t = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B],
                dtype=torch.long,
                device=self.device,
            )
            
            # add noise
            noise = torch.randn_like(z_0)
            z_t = self.scheduler.add_noise(z_0, noise, t)
            
            # predict noise
            latent_model_input = torch.cat([z_t] * 2, dim=0)
            with self.disable_unet_class_embedding(self.unet) as unet:
                noise_pred = self.forward_unet(
                    unet,
                    latent_model_input,
                    torch.cat([t] * 2),
                    encoder_hidden_states=text_embeddings,
                )

        noise_pred = self.classifier_free_guidance(noise_pred, self.cfg.guidance_scale)

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred - noise)
        return grad

    def compute_grad_cascaded(
        self,
        z_0: torch.FloatTensor, # B, 3, H, W
        x_0: torch.FloatTensor, # B, 3, H, W
        text_embeddings: torch.FloatTensor, # 2B, 77, 4096
    ):
        with torch.no_grad():
            B = z_0.shape[0]

            # timestep ~ U(min_step, max_step)
            timesteps = torch.randint(
                self.min_step,
                self.max_step + 1,
                [B*2],
                dtype=torch.long,
                device=self.device,
            )
            t, s = timesteps.chunk(2, dim=0)
            
            # add noise
            noise = torch.randn_like(x_0.repeat(2, 1, 1, 1))
            noise_s, noise_t = noise.chunk(2, dim=0)
            upscaled_z_0 = F.interpolate(z_0, (256, 256), mode="bilinear", align_corners=False)
            upscaled_z_s = self.scheduler.add_noise(upscaled_z_0, noise_s, s)
            x_t = self.scheduler.add_noise(x_0, noise_t, t)
            latents_noisy = torch.cat([x_t, upscaled_z_s], dim=1)
            
            # predict noise
            model_input = torch.cat([latents_noisy] * 2, dim=0)
            noise_pred = self.forward_unet(
                self.unet,
                model_input,
                torch.cat([t] * 2),
                encoder_hidden_states=text_embeddings,
                class_labels=torch.cat([s] * 2),
            )

        noise_pred = self.classifier_free_guidance(noise_pred, self.cfg.guidance_scale)

        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        grad = w * (noise_pred - noise_t)
        return grad

    def __call__(
        self,
        rgb_img: torch.FloatTensor, # B 3 H W
        text_embeddings: torch.FloatTensor, # B 77 4096
        negative_text_embeddings: torch.FloatTensor, # B 77 4096
        **kwargs,
    ):
        batch_size = rgb_img.shape[0]
        rgb_img = rgb_img * 2.0 - 1.0  # scale to [-1, 1] to match the diffusion range
        z_0 = F.interpolate(rgb_img, (64, 64), mode="bilinear", align_corners=False)

        # Set prompt embeddings
        self.text_embeddings = torch.cat([text_embeddings, negative_text_embeddings])

        # Cascaded Score Distillation
        if self.stage == 2:
            x_0 = F.interpolate(rgb_img, (256, 256), mode="bilinear", align_corners=False)
            grad = self.compute_grad_cascaded(z_0, x_0, self.text_embeddings)
            grad = torch.nan_to_num(grad)

            # clip grad for stable training (per threestudio)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            target = (x_0 - grad).detach()
            loss = 0.5 * F.mse_loss(x_0, target, reduction="sum") / batch_size
        else:
            grad = self.compute_grad_base(z_0, self.text_embeddings)
            grad = torch.nan_to_num(grad)

            # clip grad for stable training (per threestudio)
            if self.grad_clip_val is not None:
                grad = grad.clamp(-self.grad_clip_val, self.grad_clip_val)

            target = (z_0 - grad).detach()
            loss = 0.5 * F.mse_loss(z_0, target, reduction="sum") / batch_size

        return {
            "loss": loss,
            "grad_norm": grad.norm(),
            "min_step": self.min_step,
            "max_step": self.max_step,
        }

    
    def update_step(self, min_step_percent: float, max_step_percent: float):
        self.set_min_max_steps(
            min_step_percent=min_step_percent,
            max_step_percent=max_step_percent,
        )
