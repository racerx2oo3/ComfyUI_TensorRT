from nodes import EmptyLatentImage, CLIPTextEncode, common_ksampler
from comfy_extras.nodes_sd3 import EmptySD3LatentImage


class DiffusionPipe:
    def __init__(
        self,
        model,
        clip,
        batch_size,
        width,
        height,
        seed,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        device,
        is_sd3=False,
    ) -> None:
        self.model = model
        self.clip = clip
        self.clip_node = CLIPTextEncode()
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.device = device

        self.seed = seed
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise
        self.latent_node = EmptySD3LatentImage() if is_sd3 else EmptyLatentImage()

    def __call__(self, num_inference_steps, positive_prompt, negative_prompt):
        (positive,) = self.clip_node.encode(self.clip, positive_prompt)
        (negative,) = self.clip_node.encode(self.clip, negative_prompt)
        (latent,) = self.latent_node.generate(self.width, self.height, self.batch_size)
        (out,) = common_ksampler(
            self.model,
            self.seed,
            num_inference_steps,
            self.cfg,
            self.sampler_name,
            self.scheduler,
            positive,
            negative,
            latent,
            self.denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
        )

        return out["samples"]
