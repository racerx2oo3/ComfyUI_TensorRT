# Put this in the custom_nodes folder, put your tensorrt engine files in ComfyUI/models/tensorrt/ (you will have to create the directory)

import torch
import os
import re
import logging

import comfy.model_base
import comfy.model_management
import comfy.model_patcher
import comfy.supported_models
import folder_paths
from comfy.model_detection import model_config_from_unet

import modelopt.torch.opt as mto
import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.nn as mnn
from modelopt.torch.quantization.nn import QuantModuleRegistry

from calib.plugin_calib import PercentileCalibrator

MAX_RESOLUTION = 16384

# add output directory to modelopt search path<<<<<<
if "modelopt" in folder_paths.folder_names_and_paths:
    folder_paths.folder_names_and_paths["modelopt"][0].append(
        os.path.join(folder_paths.models_dir, "modelopt")
    )
    folder_paths.folder_names_and_paths["modelopt"][1].add(".pt")
else:
    folder_paths.folder_names_and_paths["modelopt"] = (
        [os.path.join(folder_paths.models_dir, "modelopt")],
        {".pt"},
    )


def filter_func(name):
    pattern = re.compile(
        r".*(emb_layers|time_embed|input_blocks.0.0|out.2|skip_connection|label_emb.0).*"
    )
    return pattern.match(name) is not None


def quantize_lvl(unet, quant_level=2.5):
    """
    We should disable the unwanted quantizer when exporting the onnx
    Because in the current modelopt setting, it will load the quantizer amax for all the layers even
    if we didn't add that unwanted layer into the config during the calibration
    """
    for name, module in unet.named_modules():
        if isinstance(module, (torch.nn.Conv2d)):
            module.input_quantizer.enable()
            module.weight_quantizer.enable()
        elif isinstance(module, (torch.nn.Linear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (
                    quant_level >= 2.5
                    and ("to_q" in name or "to_k" in name or "to_v" in name)
                )
                or quant_level == 3
            ):
                module.input_quantizer.enable()
                module.weight_quantizer.enable()
            else:
                module.input_quantizer.disable()
                module.weight_quantizer.disable()


def get_int8_config(
    model,
    quant_level=3,
    alpha=0.8,
    percentile=1.0,
    num_inference_steps=20,
    collect_method="min-mean",
):
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": {"method": "smoothquant", "alpha": alpha},
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if (
            w_name in quant_config["quant_cfg"].keys()
            or i_name in quant_config["quant_cfg"].keys()
        ):
            continue
        if filter_func(name):
            continue
        if isinstance(module, (torch.nn.Linear)):
            if (
                (quant_level >= 2 and "ff.net" in name)
                or (
                    quant_level >= 2.5
                    and ("to_q" in name or "to_k" in name or "to_v" in name)
                )
                or quant_level == 3
            ):
                quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
                quant_config["quant_cfg"][i_name] = {"num_bits": 8, "axis": -1}
        elif isinstance(module, (torch.nn.Conv2d)):
            quant_config["quant_cfg"][w_name] = {"num_bits": 8, "axis": 0}
            quant_config["quant_cfg"][i_name] = {
                "num_bits": 8,
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": 8,
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "collect_method": collect_method,
                    },
                ),
            }
    return quant_config


def get_fp8_config(
    model, percentile=1.0, num_inference_steps=20, collect_method="min-mean"
):
    quant_config = {
        "quant_cfg": {
            "*lm_head*": {"enable": False},
            "*output_layer*": {"enable": False},
            "default": {"num_bits": 8, "axis": None},
        },
        "algorithm": "max",
    }
    for name, module in model.named_modules():
        w_name = f"{name}*weight_quantizer"
        i_name = f"{name}*input_quantizer"

        if w_name in quant_config["quant_cfg"].keys() or i_name in quant_config["quant_cfg"].keys():  # type: ignore
            continue
        if filter_func(name):
            continue
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            quant_config["quant_cfg"][w_name] = {"num_bits": (4, 3), "axis": None}  # type: ignore
            quant_config["quant_cfg"][i_name] = {  # type: ignore
                "num_bits": (4, 3),
                "axis": None,
                "calibrator": (
                    PercentileCalibrator,
                    (),
                    {
                        "num_bits": (4, 3),
                        "axis": None,
                        "percentile": percentile,
                        "total_step": num_inference_steps,
                        "collect_method": collect_method,
                    },
                ),
            }
    return quant_config


def generate_fp8_scales(unet):
    # temporary solution due to a known bug in torch.onnx._dynamo_export
    for _, module in unet.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            module.input_quantizer._num_bits = 8
            module.weight_quantizer._num_bits = 8
            module.input_quantizer._amax = (module.input_quantizer._amax * 127) / 448.0
            module.weight_quantizer._amax = (
                module.weight_quantizer._amax * 127
            ) / 448.0


def load_calib_prompts(batch_size, calib_data_path="./calib_prompts.txt"):
    with open(calib_data_path, "r", encoding="utf8") as file:
        lst = [line.rstrip("\n") for line in file]
    return [lst[i : i + batch_size] for i in range(0, len(lst), batch_size)]


def do_calibrate(pipe, calibration_prompts, **kwargs):
    for i_th, prompts in enumerate(calibration_prompts):
        if i_th >= kwargs["calib_size"]:
            return
        pipe(
            prompt=prompts,
            num_inference_steps=kwargs["n_steps"],
            negative_prompt=[
                "normal quality, low quality, worst quality, low res, blurry, nsfw, nude"
            ]
            * len(prompts),
        )


def register_quant_modules():
    for op in comfy.ops.disable_weight_init.__dict__.keys():
        if not hasattr(mnn, f"Quant{op}"):
            continue
        QuantModuleRegistry.register(
            {getattr(comfy.ops.disable_weight_init, op): f"comfy.{op}"}
        )(QuantModuleRegistry._registry[getattr(torch.nn, op)])


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
    ) -> None:
        self.model = model
        self.clip = clip
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.device = device

        self.seed = seed
        self.cfg = cfg
        self.sampler_name = sampler_name
        self.scheduler = scheduler
        self.denoise = denoise

    def encode(self, text):
        out = []
        for prompt in text:
            tokens = self.clip.tokenize(prompt)
            cond, pooled = self.clip.encode_from_tokens(tokens, return_pooled=True)
            out.append([cond, {"pooled_output": pooled}])
        return out

    def __call__(self, num_inference_steps, prompt, negative_prompt):
        latent_image = torch.zeros(
            [self.batch_size, 4, self.height // 8, self.width // 8], device=self.device
        )
        positive = self.encode(prompt)
        negative = self.encode(negative_prompt)

        noise = torch.zeros(
            latent_image.size(),
            dtype=latent_image.dtype,
            layout=latent_image.layout,
            device="cpu",
        )

        samples = comfy.sample.sample(
            self.model,
            noise,
            num_inference_steps,
            self.cfg,
            self.sampler_name,
            self.scheduler,
            positive,
            negative,
            latent_image,
            denoise=self.denoise,
            disable_noise=False,
            start_step=None,
            last_step=None,
            force_full_denoise=False,
            noise_mask=None,
            callback=None,
            disable_pbar=False,
            seed=self.seed,
        )

        return samples


class ModelOptQuantizer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "name": ("STRING", {"default": "unet"}),
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "format": (["int8"],),
                "calib_prompts": ("STRING", {"default": "calib/calib_prompts.txt"}),
                "width": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 512, "min": 16, "max": MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 8.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                    },
                ),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "denoise": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "percentile": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "alpha": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "calib_size": ("INT", {"default": 128, "min": 1, "max": 10000}),
                "collect_method": (
                    ["min-mean", "min-max" "mean-max", "global_min", "default"],
                ),
                "quant_level": ([3.0, 2.5, 2.0, 1.0],),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "quantize"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    def quantize(
        self,
        name,
        model,
        clip,
        format,
        calib_prompts,
        width,
        height,
        batch_size,
        seed,
        steps,
        cfg,
        sampler_name,
        scheduler,
        denoise,
        percentile,
        alpha,
        calib_size,
        collect_method,
        quant_level,
    ):
        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([model], force_patch_weights=True)
        unet = model.model.diffusion_model
        device = comfy.model_management.get_torch_device()

        # This is a list of prompts
        cali_prompts = load_calib_prompts(
            batch_size,
            os.path.join(os.path.dirname(os.path.realpath(__file__)), calib_prompts),
        )
        calib_size = calib_size // batch_size
        quant_level = float(quant_level)

        is_sd1 = True  # TODO
        extra_step = (
            1 if is_sd1 else 0
        )  # Depending on the scheduler. some schedulers will do n+1 steps

        if format == "int8":
            # Making sure to use global_min in the calibrator for SD 1.5
            # assert collect_method != "default"
            if is_sd1:
                collect_method = "global_min"
            quant_config = get_int8_config(
                unet,
                quant_level,
                alpha,
                percentile,
                steps + extra_step,
                collect_method=collect_method,
            )
        elif format == "fp8":
            if collect_method == "default":
                quant_config = mtq.FP8_DEFAULT_CFG
            else:
                quant_config = get_fp8_config(
                    unet,
                    percentile,
                    steps + extra_step,
                    collect_method=collect_method,
                )

        pipe = DiffusionPipe(
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
            device=device,
        )

        def forward_loop(unet):
            pipe.model.model.diffusion_model = unet
            do_calibrate(
                pipe=pipe,
                calibration_prompts=cali_prompts,
                calib_size=calib_size,
                n_steps=steps,
            )

        register_quant_modules()
        mtq.quantize(unet, quant_config, forward_loop)

        out_path = os.path.join(
            folder_paths.folder_names_and_paths["modelopt"][0][0],
            "{}_{}.pt".format(name, format),
        )
        print(out_path)
        mto.save(unet, out_path)

        return ()


class ModelOptLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "quantized_ckpt": (folder_paths.get_filename_list("modelopt"),),
                "quant_level": ([3.0, 2.5, 2.0, 1.0],),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    OUTPUT_NODE = True
    CATEGORY = "TensorRT"

    def load(self, model, quantized_ckpt, quant_level):
        quantized_ckpt_path = folder_paths.get_full_path("modelopt", quantized_ckpt)
        if not os.path.isfile(quantized_ckpt_path):
            raise FileNotFoundError(f"File {quantized_ckpt_path} does not exist")

        comfy.model_management.unload_all_models()
        comfy.model_management.load_models_gpu([model], force_patch_weights=True)
        unet = model.model.diffusion_model
        register_quant_modules()

        # Lets restore the quantized model
        mto.restore(unet, quantized_ckpt_path)
        quantize_lvl(unet, quant_level)
        mtq.disable_quantizer(unet, filter_func)

        if format == "fp8":
            generate_fp8_scales(unet)

        return (model,)


NODE_CLASS_MAPPINGS = {
    "ModelOpt Quantizer": ModelOptQuantizer,
    "ModelOpt Loader": ModelOptLoader,
}
