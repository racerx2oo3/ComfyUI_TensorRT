import torch
import modelopt.torch.quantization.nn as mnn
from modelopt.torch.quantization.nn import QuantModuleRegistry

from typing import Callable, Iterator, Tuple
from types import ModuleType
from functools import partial
from modelopt.torch.quantization.nn import (
    QuantInputBase,
    TensorQuantizer,
)

import comfy
from comfy.ldm.modules.attention import CrossAttention
from comfy.ldm.modules.diffusionmodules.mmdit import SelfAttention as MMDITAttn
from comfy.ldm.flux.layers import SelfAttention as FLUXAttn

attn_cls = (CrossAttention, MMDITAttn, FLUXAttn)

from modelopt.torch.quantization.plugins.custom import _QuantFunctionalMixin


def _quantized_bmm(self, input, mat2, *args, **kwargs):
    attn, v = input, mat2
    return torch._bmm(
        self.softmax_quantizer(attn), self.v_bmm_quantizer(v), *args, **kwargs
    )


def _quantized_baddbmm(self, input, batch1, batch2, *args, **kwargs):
    q, k = batch1, batch2
    return torch._baddbmm(
        input, self.q_bmm_quantizer(q), self.k_bmm_quantizer(k), *args, **kwargs
    )


class _QuantAttention(_QuantFunctionalMixin):
    """FP8 processor for performing attention-related computations."""

    _functionals_to_replace = [
        (torch, "bmm", _quantized_bmm),
        (torch, "baddbmm", _quantized_baddbmm),
    ]

    @property
    def functionals_to_replace(self) -> Iterator[Tuple[ModuleType, str, Callable]]:
        for package, func_name, quantized_func in self._functionals_to_replace:
            if not hasattr(package, func_name):
                continue
            quantized_func = partial(quantized_func, self)
            yield package, func_name, quantized_func

    def _setup(self):
        self.q_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.k_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.v_bmm_quantizer = TensorQuantizer(QuantInputBase.default_quant_desc_input)
        self.softmax_quantizer = TensorQuantizer(
            QuantInputBase.default_quant_desc_input
        )


def register_quant_modules():
    for attn in attn_cls:
        QuantModuleRegistry.register({attn: attn.__name__})(_QuantAttention)

    for op in comfy.ops.disable_weight_init.__dict__.keys():
        if not hasattr(mnn, f"Quant{op}"):
            continue
        if getattr(comfy.ops.disable_weight_init, op) in QuantModuleRegistry:
            continue
        QuantModuleRegistry.register(
            {getattr(comfy.ops.disable_weight_init, op): f"comfy.{op}"}
        )(QuantModuleRegistry._registry[getattr(torch.nn, op)])
