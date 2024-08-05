from .tensorrt_convert import DYNAMIC_TRT_MODEL_CONVERSION, STATIC_TRT_MODEL_CONVERSION
from .model_opt import ModelOptQuantizer, ModelOptLoader
from .tensorrt_loader import TrTUnet, TensorRTLoader

NODE_CLASS_MAPPINGS = {
    "DYNAMIC_TRT_MODEL_CONVERSION": DYNAMIC_TRT_MODEL_CONVERSION,
    "STATIC_TRT_MODEL_CONVERSION": STATIC_TRT_MODEL_CONVERSION,
    "TensorRTLoader": TensorRTLoader,
    "ModelOptQuantizer": ModelOptQuantizer,
    "ModelOptLoader": ModelOptLoader,
}


NODE_DISPLAY_NAME_MAPPINGS = {
    "DYNAMIC_TRT_MODEL_CONVERSION": "DYNAMIC TRT_MODEL CONVERSION",
    "STATIC TRT_MODEL CONVERSION": STATIC_TRT_MODEL_CONVERSION,
    "TensorRTLoader": "TensorRT Loader",
    "ModelOpt Quantizer": ModelOptQuantizer,
    "ModelOpt Loader": ModelOptLoader,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
