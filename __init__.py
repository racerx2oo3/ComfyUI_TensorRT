from .tensorrt_convert import NODE_CLASS_MAPPINGS as CONVERT_CLASS_MAP
from .tensorrt_convert import NODE_DISPLAY_NAME_MAPPINGS as CONVERT_NAME_MAP

from .tensorrt_loader import NODE_CLASS_MAPPINGS as LOADER_CLASS_MAP
from .tensorrt_loader import NODE_DISPLAY_NAME_MAPPINGS as LOADER_NAME_MAP

from .onnx_nodes import NODE_CLASS_MAPPING as ONNX_CLASS_MAP
from .onnx_nodes import NODE_DISPLAY_NAME_MAPPINGS as ONNX_NAME_MAP

NODE_CLASS_MAPPINGS = CONVERT_CLASS_MAP | LOADER_CLASS_MAP | ONNX_CLASS_MAP
NODE_DISPLAY_NAME_MAPPINGS = CONVERT_NAME_MAP | LOADER_NAME_MAP | ONNX_NAME_MAP

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']