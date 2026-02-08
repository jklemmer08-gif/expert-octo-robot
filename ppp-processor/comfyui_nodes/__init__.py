"""PPP Processor — ComfyUI Custom Nodes.

Wraps the PPP video processing pipeline (matting, upscaling, VR, QA)
as ComfyUI nodes for visual workflow composition.
"""

from .nodes.analyze import PPP_VideoAnalyze
from .nodes.encode import PPP_EncodeVideo
from .nodes.matte import PPP_MatteVideo
from .nodes.pipeline import PPP_FullPipeline
from .nodes.qa import PPP_QACheck
from .nodes.upscale import PPP_UpscaleVideo
from .nodes.vr import PPP_VRMerge, PPP_VRSplit

NODE_CLASS_MAPPINGS = {
    "PPP_VideoAnalyze": PPP_VideoAnalyze,
    "PPP_MatteVideo": PPP_MatteVideo,
    "PPP_UpscaleVideo": PPP_UpscaleVideo,
    "PPP_EncodeVideo": PPP_EncodeVideo,
    "PPP_VRSplit": PPP_VRSplit,
    "PPP_VRMerge": PPP_VRMerge,
    "PPP_QACheck": PPP_QACheck,
    "PPP_FullPipeline": PPP_FullPipeline,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PPP_VideoAnalyze": "PPP Video Analyze",
    "PPP_MatteVideo": "PPP Matte Video",
    "PPP_UpscaleVideo": "PPP Upscale Video",
    "PPP_EncodeVideo": "PPP Encode Video",
    "PPP_VRSplit": "PPP VR Split",
    "PPP_VRMerge": "PPP VR Merge",
    "PPP_QACheck": "PPP QA Check",
    "PPP_FullPipeline": "PPP Full Pipeline",
}

WEB_DIRECTORY = None

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
