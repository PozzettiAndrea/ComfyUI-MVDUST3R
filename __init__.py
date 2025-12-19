"""
ComfyUI-MVDUST3R

Custom nodes for MVDUST3R multi-view 3D reconstruction in ComfyUI.

Author: ComfyUI-MVDUST3R Contributors
License: MIT
"""

from .nodes.load_model import LoadMVDUST3RModel
from .nodes.inference import MVDUST3RInference
from .nodes.export_pointcloud import ExportPointCloud
from .nodes.export_mesh import ExportMesh, MVDUST3RGridMesh
from .nodes.visualizer import MVDUST3DVisualizer
from .nodes.blur_detection import BlurDetection

# Node class mappings for ComfyUI
NODE_CLASS_MAPPINGS = {
    "LoadMVDUST3RModel": LoadMVDUST3RModel,
    "MVDUST3RInference": MVDUST3RInference,
    "ExportPointCloud": ExportPointCloud,
    "ExportMesh": ExportMesh,
    "MVDUST3RGridMesh": MVDUST3RGridMesh,
    "MVDUST3DVisualizer": MVDUST3DVisualizer,
    "BlurDetection": BlurDetection,
}

# Display names for nodes in ComfyUI UI
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMVDUST3RModel": "Load MVDUST3R Model",
    "MVDUST3RInference": "MVDUST3R Inference",
    "ExportPointCloud": "Export Point Cloud",
    "ExportMesh": "Export Mesh (Poisson)",
    "MVDUST3RGridMesh": "MVDUST3R Grid Mesh",
    "MVDUST3DVisualizer": "Visualize Point Cloud",
    "BlurDetection": "Blur Detection",
}

# Web directory for UI extensions (if any)
WEB_DIRECTORY = "./web"

__all__ = [
    'NODE_CLASS_MAPPINGS',
    'NODE_DISPLAY_NAME_MAPPINGS',
    'WEB_DIRECTORY',
]

__version__ = "0.1.0"

print("[ComfyUI-MVDUST3R] Loaded successfully!")
print(f"[ComfyUI-MVDUST3R] Version: {__version__}")
print(f"[ComfyUI-MVDUST3R] Registered {len(NODE_CLASS_MAPPINGS)} nodes:")
for node_name in NODE_DISPLAY_NAME_MAPPINGS.values():
    print(f"  - {node_name}")
