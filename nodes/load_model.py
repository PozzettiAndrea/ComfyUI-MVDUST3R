"""
LoadMVDUST3RModel node for ComfyUI
Loads the MVDUST3R model from HuggingFace or local checkpoint
"""

import torch
import os
from pathlib import Path
import folder_paths

# Import from vendored mvdust3r
import sys
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir / "vendor"))

from mvdust3r.dust3r.model import AsymmetricCroCo3DStereoMultiView, load_model

# Global model cache to avoid reloading
_MODEL_CACHE = {}


class LoadMVDUST3RModel:
    """
    Load MVDUST3R model for multi-view 3D reconstruction.

    Models are automatically downloaded from HuggingFace Hub or loaded from local checkpoint.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ([
                    "naver/MV-DUSt3R-Plus",
                    "naver/MV-DUSt3R",
                ], {
                    "default": "naver/MV-DUSt3R-Plus",
                    "tooltip": "Select model variant from HuggingFace Hub"
                }),
                "device": (["cuda", "cpu"], {
                    "default": "cuda",
                    "tooltip": "Device to load model on"
                }),
                "precision": (["auto", "float32", "float16", "bfloat16"], {
                    "default": "auto",
                    "tooltip": "Model precision (auto selects based on device)"
                }),
            },
            "optional": {
                "custom_checkpoint": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to custom checkpoint file (overrides model_name)"
                }),
            }
        }

    RETURN_TYPES = ("MVDUST3R_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MVDUST3R"
    DESCRIPTION = "Load MVDUST3R model for multi-view 3D reconstruction"

    def load_model(self, model_name, device, precision, custom_checkpoint=""):
        """
        Load MVDUST3R model from HuggingFace or local checkpoint.

        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
            precision: Model precision
            custom_checkpoint: Optional path to local checkpoint

        Returns:
            Tuple containing model wrapper
        """

        # Use custom checkpoint if provided
        if custom_checkpoint and custom_checkpoint.strip():
            checkpoint_path = custom_checkpoint.strip()
            cache_key = f"custom:{checkpoint_path}:{device}:{precision}"
        else:
            checkpoint_path = model_name
            cache_key = f"{model_name}:{device}:{precision}"

        # Check cache
        if cache_key in _MODEL_CACHE:
            print(f"[MVDUST3R] Using cached model: {cache_key}")
            return (_MODEL_CACHE[cache_key],)

        # Determine device
        if device == "cuda" and not torch.cuda.is_available():
            print("[MVDUST3R] CUDA not available, falling back to CPU")
            device = "cpu"

        # Determine precision
        if precision == "auto":
            if device == "cpu":
                precision = "float32"
            else:
                # Check GPU capability
                if torch.cuda.is_available():
                    capability = torch.cuda.get_device_capability()
                    if capability[0] >= 8:  # Ampere or newer (RTX 30xx+, A100, H100)
                        precision = "bfloat16"
                    else:
                        precision = "float16"
                else:
                    precision = "float32"

        print(f"[MVDUST3R] Loading model: {checkpoint_path}")
        print(f"[MVDUST3R] Device: {device}, Precision: {precision}")

        try:
            # Load model
            if os.path.isfile(checkpoint_path):
                # Load from local checkpoint
                print(f"[MVDUST3R] Loading from local checkpoint: {checkpoint_path}")
                model = load_model(checkpoint_path, device=device, verbose=True)
            else:
                # Load from HuggingFace Hub
                print(f"[MVDUST3R] Downloading from HuggingFace: {checkpoint_path}")
                model = AsymmetricCroCo3DStereoMultiView.from_pretrained(checkpoint_path)
                model = model.to(device)

            # Set precision
            if precision == "float16":
                model = model.half()
            elif precision == "bfloat16":
                model = model.bfloat16()
            # float32 is already default

            # Set to eval mode
            model.eval()

            # Cache the model
            _MODEL_CACHE[cache_key] = model

            print(f"[MVDUST3R] Model loaded successfully")

            return (model,)

        except Exception as e:
            print(f"[MVDUST3R] Error loading model: {str(e)}")
            raise RuntimeError(f"Failed to load MVDUST3R model: {str(e)}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "LoadMVDUST3RModel": LoadMVDUST3RModel
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMVDUST3RModel": "Load MVDUST3R Model"
}
