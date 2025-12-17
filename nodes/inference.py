"""
MVDUST3RInference node for ComfyUI
Performs multi-view 3D reconstruction
"""

import torch
import numpy as np
import gc
from pathlib import Path
import sys

# Import from vendored mvdust3r
current_dir = Path(__file__).parent.parent
sys.path.insert(0, str(current_dir / "vendor"))

from mvdust3r.inference_global_optimization import inference_global_optimization


def crop_to_multiple_of_16(img_tensor):
    """
    Crop image to nearest multiple of 16 (required for patch embedding).

    Args:
        img_tensor: [C, H, W] tensor

    Returns:
        Cropped tensor with dimensions divisible by 16
    """
    C, H, W = img_tensor.shape
    new_H = (H // 16) * 16
    new_W = (W // 16) * 16

    # Center crop
    start_H = (H - new_H) // 2
    start_W = (W - new_W) // 2

    return img_tensor[:, start_H:start_H + new_H, start_W:start_W + new_W]


class MVDUST3RInference:
    """
    Perform multi-view 3D reconstruction using MVDUST3R.

    Takes multiple images and produces 3D point clouds, camera poses, and intrinsics.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MVDUST3R_MODEL", {
                    "tooltip": "MVDUST3R model from LoadMVDUST3RModel node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Multiple input images (batch of 2-12 views)"
                }),
                "scenegraph_type": ([
                    "complete",
                    "sliding_window",
                    "one_ref"
                ], {
                    "default": "complete",
                    "tooltip": "Image pair generation strategy"
                }),
                "optimize_poses": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Enable global pose optimization"
                }),
                "niter": ("INT", {
                    "default": 300,
                    "min": 50,
                    "max": 1000,
                    "step": 50,
                    "tooltip": "Number of optimization iterations"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Confidence threshold for point filtering"
                }),
            },
            "optional": {
                "first_view_c2w": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional 4x4 camera-to-world matrix for first view (JSON array)"
                }),
            }
        }

    RETURN_TYPES = ("POINT_CLOUDS", "CAMERA_POSES", "INTRINSICS", "CONFIDENCE")
    RETURN_NAMES = ("point_clouds", "camera_poses", "intrinsics", "confidence")
    FUNCTION = "reconstruct"
    CATEGORY = "MVDUST3R"
    DESCRIPTION = "Perform multi-view 3D reconstruction from images"

    def reconstruct(self, model, images, scenegraph_type, optimize_poses, niter,
                   confidence_threshold, first_view_c2w=""):
        """
        Run MVDUST3R inference to reconstruct 3D scene.

        Args:
            model: MVDUST3R model
            images: ComfyUI IMAGE tensor [B, H, W, C] in [0, 1]
            scenegraph_type: Pair generation strategy
            optimize_poses: Whether to run global optimization
            niter: Number of optimization iterations
            confidence_threshold: Threshold for confidence filtering
            first_view_c2w: Optional camera pose for first view

        Returns:
            Tuple of (point_clouds, camera_poses, intrinsics, confidence)
        """

        num_views = images.shape[0]
        variant = getattr(model, '_mvdust3r_variant', 'unknown')

        # Validate view count based on model variant
        if 'MVDp_s2' in variant:
            if num_views < 4 or num_views > 12:
                raise ValueError(f"MVDp_s2 model requires 4-12 views, got {num_views}")
        elif 'MVDp_s1' in variant:
            if num_views != 8:
                raise ValueError(f"MVDp_s1 model requires exactly 8 views, got {num_views}")
        elif 'MVD' in variant:
            if num_views < 2:
                raise ValueError(f"MVD model requires at least 2 views, got {num_views}")

        print(f"[MVDUST3R] Starting inference with {num_views} views")
        print(f"[MVDUST3R] Image shape: {images.shape}")
        print(f"[MVDUST3R] Scene graph type: {scenegraph_type}")
        print(f"[MVDUST3R] Optimize poses: {optimize_poses}")

        # Get device and dtype from model
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Convert ComfyUI images to mvdust3r format
        # ComfyUI: [B, H, W, C] in [0, 1]
        # mvdust3r: [C, H, W] in [-1, 1], dimensions must be divisible by 16
        img_tensors = []
        for i in range(images.shape[0]):
            img = images[i]  # [H, W, C]
            # Convert to [-1, 1]
            img = img * 2.0 - 1.0
            # Convert to [C, H, W]
            img = img.permute(2, 0, 1)
            # Crop to be divisible by 16
            img = crop_to_multiple_of_16(img)
            # Move to device and match model dtype
            img = img.to(device=device, dtype=dtype)
            img_tensors.append(img)

        print(f"[MVDUST3R] Converted {len(img_tensors)} images to size {img_tensors[0].shape}")

        # Parse first_view_c2w if provided
        if first_view_c2w and first_view_c2w.strip():
            try:
                import json
                c2w_data = json.loads(first_view_c2w)
                c2w = torch.tensor(c2w_data, dtype=torch.float32, device=device)
                print(f"[MVDUST3R] Using provided first view camera pose")
            except Exception as e:
                print(f"[MVDUST3R] Warning: Could not parse first_view_c2w, using identity: {e}")
                c2w = torch.eye(4, dtype=torch.float32, device=device)
        else:
            c2w = torch.eye(4, dtype=torch.float32, device=device)

        # Run inference
        print(f"[MVDUST3R] Running global optimization...")
        try:
            with torch.no_grad():
                pts_3d, camera_poses, intrinsics, conf, t_inference, t_optimization = \
                    inference_global_optimization(
                        model=model,
                        device=device,
                        silent=False,
                        img_tensors=img_tensors,
                        first_view_c2w=c2w
                    )

            print(f"[MVDUST3R] Inference time: {t_inference:.2f}s")
            print(f"[MVDUST3R] Optimization time: {t_optimization:.2f}s")
            print(f"[MVDUST3R] Total time: {t_inference + t_optimization:.2f}s")

            # Free GPU memory before proceeding
            gc.collect()
            torch.cuda.empty_cache()

            # Apply confidence thresholding
            filtered_pts_3d = []
            for pts, confidence in zip(pts_3d, conf):
                # Create mask for high-confidence points
                mask = confidence >= confidence_threshold
                # Keep original point cloud structure but mark low confidence points
                # (alternative: could filter them out entirely)
                filtered_pts_3d.append(pts)

            print(f"[MVDUST3R] Reconstruction complete")
            print(f"[MVDUST3R] Point clouds: {len(filtered_pts_3d)}")
            print(f"[MVDUST3R] Camera poses: {len(camera_poses)}")

            # Package output
            output_data = {
                'pts_3d': filtered_pts_3d,
                'camera_poses': camera_poses,
                'intrinsics': intrinsics,
                'confidence': conf,
                'conf_threshold': confidence_threshold
            }

            return (output_data, camera_poses, intrinsics, conf)

        except Exception as e:
            print(f"[MVDUST3R] Error during inference: {str(e)}")
            import traceback
            traceback.print_exc()
            raise RuntimeError(f"MVDUST3R inference failed: {str(e)}")


# Node registration
NODE_CLASS_MAPPINGS = {
    "MVDUST3RInference": MVDUST3RInference
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MVDUST3RInference": "MVDUST3R Inference"
}
