"""
MVDUST3RInference node for ComfyUI
Performs multi-view 3D reconstruction using MV-DUSt3R / MV-DUSt3R+
"""

import torch
import torch.nn.functional as F
import numpy as np
import gc
from copy import deepcopy
from pathlib import Path
import sys

# Import from vendored mvdust3r
current_dir = Path(__file__).parent.parent
vendor_dir = current_dir / "vendor"
sys.path.insert(0, str(vendor_dir))

from mvdust3r.dust3r.inference import inference_mv
from mvdust3r.dust3r.losses import calibrate_camera_pnpransac, estimate_focal_knowing_depth
from mvdust3r.dust3r.utils.device import to_numpy


def process_mvdust3r_output(output, min_conf_thr=3.0, device='cuda', niter_pnp=10):
    """
    Process MV-DUSt3R output to extract point clouds, camera poses, and intrinsics.
    Uses iterative PnP-RANSAC for camera pose estimation.
    """
    with torch.no_grad():
        _, h, w = output['pred1']['pts3d'].shape[0:3]  # [1, H, W, 3]

        # Extract point clouds from predictions
        pts3d = [output['pred1']['pts3d'][0]] + [x['pts3d_in_other_view'][0] for x in output['pred2s']]

        # Extract confidence maps
        conf = torch.stack([output['pred1']['conf'][0]] + [x['conf'][0] for x in output['pred2s']], 0)  # [N, H, W]

        # Calculate confidence threshold
        conf_sorted = conf.reshape(-1).sort()[0]
        conf_thres = conf_sorted[int(conf_sorted.shape[0] * float(min_conf_thr) * 0.01)]
        msk = conf >= conf_thres

        # Estimate focal length from first view
        conf_first = conf[0].reshape(-1)
        conf_sorted_first = conf_first.sort()[0]
        conf_thres_first = conf_sorted_first[int(conf_first.shape[0] * 0.03)]
        valid_first = (conf_first >= conf_thres_first).reshape(h, w)

        focal = estimate_focal_knowing_depth(
            pts3d[0][None].to(device),
            valid_first[None].to(device)
        ).cpu().item()

        # Build intrinsics matrix
        intrinsics = torch.eye(3)
        intrinsics[0, 0] = focal
        intrinsics[1, 1] = focal
        intrinsics[0, 2] = w / 2
        intrinsics[1, 2] = h / 2
        intrinsics = intrinsics.to(device)

        # Create per-view intrinsics
        n_views = len(pts3d)
        intrinsics_all = intrinsics.unsqueeze(0).repeat(n_views, 1, 1)  # [N, 3, 3]

        focals = torch.Tensor([focal]).reshape(1,).repeat(n_views)

        # Create pixel coordinate grid for PnP
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        pixel_coords = torch.stack([x_coords, y_coords], dim=-1).float().to(device)

        # Estimate camera poses using PnP-RANSAC with multiple iterations
        c2ws = []
        for view_idx, (pr_pt, valid) in enumerate(zip(pts3d, msk)):
            # Run PnP-RANSAC
            c2ws_i = calibrate_camera_pnpransac(
                pr_pt.to(device).float().flatten(0, 1)[None],
                pixel_coords.float().flatten(0, 1)[None],
                valid.to(device).flatten(0, 1)[None],
                intrinsics.float()[None]
            )
            c2ws.append(c2ws_i[0])

        cams2world = torch.stack(c2ws, dim=0).cpu()  # [N, 4, 4]

        # Convert to numpy for output
        pts3d_np = [to_numpy(p) for p in pts3d]
        msk_np = to_numpy(msk)
        conf_np = to_numpy(conf)

        return pts3d_np, cams2world, intrinsics_all.cpu(), conf_np, msk_np, focals


def clean_pointcloud_simple(pts3d, conf, masks, percentile=95):
    """
    Simple point cloud cleaning based on confidence and distance outliers.
    """
    cleaned_pts = []
    cleaned_masks = []

    for i, (pts, mask, c) in enumerate(zip(pts3d, masks, conf)):
        # Start with confidence mask
        clean_mask = mask.copy()

        # Remove distance outliers (points too far from median)
        pts_flat = pts.reshape(-1, 3)
        valid_pts = pts_flat[mask.reshape(-1)]

        if len(valid_pts) > 100:
            # Compute distances from centroid
            centroid = np.median(valid_pts, axis=0)
            distances = np.linalg.norm(valid_pts - centroid, axis=1)

            # Remove points beyond percentile threshold
            dist_threshold = np.percentile(distances, percentile)

            # Update mask for outliers
            all_distances = np.linalg.norm(pts_flat - centroid, axis=1).reshape(mask.shape)
            clean_mask = clean_mask & (all_distances <= dist_threshold)

        cleaned_masks.append(clean_mask)

    return cleaned_masks


class MVDUST3RInference:
    """
    Perform multi-view 3D reconstruction using MV-DUSt3R / MV-DUSt3R+.

    Takes multiple 224x224 images and produces 3D point clouds, camera poses, and intrinsics.
    Uses native multi-view inference for best quality.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MVDUST3R_MODEL", {
                    "tooltip": "MVDUST3R model from LoadMVDUST3RModel node"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Prepared 224x224 images (use Prepare Images node first)"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "Confidence threshold (percentile) for point filtering"
                }),
            },
            "optional": {
                "clean_pointcloud": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Remove outlier points based on distance from median"
                }),
                "outlier_percentile": ("FLOAT", {
                    "default": 95.0,
                    "min": 80.0,
                    "max": 100.0,
                    "step": 1.0,
                    "tooltip": "Percentile threshold for outlier removal (higher = keep more)"
                }),
            }
        }

    RETURN_TYPES = ("POINT_CLOUDS", "CAMERA_POSES", "INTRINSICS", "CONFIDENCE")
    RETURN_NAMES = ("point_clouds", "camera_poses", "intrinsics", "confidence")
    FUNCTION = "reconstruct"
    CATEGORY = "MVDUST3R"
    DESCRIPTION = "Multi-view 3D reconstruction from images using MV-DUSt3R"

    def reconstruct(self, model, images, confidence_threshold,
                    clean_pointcloud=True, outlier_percentile=95.0):
        """
        Run MV-DUSt3R inference to reconstruct 3D scene.
        """
        # Validate input image dimensions - must be exactly 224x224
        for i in range(images.shape[0]):
            H, W = images[i].shape[0], images[i].shape[1]
            if H != 224 or W != 224:
                raise ValueError(
                    f"meeep you're wrong - Image {i} is {W}x{H}, but MVDUST3R requires 224x224 images. "
                    f"Use the 'Prepare Images' node first!"
                )

        num_views = images.shape[0]
        variant = getattr(model, '_mvdust3r_variant', 'unknown')

        print(f"[MVDUST3R] Starting inference with {num_views} views")
        print(f"[MVDUST3R] Model variant: {variant}")
        print(f"[MVDUST3R] Image shape: {images.shape}")

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Convert ComfyUI images to mvdust3r format
        # ComfyUI: [B, H, W, C] in [0, 1]
        # mvdust3r expects list of dicts with 'img' key: [1, C, H, W] in [-1, 1]
        imgs = []
        for i in range(num_views):
            img = images[i]  # [H, W, C]
            img = img.permute(2, 0, 1)  # [C, H, W]
            # Convert to [-1, 1]
            img = img * 2.0 - 1.0
            img = img.to(device=device, dtype=dtype)

            # Format as mvdust3r expects
            h, w = img.shape[1], img.shape[2]
            imgs.append({
                'img': img.unsqueeze(0),  # [1, C, H, W]
                'true_shape': torch.tensor([[h, w]]).long().to(device),
                'idx': i,
                'instance': str(i),
            })

        print(f"[MVDUST3R] Converted {len(imgs)} images")

        # Reorder images for better multi-view coverage (from demo.py)
        if len(imgs) < 12:
            if len(imgs) > 3:
                imgs[1], imgs[3] = deepcopy(imgs[3]), deepcopy(imgs[1])
            if len(imgs) > 6:
                imgs[2], imgs[6] = deepcopy(imgs[6]), deepcopy(imgs[2])
        else:
            change_id = len(imgs) // 4 + 1
            imgs[1], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[1])
            change_id = (len(imgs) * 2) // 4 + 1
            imgs[2], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[2])
            change_id = (len(imgs) * 3) // 4 + 1
            imgs[3], imgs[change_id] = deepcopy(imgs[change_id]), deepcopy(imgs[3])

        # Run multi-view inference
        print(f"[MVDUST3R] Running multi-view inference...")
        try:
            with torch.no_grad():
                output = inference_mv(imgs, model, device, verbose=True)

            # Add RGB from original images to output
            output['pred1']['rgb'] = imgs[0]['img'].permute(0, 2, 3, 1)  # [1, H, W, 3]
            for x, img in zip(output['pred2s'], imgs[1:]):
                x['rgb'] = img['img'].permute(0, 2, 3, 1)

            print(f"[MVDUST3R] Inference complete, processing output...")

            # Process output to get camera poses and intrinsics
            pts3d, camera_poses, intrinsics, conf, mask, focals = process_mvdust3r_output(
                output,
                min_conf_thr=confidence_threshold,
                device=device
            )

            # Optional: Clean point cloud by removing outliers
            if clean_pointcloud:
                print(f"[MVDUST3R] Cleaning point cloud (removing outliers beyond {outlier_percentile}th percentile)...")
                mask = clean_pointcloud_simple(pts3d, conf, mask, percentile=outlier_percentile)

            # Free GPU memory
            gc.collect()
            torch.cuda.empty_cache()

            print(f"[MVDUST3R] Reconstruction complete")
            print(f"[MVDUST3R] Point clouds: {len(pts3d)} views")
            print(f"[MVDUST3R] Camera poses: {camera_poses.shape}")
            print(f"[MVDUST3R] Focal length: {focals[0].item():.2f}")

            # Package output
            output_data = {
                'pts_3d': pts3d,
                'camera_poses': camera_poses,
                'intrinsics': intrinsics,
                'confidence': conf,
                'mask': mask,
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
