"""
Blur Detection Node for ComfyUI-MVDUST3R

Measures image sharpness using Laplacian variance to filter out blurry frames
before 3D reconstruction.
"""

import torch
import numpy as np
import cv2
import json


class BlurDetection:
    """Detect blur in images using Laplacian variance."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Batch of images to analyze"}),
                "threshold": ("FLOAT", {
                    "default": 100.0,
                    "min": 0.0,
                    "max": 10000.0,
                    "step": 10.0,
                    "tooltip": "Blur score threshold. Higher = require sharper images. Typical range: 50-500"
                }),
            },
            "optional": {
                "return_all_if_none_pass": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If no images pass threshold, return all images instead of empty"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("sharp_images", "blurry_images", "blur_scores", "sharp_indices")
    FUNCTION = "detect_blur"
    CATEGORY = "MVDUST3R"
    DESCRIPTION = "Analyze image sharpness using Laplacian variance. Higher scores = sharper."

    def detect_blur(self, images, threshold, return_all_if_none_pass=True):
        """
        Analyze blur in a batch of images using Laplacian variance.

        Args:
            images: [B, H, W, C] tensor in [0, 1]
            threshold: Minimum blur score to pass (higher = sharper required)
            return_all_if_none_pass: If True, return all images when none pass threshold

        Returns:
            sharp_images: Images above threshold
            blurry_images: Images below threshold
            blur_scores: JSON array of [index, score] pairs
            sharp_indices: Comma-separated indices of sharp frames
        """
        batch_size = images.shape[0]

        scores = []
        sharp_indices = []
        blurry_indices = []

        for i in range(batch_size):
            # Convert to numpy uint8 for OpenCV
            img = images[i].cpu().numpy()
            img_uint8 = (img * 255).astype(np.uint8)

            # Calculate Laplacian variance (higher = sharper)
            gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
            score = cv2.Laplacian(gray, cv2.CV_64F).var()
            scores.append((i, float(score)))

            if score >= threshold:
                sharp_indices.append(i)
            else:
                blurry_indices.append(i)

        # Handle case where nothing passes
        if len(sharp_indices) == 0 and return_all_if_none_pass:
            print(f"[BLUR_DETECTION] Warning: No images passed threshold {threshold}, returning all")
            sharp_indices = list(range(batch_size))
            blurry_indices = []

        # Split images
        if len(sharp_indices) > 0:
            sharp_images = images[sharp_indices]
        else:
            sharp_images = torch.empty(0, *images.shape[1:])

        if len(blurry_indices) > 0:
            blurry_images = images[blurry_indices]
        else:
            blurry_images = torch.empty(0, *images.shape[1:])

        # Format outputs
        blur_scores_json = json.dumps(scores, indent=2)
        sharp_indices_str = ",".join(map(str, sharp_indices))

        # Log results
        print(f"[BLUR_DETECTION] Analyzed {batch_size} images")
        print(f"[BLUR_DETECTION] Threshold: {threshold}")
        print(f"[BLUR_DETECTION] Sharp: {len(sharp_indices)}, Blurry: {len(blurry_indices)}")

        # Show score range
        if scores:
            min_score = min(s[1] for s in scores)
            max_score = max(s[1] for s in scores)
            print(f"[BLUR_DETECTION] Score range: {min_score:.1f} - {max_score:.1f}")

        return (sharp_images, blurry_images, blur_scores_json, sharp_indices_str)


NODE_CLASS_MAPPINGS = {
    "BlurDetection": BlurDetection,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BlurDetection": "Blur Detection",
}
