"""
PrepareImages node for ComfyUI-MVDUST3R
Resizes and crops images to 224x224 for MVDUST3R inference.
"""

import torch
import math
import numpy as np
from PIL import Image


class PrepareImages:
    """
    Prepares images for MVDUST3R by resizing and taking equidistant 224x224 crops.

    Resizes images so the smallest dimension is 224, then takes up to 3 equidistant
    crops along the longer dimension to cover the full image.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {
                    "tooltip": "Input images to prepare for MVDUST3R (any size)"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("prepared_images",)
    FUNCTION = "prepare"
    CATEGORY = "MVDUST3R"
    DESCRIPTION = "Resize and crop images to 224x224 for MVDUST3R inference. Takes up to 3 equidistant crops per image."

    def prepare(self, images):
        """
        Resize images so smallest dimension is 224, then take equidistant 224x224 crops.

        Args:
            images: Tensor of shape [B, H, W, C] in [0, 1]

        Returns:
            Tensor of shape [N_crops, 224, 224, C] where N_crops is sum of crops from all images
        """
        crops = []

        for i in range(images.shape[0]):
            img = images[i]  # [H, W, C]
            H, W, C = img.shape

            # 1. Resize so smallest dimension is 224
            if H < W:
                scale = 224 / H
                new_H = 224
                new_W = int(W * scale)
            else:
                scale = 224 / W
                new_W = 224
                new_H = int(H * scale)

            # Resize using PIL with Lanczos interpolation
            img_np = (img * 255).byte().cpu().numpy()
            pil_img = Image.fromarray(img_np)
            pil_resized = pil_img.resize((new_W, new_H), Image.LANCZOS)
            img_resized = torch.from_numpy(np.array(pil_resized)).float() / 255.0
            img_resized = img_resized.to(img.device)  # [new_H, new_W, C]

            # 2. Calculate number of crops (minimum needed to cover, capped at 3)
            longer_dim = max(new_H, new_W)
            num_crops = min(3, max(1, math.ceil(longer_dim / 224)))

            # 3. Generate equidistant crop positions along the longer dimension
            if num_crops == 1:
                # Single crop - take from the start (image is already 224x224 or close)
                positions = [0]
            else:
                # Multiple crops - space evenly from start to end
                max_start = longer_dim - 224
                positions = [
                    int(j * max_start / (num_crops - 1))
                    for j in range(num_crops)
                ]

            # 4. Take crops
            for pos in positions:
                if new_H > new_W:
                    # Portrait: crop vertically
                    crop = img_resized[pos:pos+224, :, :]
                else:
                    # Landscape or square: crop horizontally
                    crop = img_resized[:, pos:pos+224, :]
                crops.append(crop)

        # Stack all crops into a batch
        result = torch.stack(crops, dim=0)  # [N_crops, 224, 224, C]

        print(f"[MVDUST3R PrepareImages] Input: {images.shape[0]} images")
        print(f"[MVDUST3R PrepareImages] Output: {result.shape[0]} crops of 224x224")

        return (result,)


# Node registration
NODE_CLASS_MAPPINGS = {
    "PrepareImages": PrepareImages
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PrepareImages": "Prepare Images"
}
