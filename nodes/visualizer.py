"""
MVDUST3DVisualizer node for ComfyUI
Renders point clouds to images/video
"""

import torch
import numpy as np
from pathlib import Path


class MVDUST3DVisualizer:
    """
    Render point clouds to images.

    Supports orbit, static, and path rendering modes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_clouds": ("POINT_CLOUDS", {
                    "tooltip": "Point cloud data from MVDUST3RInference"
                }),
                "camera_poses": ("CAMERA_POSES", {
                    "tooltip": "Camera poses from MVDUST3RInference"
                }),
                "confidence": ("CONFIDENCE", {
                    "tooltip": "Confidence maps from MVDUST3RInference"
                }),
                "render_mode": (["orbit", "static", "original_views"], {
                    "default": "orbit",
                    "tooltip": "Rendering camera mode"
                }),
                "num_frames": ("INT", {
                    "default": 60,
                    "min": 1,
                    "max": 360,
                    "step": 1,
                    "tooltip": "Number of frames for orbit/path modes"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Filter points below this confidence"
                }),
                "merge_views": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge all views into single point cloud"
                }),
                "point_size": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 10,
                    "step": 1,
                    "tooltip": "Size of rendered points in pixels"
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "visualize"
    CATEGORY = "MVDUST3R"
    DESCRIPTION = "Render point clouds to images"

    def visualize(self, point_clouds, camera_poses, confidence, render_mode,
                 num_frames, confidence_threshold, merge_views, point_size):
        """
        Render point cloud to images.

        Args:
            point_clouds: Point cloud data dictionary
            camera_poses: Camera poses
            confidence: Confidence maps
            render_mode: 'orbit', 'static', or 'original_views'
            num_frames: Number of frames to generate
            confidence_threshold: Minimum confidence to render points
            merge_views: Whether to merge all views
            point_size: Point size in pixels

        Returns:
            Tuple containing rendered images as ComfyUI IMAGE tensor
        """

        print(f"[MVDUST3R] Rendering point cloud in {render_mode} mode")

        # Import Open3D for visualization
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d is required for visualization. Install with: pip install open3d")

        # Extract point cloud data
        pts_3d = point_clouds['pts_3d']
        conf = confidence

        # Convert to numpy and filter by confidence
        filtered_points = []
        for i, (pts, confidence_map) in enumerate(zip(pts_3d, conf)):
            # Convert to numpy
            if isinstance(pts, torch.Tensor):
                pts_np = pts.detach().cpu().numpy()
            else:
                pts_np = np.array(pts)

            if isinstance(confidence_map, torch.Tensor):
                conf_np = confidence_map.detach().cpu().numpy()
            else:
                conf_np = np.array(confidence_map)

            # Reshape to point cloud
            h, w, _ = pts_np.shape
            pts_flat = pts_np.reshape(-1, 3)
            conf_flat = conf_np.reshape(-1)

            # Filter by confidence
            mask = conf_flat >= confidence_threshold
            pts_filtered = pts_flat[mask]

            filtered_points.append(pts_filtered)

        # Merge views if requested
        if merge_views:
            points = np.concatenate(filtered_points, axis=0)
        else:
            points = filtered_points[0]  # Use first view only

        print(f"[MVDUST3R] Point cloud size: {points.shape[0]} points")

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Estimate colors (for now use uniform color)
        # TODO: Could extract colors from original images
        colors = np.ones_like(points) * [0.7, 0.7, 0.7]
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # Create visualizer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=512, height=512)
        vis.add_geometry(pcd)

        # Set rendering options
        opt = vis.get_render_option()
        opt.point_size = float(point_size)
        opt.background_color = np.array([0, 0, 0])

        # Generate camera viewpoints
        if render_mode == "orbit":
            # Orbit around the point cloud
            rendered_images = self._render_orbit(vis, num_frames)
        elif render_mode == "static":
            # Single static view
            rendered_images = self._render_static(vis)
        elif render_mode == "original_views":
            # Render from original camera poses
            rendered_images = self._render_from_poses(vis, camera_poses)
        else:
            raise ValueError(f"Unsupported render mode: {render_mode}")

        vis.destroy_window()

        # Convert to ComfyUI IMAGE format [B, H, W, C] in [0, 1]
        images_tensor = torch.from_numpy(np.array(rendered_images)).float() / 255.0

        print(f"[MVDUST3R] Rendered {images_tensor.shape[0]} frames")

        return (images_tensor,)

    def _render_orbit(self, vis, num_frames):
        """Render orbit animation."""
        images = []

        ctr = vis.get_view_control()

        for i in range(num_frames):
            # Rotate camera
            ctr.rotate(10.0 * (360.0 / num_frames), 0.0)

            # Update and capture
            vis.poll_events()
            vis.update_renderer()

            # Capture image
            img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            img = (img * 255).astype(np.uint8)
            images.append(img)

        return images

    def _render_static(self, vis):
        """Render single static view."""
        vis.poll_events()
        vis.update_renderer()

        # Capture image
        img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
        img = (img * 255).astype(np.uint8)

        return [img]

    def _render_from_poses(self, vis, camera_poses):
        """Render from original camera poses."""
        images = []

        ctr = vis.get_view_control()

        for i, pose in enumerate(camera_poses):
            # Convert pose to Open3D camera parameters
            if isinstance(pose, torch.Tensor):
                pose_np = pose.detach().cpu().numpy()
            else:
                pose_np = np.array(pose)

            # Set camera pose
            # Note: This is simplified, proper implementation would convert
            # camera-to-world matrix to Open3D camera parameters
            cam_params = ctr.convert_to_pinhole_camera_parameters()
            # TODO: Properly set extrinsic matrix from pose
            ctr.convert_from_pinhole_camera_parameters(cam_params)

            # Update and capture
            vis.poll_events()
            vis.update_renderer()

            # Capture image
            img = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            img = (img * 255).astype(np.uint8)
            images.append(img)

        return images


# Node registration
NODE_CLASS_MAPPINGS = {
    "MVDUST3DVisualizer": MVDUST3DVisualizer
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MVDUST3DVisualizer": "Visualize Point Cloud"
}
