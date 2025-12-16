"""
ExportPointCloud node for ComfyUI
Exports point clouds to PLY/PCD files
"""

import torch
import numpy as np
from pathlib import Path
import folder_paths
import os


class ExportPointCloud:
    """
    Export point clouds to PLY or PCD file format.

    Supports merging multiple views and confidence-based filtering.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_clouds": ("POINT_CLOUDS", {
                    "tooltip": "Point cloud data from MVDUST3RInference"
                }),
                "confidence": ("CONFIDENCE", {
                    "tooltip": "Confidence maps from MVDUST3RInference"
                }),
                "filename": ("STRING", {
                    "default": "pointcloud",
                    "tooltip": "Output filename (without extension)"
                }),
                "format": (["ply", "pcd"], {
                    "default": "ply",
                    "tooltip": "Output file format"
                }),
                "merge_views": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge all views into single point cloud"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.1,
                    "tooltip": "Filter points below this confidence"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_pointcloud"
    CATEGORY = "MVDUST3R"
    DESCRIPTION = "Export point clouds to PLY or PCD format"
    OUTPUT_NODE = True

    def export_pointcloud(self, point_clouds, confidence, filename, format,
                         merge_views, confidence_threshold):
        """
        Export point clouds to file.

        Args:
            point_clouds: Point cloud data dictionary
            confidence: Confidence maps
            filename: Output filename
            format: 'ply' or 'pcd'
            merge_views: Whether to merge all views
            confidence_threshold: Minimum confidence to keep points

        Returns:
            Tuple containing output filepath(s)
        """

        print(f"[MVDUST3R] Exporting point cloud(s) to {format.upper()}")

        # Create output directory
        output_dir = Path(folder_paths.get_output_directory()) / "mvdust3r"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract point cloud data
        pts_3d = point_clouds['pts_3d']
        conf = confidence

        print(f"[MVDUST3R] Number of views: {len(pts_3d)}")

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
            # pts_np shape: [H, W, 3]
            h, w, _ = pts_np.shape
            pts_flat = pts_np.reshape(-1, 3)
            conf_flat = conf_np.reshape(-1)

            # Filter by confidence
            mask = conf_flat >= confidence_threshold
            pts_filtered = pts_flat[mask]

            print(f"[MVDUST3R] View {i}: {pts_flat.shape[0]} points, {pts_filtered.shape[0]} after filtering")

            filtered_points.append(pts_filtered)

        # Export
        if merge_views:
            # Merge all views into single point cloud
            merged_points = np.concatenate(filtered_points, axis=0)
            print(f"[MVDUST3R] Merged point cloud: {merged_points.shape[0]} points")

            # Export merged point cloud
            output_path = output_dir / f"{filename}.{format}"
            self._save_pointcloud(merged_points, output_path, format)

            return (str(output_path),)
        else:
            # Export each view separately
            output_paths = []
            for i, pts in enumerate(filtered_points):
                output_path = output_dir / f"{filename}_view{i:02d}.{format}"
                self._save_pointcloud(pts, output_path, format)
                output_paths.append(str(output_path))

            return (",".join(output_paths),)

    def _save_pointcloud(self, points, output_path, format):
        """
        Save point cloud to file.

        Args:
            points: Nx3 numpy array of points
            output_path: Path object for output file
            format: 'ply' or 'pcd'
        """
        print(f"[MVDUST3R] Saving to {output_path}")

        if format == "ply":
            self._save_ply(points, output_path)
        elif format == "pcd":
            self._save_pcd(points, output_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"[MVDUST3R] Saved {points.shape[0]} points to {output_path}")

    def _save_ply(self, points, output_path):
        """Save point cloud to PLY format."""
        with open(output_path, 'w') as f:
            # Write header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {points.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")

            # Write points
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    def _save_pcd(self, points, output_path):
        """Save point cloud to PCD format."""
        with open(output_path, 'w') as f:
            # Write header
            f.write("# .PCD v0.7 - Point Cloud Data file format\n")
            f.write("VERSION 0.7\n")
            f.write("FIELDS x y z\n")
            f.write("SIZE 4 4 4\n")
            f.write("TYPE F F F\n")
            f.write("COUNT 1 1 1\n")
            f.write(f"WIDTH {points.shape[0]}\n")
            f.write("HEIGHT 1\n")
            f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
            f.write(f"POINTS {points.shape[0]}\n")
            f.write("DATA ascii\n")

            # Write points
            for point in points:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")


# Node registration
NODE_CLASS_MAPPINGS = {
    "ExportPointCloud": ExportPointCloud
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExportPointCloud": "Export Point Cloud"
}
