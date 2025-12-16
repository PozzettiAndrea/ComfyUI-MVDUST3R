"""
ExportMesh node for ComfyUI
Converts point clouds to meshes and exports them
"""

import torch
import numpy as np
from pathlib import Path
import folder_paths
import os


class ExportMesh:
    """
    Convert point clouds to mesh and export to OBJ/PLY/GLB format.

    Uses Poisson reconstruction or Ball-Pivoting algorithm.
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
                    "default": "mesh",
                    "tooltip": "Output filename (without extension)"
                }),
                "format": (["obj", "ply", "glb"], {
                    "default": "obj",
                    "tooltip": "Output mesh format"
                }),
                "method": (["poisson", "ball_pivoting"], {
                    "default": "poisson",
                    "tooltip": "Meshing algorithm"
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
                    "tooltip": "Merge all views before meshing"
                }),
            },
            "optional": {
                "poisson_depth": ("INT", {
                    "default": 9,
                    "min": 5,
                    "max": 12,
                    "step": 1,
                    "tooltip": "Poisson reconstruction depth (higher = more detail)"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filepath",)
    FUNCTION = "export_mesh"
    CATEGORY = "MVDUST3R"
    DESCRIPTION = "Convert point cloud to mesh and export"
    OUTPUT_NODE = True

    def export_mesh(self, point_clouds, confidence, filename, format, method,
                   confidence_threshold, merge_views, poisson_depth=9):
        """
        Convert point cloud to mesh and export.

        Args:
            point_clouds: Point cloud data dictionary
            confidence: Confidence maps
            filename: Output filename
            format: 'obj', 'ply', or 'glb'
            method: 'poisson' or 'ball_pivoting'
            confidence_threshold: Minimum confidence to keep points
            merge_views: Whether to merge all views
            poisson_depth: Depth parameter for Poisson reconstruction

        Returns:
            Tuple containing output filepath
        """

        print(f"[MVDUST3R] Converting point cloud to mesh using {method}")

        # Import Open3D for meshing
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("open3d is required for mesh export. Install with: pip install open3d")

        # Create output directory
        output_dir = Path(folder_paths.get_output_directory()) / "mvdust3r"
        output_dir.mkdir(parents=True, exist_ok=True)

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

        # Estimate normals
        print(f"[MVDUST3R] Estimating normals...")
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

        # Orient normals consistently
        pcd.orient_normals_consistent_tangent_plane(k=15)

        # Perform meshing
        print(f"[MVDUST3R] Creating mesh with {method}...")
        if method == "poisson":
            mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
                pcd, depth=poisson_depth
            )

            # Remove low density vertices
            vertices_to_remove = densities < np.quantile(densities, 0.1)
            mesh.remove_vertices_by_mask(vertices_to_remove)

        elif method == "ball_pivoting":
            # Compute bounding box diagonal for radius estimation
            distances = pcd.compute_nearest_neighbor_distance()
            avg_dist = np.mean(distances)
            radii = [avg_dist, avg_dist * 2, avg_dist * 4]

            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
                pcd,
                o3d.utility.DoubleVector(radii)
            )
        else:
            raise ValueError(f"Unsupported meshing method: {method}")

        print(f"[MVDUST3R] Mesh created: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        # Clean up mesh
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()

        print(f"[MVDUST3R] After cleanup: {len(mesh.vertices)} vertices, {len(mesh.triangles)} triangles")

        # Export mesh
        output_path = output_dir / f"{filename}.{format}"

        if format == "obj":
            o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=True)
        elif format == "ply":
            o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=True)
        elif format == "glb":
            o3d.io.write_triangle_mesh(str(output_path), mesh, write_ascii=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"[MVDUST3R] Mesh saved to {output_path}")

        return (str(output_path),)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ExportMesh": ExportMesh
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExportMesh": "Export Mesh"
}
