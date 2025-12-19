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


def pts3d_to_trimesh(img, pts3d, valid=None):
    """
    Convert structured point cloud to mesh by connecting adjacent pixels.
    Based on MVDUST3R viz.py pts3d_to_trimesh.

    Creates double-sided faces (4 triangles per quad) to avoid face culling issues.

    Args:
        img: [H, W, 3] RGB image (0-1 float or 0-255 uint8)
        pts3d: [H, W, 3] point cloud
        valid: [H, W] boolean mask (optional)

    Returns:
        trimesh.Trimesh object
    """
    import trimesh

    H, W, _ = pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # Make quads from adjacent pixels, each quad = 4 triangles (double-sided)
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left
    idx2 = idx[:-1, +1:].ravel()  # top-right
    idx3 = idx[+1:, :-1].ravel()  # bottom-left
    idx4 = idx[+1:, +1:].ravel()  # bottom-right

    # Four triangles per quad (double-sided to avoid culling)
    faces = np.concatenate([
        np.c_[idx1, idx2, idx3],
        np.c_[idx3, idx2, idx1],  # backward
        np.c_[idx2, idx3, idx4],
        np.c_[idx4, idx3, idx2],  # backward
    ], axis=0)

    # Face colors from image pixels (4x for double-sided faces)
    img_normalized = img.astype(np.float32)
    if img_normalized.max() > 1.0:
        img_normalized = img_normalized / 255.0

    face_colors = np.concatenate([
        img_normalized[:-1, :-1].reshape(-1, 3),
        img_normalized[:-1, :-1].reshape(-1, 3),  # same color for backward face
        img_normalized[+1:, +1:].reshape(-1, 3),
        img_normalized[+1:, +1:].reshape(-1, 3),  # same color for backward face
    ], axis=0)

    # Convert to RGBA (0-255)
    face_colors_rgba = np.zeros((len(face_colors), 4), dtype=np.uint8)
    face_colors_rgba[:, :3] = (face_colors * 255).astype(np.uint8)
    face_colors_rgba[:, 3] = 255

    # Apply validity mask if provided
    if valid is not None:
        valid_flat = valid.ravel()
        # A face is valid if all its vertices are valid
        valid_faces_1 = valid_flat[idx1] & valid_flat[idx2] & valid_flat[idx3]
        valid_faces_2 = valid_flat[idx2] & valid_flat[idx3] & valid_flat[idx4]
        # Duplicate for backward faces
        valid_faces = np.concatenate([valid_faces_1, valid_faces_1, valid_faces_2, valid_faces_2])
        faces = faces[valid_faces]
        face_colors_rgba = face_colors_rgba[valid_faces]

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        face_colors=face_colors_rgba,
        process=False
    )

    return mesh


class MVDUST3RGridMesh:
    """
    Convert MVDUST3R point clouds to mesh using grid-based triangulation.

    Uses the structured nature of MVDUST3R output (each view is H×W grid)
    to create meshes by connecting adjacent pixels as triangles.
    Much faster than Poisson and preserves colors from input images.
    Creates double-sided faces to avoid culling issues.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "point_clouds": ("POINT_CLOUDS", {
                    "tooltip": "Point cloud data from MVDUST3RInference"
                }),
                "images": ("IMAGE", {
                    "tooltip": "Original input images (for colors)"
                }),
                "confidence": ("CONFIDENCE", {
                    "tooltip": "Confidence maps from MVDUST3RInference"
                }),
                "confidence_threshold": ("FLOAT", {
                    "default": 3.0,
                    "min": 0.0,
                    "max": 50.0,
                    "step": 1.0,
                    "tooltip": "Filter lowest N% confidence points (0 = keep all, 3 = remove bottom 3%)"
                }),
                "merge_views": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Merge all views into single mesh"
                }),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("mesh",)
    FUNCTION = "create_mesh"
    CATEGORY = "MVDUST3R"
    DESCRIPTION = "Fast grid-based mesh from MVDUST3R (uses image colors)"

    def create_mesh(self, point_clouds, images, confidence,
                    confidence_threshold, merge_views):
        """
        Create mesh from MVDUST3R structured point clouds.
        """
        import trimesh
        from PIL import Image

        pts_3d = point_clouds['pts_3d']
        conf = confidence

        num_views = len(pts_3d)
        print(f"[MVDUST3R GridMesh] Processing {num_views} views")

        # Get dimensions from first point cloud
        sample_pts = pts_3d[0]
        if isinstance(sample_pts, torch.Tensor):
            sample_pts = sample_pts.detach().cpu().numpy()
        target_h, target_w = sample_pts.shape[:2]
        print(f"[MVDUST3R GridMesh] Point cloud size: {target_h}x{target_w}")

        meshes = []

        for i in range(num_views):
            # Get point cloud
            pts = pts_3d[i]
            if isinstance(pts, torch.Tensor):
                pts = pts.detach().cpu().numpy()

            # Get confidence map
            conf_map = conf[i]
            if isinstance(conf_map, torch.Tensor):
                conf_map = conf_map.detach().cpu().numpy()

            # Debug: print confidence range for first view
            if i == 0:
                print(f"[MVDUST3R GridMesh] Confidence range: {conf_map.min():.2f} - {conf_map.max():.2f}")

            # Get and resize image to match point cloud dimensions
            img = images[i].detach().cpu().numpy()  # [H, W, 3]

            # Resize image to match pts3d dimensions
            img_pil = Image.fromarray((img * 255).astype(np.uint8))
            img_pil = img_pil.resize((target_w, target_h), Image.BILINEAR)
            img_resized = np.array(img_pil).astype(np.float32) / 255.0

            # Create validity mask from confidence (use percentile-based threshold)
            # confidence_threshold is treated as a percentile (0-100) to filter lowest confidence points
            if confidence_threshold > 0:
                threshold_value = np.percentile(conf_map, confidence_threshold)
                valid = conf_map >= threshold_value
            else:
                valid = np.ones(conf_map.shape, dtype=bool)

            # Count valid pixels
            valid_count = valid.sum()
            total_count = valid.size
            print(f"[MVDUST3R GridMesh] View {i}: {valid_count}/{total_count} valid ({100*valid_count/total_count:.1f}%)")

            # Create mesh for this view
            mesh = pts3d_to_trimesh(img_resized, pts, valid=valid)

            if len(mesh.faces) > 0:
                meshes.append(mesh)
                print(f"[MVDUST3R GridMesh] View {i}: {len(mesh.vertices):,} verts, {len(mesh.faces):,} faces")
            else:
                print(f"[MVDUST3R GridMesh] View {i}: No valid faces")

        if len(meshes) == 0:
            raise ValueError("No valid meshes. Try lowering confidence threshold.")

        # Merge or return first mesh
        if merge_views and len(meshes) > 1:
            print(f"[MVDUST3R GridMesh] Merging {len(meshes)} meshes...")
            combined = trimesh.util.concatenate(meshes)
            print(f"[MVDUST3R GridMesh] Final: {len(combined.vertices):,} verts, {len(combined.faces):,} faces")
            return (combined,)
        else:
            return (meshes[0],)


# Node registration
NODE_CLASS_MAPPINGS = {
    "ExportMesh": ExportMesh,
    "MVDUST3RGridMesh": MVDUST3RGridMesh,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ExportMesh": "Export Mesh (Poisson)",
    "MVDUST3RGridMesh": "MVDUST3R Grid Mesh",
}
