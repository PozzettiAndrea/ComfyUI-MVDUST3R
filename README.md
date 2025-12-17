# ComfyUI-MVDUST3R

ComfyUI custom nodes for **MVDUST3R** (Multi-View DUSt3R+), a state-of-the-art single-stage, pose-free multi-view 3D reconstruction system.

## Features

- **Multi-View 3D Reconstruction**: Reconstruct 3D scenes from 2-12 camera views in ~2 seconds
- **Pose-Free**: Automatically estimates camera poses without prior calibration
- **Point Cloud Export**: Export high-quality point clouds to PLY/PCD formats
- **Mesh Generation**: Convert point clouds to meshes using Poisson/Ball-Pivoting algorithms
- **Visualization**: Render point clouds with orbit/static/path camera modes
- **ComfyUI Integration**: Seamless integration with ComfyUI workflows

## Installation

### Option 1: ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "MVDUST3R"
3. Click Install

### Option 2: Manual Installation
```bash
cd ComfyUI/custom_nodes/
git clone https://github.com/yourusername/ComfyUI-MVDUST3R.git
cd ComfyUI-MVDUST3R
pip install -r requirements.txt
```

### Dependencies
- Python >= 3.10
- PyTorch >= 2.0.0 with CUDA support
- 8GB+ VRAM recommended (4-view reconstruction)
- See `requirements.txt` for full list

## Nodes

### 1. Load MVDUST3R Model
**Purpose**: Load the MVDUST3R model checkpoint

**Inputs**:
- `model_name`: Model variant selection
- `device`: cuda/cpu
- `precision`: float32/float16/bfloat16

**Outputs**:
- `MODEL`: MVDUST3R model instance

### 2. MVDUST3R Inference
**Purpose**: Perform multi-view 3D reconstruction

**Inputs**:
- `model`: MVDUST3R model
- `images`: Multiple input images (IMAGE batch)
- `pair_generation`: complete/sliding_window/one_ref
- `optimize_poses`: Enable global pose optimization
- `n_iter`: Optimization iterations (default: 300)
- `confidence_threshold`: Filter low-confidence points

**Outputs**:
- `POINT_CLOUDS`: 3D point clouds per view
- `CAMERA_POSES`: Camera-to-world matrices
- `INTRINSICS`: Camera intrinsics matrices
- `CONFIDENCE`: Per-pixel confidence maps

### 3. Export Point Cloud
**Purpose**: Export point clouds to files

**Inputs**:
- `point_clouds`: Point cloud data
- `confidence`: Confidence maps
- `filename`: Output filename
- `format`: ply/pcd
- `merge_views`: Merge all views into single file

**Outputs**:
- `filepath`: Path to saved file

### 4. Export Mesh
**Purpose**: Convert point clouds to meshes

**Inputs**:
- `point_clouds`: Point cloud data
- `confidence`: Confidence maps
- `method`: poisson/ball_pivoting
- `filename`: Output filename
- `format`: obj/ply/glb

**Outputs**:
- `filepath`: Path to saved mesh

### 5. Visualize Point Cloud
**Purpose**: Render point clouds to images

**Inputs**:
- `point_clouds`: Point cloud data
- `camera_poses`: Camera poses
- `render_mode`: orbit/static/path
- `num_frames`: Number of frames for video
- `resolution`: Output resolution

**Outputs**:
- `IMAGE`: Rendered frames (ComfyUI IMAGE tensor)

## Example Data

Sample images are included in `assets/examples/office_room/` for quick testing:

```
assets/examples/office_room/
├── frame_01.png
├── frame_02.png
├── frame_03.png
├── frame_04.png
├── frame_05.png
├── frame_06.png
├── frame_07.png
└── frame_08.png
```

These 8 frames are from the **TUM RGB-D Dataset** (freiburg1_room sequence) - a 49-second trajectory through a full office room with loop closure, perfect for testing multi-view reconstruction.

### Data Attribution

The example images are from the [TUM RGB-D Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download):

> J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers.
> **A Benchmark for the Evaluation of RGB-D SLAM Systems**.
> In Proc. of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2012.

## Example Workflow

```
Load Images (4-6 views)
    ↓
Load MVDUST3R Model
    ↓
MVDUST3R Inference
    ├─→ Export Point Cloud (.ply)
    ├─→ Export Mesh (.obj)
    └─→ Visualize Point Cloud
```

See `workflows/basic_reconstruction.json` for a complete example.

## Usage Tips

1. **Input Images**: Use 4-12 views of the same object/scene from different angles
2. **Image Quality**: Higher resolution images (512x512+) produce better results
3. **Camera Coverage**: Ensure good viewpoint coverage around the object
4. **Confidence Threshold**: Lower values (1.0-2.0) for clean objects, higher (5.0+) for noisy scenes
5. **VRAM**: Reduce image resolution or number of views if running out of memory

## Model Download

Models are automatically downloaded from HuggingFace Hub to:
```
ComfyUI/models/mvdust3r/{model_name}/
```

Default model: `naver/MV-DUSt3R-Plus`

## Technical Details

### Image Format Conversion
ComfyUI uses `[B, H, W, C]` float32 in [0, 1]
MVDUST3R expects `[B, C, H, W]` float32 in [-1, 1]

Conversion happens automatically inside the inference node.

### Custom Data Types
- `MVDUST3R_MODEL`: Model wrapper
- `POINT_CLOUDS`: List of point cloud tensors
- `CAMERA_POSES`: List of 4x4 camera-to-world matrices
- `INTRINSICS`: List of 3x3 intrinsics matrices
- `CONFIDENCE`: List of confidence maps

## Troubleshooting

### Out of Memory
- Reduce image resolution
- Use fewer views (4-6 instead of 12)
- Enable float16 precision
- Close other GPU-intensive applications

### Poor Reconstruction Quality
- Ensure sufficient viewpoint coverage
- Use higher resolution images
- Increase confidence threshold to filter noise
- Try different pair generation strategies

### Model Download Fails
- Check internet connection
- Verify HuggingFace Hub access
- Manually download to `ComfyUI/models/mvdust3r/`

## Credits

- **MVDUST3R**: [Original Paper](https://arxiv.org/abs/2412.06974) by Meta/Facebook Research
- **DUSt3R**: Foundation work by [NAVER Labs](https://github.com/naver/dust3r)
- **ComfyUI**: [ComfyUI](https://github.com/comfyanonymous/ComfyUI) by comfyanonymous
- **Example Data**: [TUM RGB-D Dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset) by TU Munich Computer Vision Group

## License

MIT License - See LICENSE file for details

Vendored MVDUST3R code retains its original license (see `vendor/mvdust3r/LICENSE`).

## Citation

If you use this in your research, please cite:
```bibtex
@article{mvdust3r2024,
  title={MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds},
  author={...},
  journal={arXiv preprint arXiv:2410.17504},
  year={2024}
}
```
