#!/usr/bin/env python3
"""
ComfyUI-MVDUST3R Installer

Automatically installs pytorch3d and other dependencies based on detected
PyTorch, CUDA, and Python versions.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path


def get_python_version():
    """Get Python version as string like '310' for 3.10."""
    return f"{sys.version_info.major}{sys.version_info.minor}"


def get_torch_info():
    """Get PyTorch and CUDA version info."""
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # e.g., "2.8.0"
        cuda_version = torch.version.cuda  # e.g., "12.8" or None
        return torch_version, cuda_version
    except ImportError:
        return None, None


def parse_version(version_str):
    """Parse version string to tuple of ints."""
    if not version_str:
        return (0, 0, 0)
    parts = version_str.replace('+', '.').split('.')
    result = []
    for p in parts[:3]:
        try:
            result.append(int(''.join(c for c in p if c.isdigit()) or '0'))
        except ValueError:
            result.append(0)
    while len(result) < 3:
        result.append(0)
    return tuple(result)


def get_pytorch3d_wheel_url(torch_version, cuda_version, python_version):
    """
    Construct pytorch3d wheel URL based on environment.

    PyTorch3D wheels are hosted at:
    https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/

    Format: py{pyver}_cu{cudaver}_pyt{torchver}/pytorch3d-{version}-cp{pyver}-cp{pyver}-linux_x86_64.whl
    """
    if not torch_version or not cuda_version:
        return None

    # Parse versions
    torch_parts = parse_version(torch_version)
    cuda_parts = parse_version(cuda_version)

    # Format for URL: torch "2.4.1" -> "241", cuda "12.1" -> "121"
    torch_short = f"{torch_parts[0]}{torch_parts[1]}{torch_parts[2]}"
    cuda_short = f"{cuda_parts[0]}{cuda_parts[1]}"

    # Known pytorch3d versions and their compatibility
    # pytorch3d 0.7.8 supports up to PyTorch 2.5
    # For newer PyTorch versions, we'll try building from source or using latest available

    base_url = "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels"

    # Try exact match first, then fall back to compatible versions
    wheel_configs = [
        # (pytorch3d_version, torch_version_short, cuda_version_short)
        ("0.7.8", "251", "124"),  # PyTorch 2.5.1, CUDA 12.4
        ("0.7.8", "250", "124"),  # PyTorch 2.5.0, CUDA 12.4
        ("0.7.8", "241", "121"),  # PyTorch 2.4.1, CUDA 12.1
        ("0.7.8", "240", "121"),  # PyTorch 2.4.0, CUDA 12.1
        ("0.7.7", "231", "121"),  # PyTorch 2.3.1, CUDA 12.1
        ("0.7.7", "230", "121"),  # PyTorch 2.3.0, CUDA 12.1
        ("0.7.6", "220", "121"),  # PyTorch 2.2.0, CUDA 12.1
        ("0.7.5", "210", "121"),  # PyTorch 2.1.0, CUDA 12.1
        ("0.7.5", "210", "118"),  # PyTorch 2.1.0, CUDA 11.8
        ("0.7.4", "200", "118"),  # PyTorch 2.0.0, CUDA 11.8
        ("0.7.4", "200", "117"),  # PyTorch 2.0.0, CUDA 11.7
    ]

    # Find best matching wheel
    best_match = None
    for p3d_ver, torch_ver, cuda_ver in wheel_configs:
        # Check if this wheel might be compatible
        wheel_torch = parse_version(torch_ver[0] + "." + torch_ver[1] + "." + torch_ver[2] if len(torch_ver) == 3 else torch_ver[0] + "." + torch_ver[1:])

        # Accept if torch major.minor matches or is close
        if torch_parts[0] == int(torch_ver[0]) and abs(torch_parts[1] - int(torch_ver[1])) <= 1:
            best_match = (p3d_ver, torch_ver, cuda_ver)
            break

    if best_match:
        p3d_ver, torch_ver, cuda_ver = best_match
        wheel_name = f"pytorch3d-{p3d_ver}-cp{python_version}-cp{python_version}-linux_x86_64.whl"
        folder = f"py{python_version}_cu{cuda_ver}_pyt{torch_ver}"
        return f"{base_url}/{folder}/{wheel_name}"

    return None


def install_pytorch3d_from_source():
    """Install pytorch3d from source as fallback."""
    print("[MVDUST3R] Installing pytorch3d from source (this may take a while)...")
    print("[MVDUST3R] This requires CUDA toolkit and may take 10-30 minutes...")

    try:
        # Install build dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "fvcore", "iopath", "ninja"
        ])

        # Set environment variables for build
        env = os.environ.copy()
        env["FORCE_CUDA"] = "1"
        env["MAX_JOBS"] = "4"  # Limit parallel jobs to avoid OOM

        # Install from GitHub with --no-build-isolation to use system torch
        # This is required because pytorch3d's setup.py needs torch during build
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--no-build-isolation",
            "git+https://github.com/facebookresearch/pytorch3d.git@stable"
        ], env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[MVDUST3R] Failed to install pytorch3d from source: {e}")
        return False


def install_pytorch3d():
    """Install pytorch3d with automatic version detection."""
    print("\n" + "="*60)
    print("ComfyUI-MVDUST3R: Installing pytorch3d")
    print("="*60 + "\n")

    # Check if already installed
    try:
        import pytorch3d
        print(f"[MVDUST3R] pytorch3d {pytorch3d.__version__} already installed")
        return True
    except ImportError:
        pass

    # Get environment info
    torch_version, cuda_version = get_torch_info()
    python_version = get_python_version()

    print(f"[MVDUST3R] Detected environment:")
    print(f"  Python: {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    print(f"  PyTorch: {torch_version or 'not installed'}")
    print(f"  CUDA: {cuda_version or 'not available'}")
    print(f"  Platform: {platform.system()} {platform.machine()}")

    if not torch_version:
        print("[MVDUST3R] ERROR: PyTorch not installed. Please install PyTorch first.")
        return False

    # Only Linux x86_64 has prebuilt wheels
    if platform.system() == "Linux" and platform.machine() == "x86_64" and cuda_version:
        wheel_url = get_pytorch3d_wheel_url(torch_version, cuda_version, python_version)

        if wheel_url:
            print(f"[MVDUST3R] Trying prebuilt wheel: {wheel_url}")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    wheel_url
                ])
                print("[MVDUST3R] pytorch3d installed successfully from wheel!")
                return True
            except subprocess.CalledProcessError:
                print("[MVDUST3R] Prebuilt wheel failed, trying alternatives...")

    # Try pip install (might work for some configurations)
    print("[MVDUST3R] Trying pip install pytorch3d...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "pytorch3d"
        ])
        print("[MVDUST3R] pytorch3d installed successfully!")
        return True
    except subprocess.CalledProcessError:
        print("[MVDUST3R] pip install failed, trying from source...")

    # Last resort: build from source
    return install_pytorch3d_from_source()


def install_requirements():
    """Install requirements from requirements.txt."""
    print("\n" + "="*60)
    print("ComfyUI-MVDUST3R: Installing Requirements")
    print("="*60 + "\n")

    requirements_path = Path(__file__).parent / "requirements.txt"

    if requirements_path.exists():
        print(f"[MVDUST3R] Installing from {requirements_path}")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-r", str(requirements_path)
            ])
            print("[MVDUST3R] Requirements installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[MVDUST3R] Failed to install requirements: {e}")
            return False
    else:
        print("[MVDUST3R] No requirements.txt found")
        return True


def main():
    """Main installation routine."""
    print("\n" + "="*60)
    print("ComfyUI-MVDUST3R Installer")
    print("="*60 + "\n")

    success = True

    # Install basic requirements first
    if not install_requirements():
        success = False

    # Install pytorch3d
    if not install_pytorch3d():
        print("\n[MVDUST3R] WARNING: pytorch3d installation failed!")
        print("[MVDUST3R] Some features may not work without pytorch3d.")
        print("[MVDUST3R] You can try installing it manually:")
        print("  pip install pytorch3d")
        print("  OR")
        print("  pip install git+https://github.com/facebookresearch/pytorch3d.git@stable")
        success = False

    if success:
        print("\n" + "="*60)
        print("[MVDUST3R] Installation completed successfully!")
        print("="*60 + "\n")
    else:
        print("\n" + "="*60)
        print("[MVDUST3R] Installation completed with warnings.")
        print("[MVDUST3R] Note: pytorch3d is OPTIONAL - basic inference works without it.")
        print("[MVDUST3R] pytorch3d is only needed for advanced rendering features.")
        print("="*60 + "\n")

    return 0  # Always return success since pytorch3d is optional


if __name__ == "__main__":
    sys.exit(main())
