import shutil
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(HERE, "assets")
COMFYUI_INPUT_DIR = os.path.join(HERE, "..", "..", "input")

def copy_assets():
    if not os.path.isdir(ASSETS_DIR):
        return

    os.makedirs(COMFYUI_INPUT_DIR, exist_ok=True)

    for item in os.listdir(ASSETS_DIR):
        src = os.path.join(ASSETS_DIR, item)
        dst = os.path.join(COMFYUI_INPUT_DIR, item)

        if os.path.isfile(src):
            if not os.path.exists(dst):
                shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if not os.path.exists(dst):
                shutil.copytree(src, dst)

copy_assets()
