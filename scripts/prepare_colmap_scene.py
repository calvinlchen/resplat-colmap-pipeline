"""
Prepare a COLMAP scene for ReSplat from a folder of raw images.

The final output layout matches scripts/infer_colmap.py:

    <scene_dir>/
      images/
      sparse/0/
        cameras.bin
        images.bin
        points3D.bin

Run with no arguments to open the folder-picker GUI, or use CLI arguments.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tif",
    ".tiff",
    ".bmp",
    ".webp",
}

MODEL_PRESETS = [
    "dl3dv_8v_512x960",
    "dl3dv_16v_540x960",
    "dl3dv_8v_256x448",
    "dl3dv_16v_256x448",
    "dl3dv_32v_256x448",
    "dl3dv_8v_256x448_small",
    "dl3dv_8v_256x448_large",
]

MODEL_PRESET_CHECKPOINTS = {
    "dl3dv_8v_512x960": "resplat-base-dl3dv-512x960-view8-8179ed87.pth",
    "dl3dv_16v_540x960": "resplat-base-dl3dv-540x960-view16-a72dc6d0.pth",
    "dl3dv_8v_256x448": "resplat-base-dl3dv-256x448-view8-1934a04c.pth",
    "dl3dv_16v_256x448": "resplat-base-dl3dv-256x448-view16-f38bf984.pth",
    "dl3dv_32v_256x448": "resplat-base-dl3dv-256x448-view32-439b63a6.pth",
    "dl3dv_8v_256x448_small": "resplat-small-dl3dv-256x448-view8-548993fe.pth",
    "dl3dv_8v_256x448_large": "resplat-large-dl3dv-256x448-view8-62f1703a.pth",
}

MODEL_PRESET_DESCRIPTIONS = {
    "dl3dv_8v_512x960": (
        "Meaning: Base DL3DV model, 8 context views, high-resolution 512x960 training.\n"
        "Best use: Recommended default for good quality when you have enough VRAM."
    ),
    "dl3dv_16v_540x960": (
        "Meaning: Base DL3DV model, 16 context views, high-resolution 540x960 training.\n"
        "Best use: Scenes with many well-registered images where extra view coverage helps."
    ),
    "dl3dv_8v_256x448": (
        "Meaning: Base DL3DV model, 8 context views, low-resolution 256x448 training.\n"
        "Best use: Faster runs or lower-VRAM GPUs, with lower image detail."
    ),
    "dl3dv_16v_256x448": (
        "Meaning: Base DL3DV model, 16 context views, low-resolution 256x448 training.\n"
        "Best use: Lower-VRAM 16-view inference when your scene has many usable views."
    ),
    "dl3dv_32v_256x448": (
        "Meaning: Base DL3DV model, 32 context views, low-resolution 256x448 training.\n"
        "Best use: Broad scene coverage from many registered images, while staying low-res."
    ),
    "dl3dv_8v_256x448_small": (
        "Meaning: Small ViT-S DL3DV model, 8 context views, low-resolution 256x448 training.\n"
        "Best use: Lowest memory/faster inference when quality is less critical."
    ),
    "dl3dv_8v_256x448_large": (
        "Meaning: Large ViT-L DL3DV model, 8 context views, low-resolution 256x448 training, init-only.\n"
        "Best use: Testing the larger backbone without recurrent refinement."
    ),
}

MODEL_DOWNLOAD_BASE_URL = "https://huggingface.co/haofeixu/resplat/resolve/main"

CAMERA_MODELS = [
    "OPENCV",
    "SIMPLE_RADIAL",
    "RADIAL",
    "SIMPLE_PINHOLE",
    "PINHOLE",
    "OPENCV_FISHEYE",
]

CAMERA_MODEL_DESCRIPTIONS = {
    "OPENCV": (
        "Use for most phone, action-camera, and normal lens image sets where focal length, "
        "principal point, and radial/tangential distortion should be estimated."
    ),
    "SIMPLE_RADIAL": (
        "Use for simple datasets from one camera when you want COLMAP to estimate one focal "
        "length and lightweight radial distortion."
    ),
    "RADIAL": (
        "Use when a normal lens has more noticeable radial distortion than SIMPLE_RADIAL "
        "can model."
    ),
    "SIMPLE_PINHOLE": (
        "Use for already-undistorted images with a single focal length and no lens "
        "distortion model."
    ),
    "PINHOLE": (
        "Use for already-undistorted images where horizontal and vertical focal lengths may "
        "differ."
    ),
    "OPENCV_FISHEYE": (
        "Use for fisheye or very wide-angle lenses. Avoid it for normal phone/camera photos."
    ),
}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass
class PrepareConfig:
    image_dir: Path
    output_root: Path
    scene_name: str
    colmap: str
    camera_model: str
    single_camera: bool
    max_image_size: int
    overwrite: bool


@dataclass
class ReSplatRunConfig:
    enabled: bool
    model_preset: str
    output_root: Path
    device: str
    save_images: bool
    save_ply: bool
    save_video: bool
    save_depth: bool
    no_eval: bool
    render_chunk_size: int


def sanitize_scene_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name.strip())
    name = name.strip("._-")
    return name or "scene"


def count_images(image_dir: Path) -> int:
    return sum(
        1
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def ensure_colmap(colmap: str) -> str:
    if os.path.isfile(colmap):
        return colmap

    resolved = shutil.which(colmap)
    if resolved:
        return resolved

    raise RuntimeError(
        "COLMAP executable was not found. Install COLMAP and add it to PATH, "
        "or select colmap.exe in the app."
    )


def model_checkpoint_relative_path(model_preset: str) -> Path:
    try:
        checkpoint_name = MODEL_PRESET_CHECKPOINTS[model_preset]
    except KeyError as exc:
        raise RuntimeError(f"Unknown model preset: {model_preset}") from exc
    return Path("pretrained") / checkpoint_name


def model_checkpoint_path(model_preset: str) -> Path:
    return repo_root() / model_checkpoint_relative_path(model_preset)


def model_download_url(model_preset: str) -> str:
    try:
        checkpoint_name = MODEL_PRESET_CHECKPOINTS[model_preset]
    except KeyError as exc:
        raise RuntimeError(f"Unknown model preset: {model_preset}") from exc
    return f"{MODEL_DOWNLOAD_BASE_URL}/{checkpoint_name}"


def model_checkpoint_exists(model_preset: str) -> bool:
    checkpoint = model_checkpoint_path(model_preset)
    return checkpoint.is_file() and checkpoint.stat().st_size > 0


def ensure_model_checkpoint(
    model_preset: str,
    log: Callable[[str], None] = print,
) -> Path:
    checkpoint = model_checkpoint_path(model_preset)
    if model_checkpoint_exists(model_preset):
        log(f"Using downloaded model checkpoint: {checkpoint}")
        return checkpoint

    url = model_download_url(model_preset)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    temp_path = checkpoint.with_suffix(checkpoint.suffix + ".download")

    log(f"Model checkpoint not found: {checkpoint}")
    log(f"Downloading {model_preset} from:")
    log(url)

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "resplat-colmap-pipeline/1.0"},
    )

    try:
        with urllib.request.urlopen(request) as response:
            total_size = int(response.headers.get("Content-Length", "0") or "0")
            downloaded = 0
            next_progress = 0

            with temp_path.open("wb") as file:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    file.write(chunk)
                    downloaded += len(chunk)

                    if total_size:
                        progress = int(downloaded * 100 / total_size)
                        if progress >= next_progress:
                            log(
                                "  Downloaded "
                                f"{downloaded / (1024 * 1024):.1f} / "
                                f"{total_size / (1024 * 1024):.1f} MiB "
                                f"({progress}%)"
                            )
                            next_progress = progress + 10

        temp_path.replace(checkpoint)
    except (OSError, urllib.error.URLError) as exc:
        try:
            temp_path.unlink()
        except OSError:
            pass
        raise RuntimeError(
            "Could not download the selected ReSplat model. Check your network "
            f"connection or download it manually from MODEL_ZOO.md into {checkpoint.parent}."
        ) from exc

    log(f"Downloaded model checkpoint: {checkpoint}")
    return checkpoint


def run_command(
    command: list[str],
    log: Callable[[str], None],
    cwd: Path | None = None,
) -> None:
    log("\n$ " + " ".join(f'"{part}"' if " " in part else part for part in command))
    if cwd is not None:
        log(f"Working directory: {cwd}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        cwd=str(cwd) if cwd is not None else None,
    )

    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {command[0]}")


def run_resplat_inference(
    scene_dir: Path,
    scene_name: str,
    cfg: ReSplatRunConfig,
    log: Callable[[str], None] = print,
) -> Path:
    scene_dir = scene_dir.resolve()
    images_dir_name = validate_resplat_scene(scene_dir)
    ensure_model_checkpoint(cfg.model_preset, log)
    output_dir = (cfg.output_root / sanitize_scene_name(scene_name)).resolve()
    if output_dir == scene_dir:
        output_dir = (cfg.output_root / f"{sanitize_scene_name(scene_name)}_resplat").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        str(repo_root() / "scripts" / "infer_colmap.py"),
        "--model_preset",
        cfg.model_preset,
        "--scene_path",
        str(scene_dir.resolve()),
        "--output_dir",
        str(output_dir),
        "--sparse_dir",
        "sparse/0",
        "--images_dir",
        images_dir_name,
        "--device",
        cfg.device,
        "--render_chunk_size",
        str(cfg.render_chunk_size),
    ]

    if cfg.save_images:
        command.append("--save_images")
    else:
        command.append("--no_save_images")
    if cfg.save_ply:
        command.append("--save_ply")
    if cfg.save_video:
        command.append("--save_video")
    if cfg.save_depth:
        command.append("--save_depth")
    if cfg.no_eval:
        command.append("--no_eval")

    log("\nRunning ReSplat inference on prepared scene...")
    log(f"Using image folder: {images_dir_name}")
    run_command(command, log, cwd=repo_root())
    log(f"\nReSplat results saved to: {output_dir}")
    return output_dir


def validate_resplat_scene(scene_dir: Path) -> str:
    scene_dir = scene_dir.resolve()
    if scene_dir.name == "_colmap_work":
        raise RuntimeError(
            "Selected _colmap_work, which is an internal scratch folder. "
            f"Select the prepared scene folder instead: {scene_dir.parent}"
        )

    image_dir_candidates = ["images", "images_4"]
    images_dir_name = next(
        (
            candidate
            for candidate in image_dir_candidates
            if (scene_dir / candidate).exists() and (scene_dir / candidate).is_dir()
        ),
        None,
    )
    sparse_dir = scene_dir / "sparse" / "0"

    if not scene_dir.exists() or not scene_dir.is_dir():
        raise RuntimeError(f"Prepared scene folder does not exist: {scene_dir}")
    if images_dir_name is None:
        raise RuntimeError(
            "Prepared scene is missing an image folder. Expected either "
            f"{scene_dir / 'images'} or {scene_dir / 'images_4'}"
        )
    if not sparse_dir.exists() or not sparse_dir.is_dir():
        raise RuntimeError(f"Prepared scene is missing sparse/0/: {sparse_dir}")
    if not (
        (sparse_dir / "cameras.bin").exists()
        or (sparse_dir / "cameras.txt").exists()
    ):
        raise RuntimeError(f"Prepared scene is missing cameras.bin/txt: {sparse_dir}")
    if not (
        (sparse_dir / "images.bin").exists()
        or (sparse_dir / "images.txt").exists()
    ):
        raise RuntimeError(f"Prepared scene is missing images.bin/txt: {sparse_dir}")
    return images_dir_name


def remove_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def copy_dir(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def choose_sparse_model(sparse_root: Path) -> Path:
    candidates = [path for path in sparse_root.iterdir() if path.is_dir()]
    if not candidates:
        raise RuntimeError(f"COLMAP mapper did not create a sparse model in {sparse_root}")

    numeric = sorted(
        [path for path in candidates if path.name.isdigit()],
        key=lambda path: int(path.name),
    )
    return numeric[0] if numeric else sorted(candidates)[0]


def normalize_undistorted_sparse_path(undistorted_root: Path) -> Path:
    sparse_root = undistorted_root / "sparse"
    sparse_zero = sparse_root / "0"

    if (sparse_zero / "cameras.bin").exists() or (sparse_zero / "cameras.txt").exists():
        return sparse_zero

    if (sparse_root / "cameras.bin").exists() or (sparse_root / "cameras.txt").exists():
        return sparse_root

    raise RuntimeError(
        "COLMAP image_undistorter did not create cameras.bin/txt under "
        f"{sparse_root} or {sparse_zero}"
    )


def validate_safe_output(cfg: PrepareConfig, scene_dir: Path) -> None:
    image_dir = cfg.image_dir.resolve()
    scene_dir_resolved = scene_dir.resolve()

    if image_dir == scene_dir_resolved:
        raise RuntimeError(
            "The output scene directory cannot be the same folder as the input images."
        )

    try:
        image_dir.relative_to(scene_dir_resolved)
    except ValueError:
        pass
    else:
        raise RuntimeError(
            "The input image folder cannot be inside the output scene directory. "
            "Choose a separate output root."
        )


def prepare_scene(cfg: PrepareConfig, log: Callable[[str], None] = print) -> Path:
    cfg.image_dir = cfg.image_dir.resolve()
    cfg.output_root = cfg.output_root.resolve()
    cfg.scene_name = sanitize_scene_name(cfg.scene_name)
    cfg.colmap = ensure_colmap(cfg.colmap)

    if not cfg.image_dir.exists() or not cfg.image_dir.is_dir():
        raise RuntimeError(f"Image folder does not exist: {cfg.image_dir}")

    image_count = count_images(cfg.image_dir)
    if image_count < 3:
        raise RuntimeError(
            f"Found {image_count} supported image(s). COLMAP needs at least 3 images."
        )

    scene_dir = cfg.output_root / cfg.scene_name
    validate_safe_output(cfg, scene_dir)

    if scene_dir.exists() and not cfg.overwrite:
        raise RuntimeError(
            f"Output scene already exists: {scene_dir}. Enable overwrite or choose a "
            "different scene name."
        )

    cfg.output_root.mkdir(parents=True, exist_ok=True)

    if cfg.overwrite and scene_dir.exists():
        log(f"Removing existing scene directory: {scene_dir}")
        shutil.rmtree(scene_dir)

    scene_dir.mkdir(parents=True, exist_ok=True)

    work_dir = scene_dir / "_colmap_work"
    database_path = work_dir / "database.db"
    sparse_raw_dir = work_dir / "sparse_raw"
    undistorted_dir = work_dir / "undistorted"

    remove_dir(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    sparse_raw_dir.mkdir(parents=True, exist_ok=True)

    log(f"Input images: {cfg.image_dir}")
    log(f"Supported images found: {image_count}")
    log(f"Output scene: {scene_dir}")
    log(f"COLMAP: {cfg.colmap}")

    feature_command = [
        cfg.colmap,
        "feature_extractor",
        "--database_path",
        str(database_path),
        "--image_path",
        str(cfg.image_dir),
        "--ImageReader.camera_model",
        cfg.camera_model,
        "--ImageReader.single_camera",
        "1" if cfg.single_camera else "0",
    ]
    run_command(feature_command, log)

    run_command(
        [
            cfg.colmap,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
        ],
        log,
    )

    run_command(
        [
            cfg.colmap,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(cfg.image_dir),
            "--output_path",
            str(sparse_raw_dir),
        ],
        log,
    )

    sparse_model = choose_sparse_model(sparse_raw_dir)
    log(f"Using sparse model: {sparse_model}")

    undistort_command = [
        cfg.colmap,
        "image_undistorter",
        "--image_path",
        str(cfg.image_dir),
        "--input_path",
        str(sparse_model),
        "--output_path",
        str(undistorted_dir),
        "--output_type",
        "COLMAP",
    ]
    if cfg.max_image_size > 0:
        undistort_command.extend(["--max_image_size", str(cfg.max_image_size)])
    run_command(undistort_command, log)

    undistorted_images = undistorted_dir / "images"
    undistorted_sparse = normalize_undistorted_sparse_path(undistorted_dir)

    if not undistorted_images.exists():
        raise RuntimeError(f"Undistorted image folder not found: {undistorted_images}")

    final_images = scene_dir / "images"
    final_sparse = scene_dir / "sparse" / "0"
    copy_dir(undistorted_images, final_images)
    final_sparse.parent.mkdir(parents=True, exist_ok=True)
    copy_dir(undistorted_sparse, final_sparse)

    log("\nPrepared ReSplat COLMAP scene:")
    log(str(scene_dir))
    log("\nSuggested ReSplat command:")
    log(
        "python scripts/infer_colmap.py "
        "--model_preset dl3dv_8v_512x960 "
        f'--scene_path "{scene_dir}" '
        "--output_dir output/colmap_inference "
        "--save_images --save_ply"
    )

    return scene_dir


def default_output_root() -> Path:
    return repo_root() / "datasets" / "colmap-custom"


def default_results_root() -> Path:
    return repo_root() / "results" / "colmap-custom"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a ReSplat-compatible COLMAP scene from raw images."
    )
    parser.add_argument("--image_dir", type=Path, help="Folder containing raw images.")
    parser.add_argument(
        "--output_root",
        type=Path,
        default=default_output_root(),
        help="Directory where the prepared scene folder will be created.",
    )
    parser.add_argument(
        "--scene_name",
        type=str,
        default=None,
        help="Name for the prepared scene folder. Defaults to the image folder name.",
    )
    parser.add_argument(
        "--colmap",
        type=str,
        default="colmap",
        help="COLMAP executable name or full path to colmap.exe.",
    )
    parser.add_argument(
        "--camera_model",
        type=str,
        default="OPENCV",
        choices=CAMERA_MODELS,
        help=(
            "Initial COLMAP camera model for raw images. Distorted models are OK "
            "because image_undistorter produces the final PINHOLE-style scene."
        ),
    )
    parser.add_argument(
        "--per_image_camera",
        action="store_true",
        help="Do not force one shared camera calibration for all images.",
    )
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=2000,
        help="Maximum undistorted image size. Use 0 to keep COLMAP's default.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing prepared scene with the same name.",
    )
    parser.add_argument(
        "--no_gui",
        action="store_true",
        help="Fail instead of opening the GUI when --image_dir is omitted.",
    )
    parser.add_argument(
        "--run_resplat",
        action="store_true",
        help=(
            "Run scripts/infer_colmap.py. If --resplat_scene_path is provided, "
            "COLMAP preparation is skipped."
        ),
    )
    parser.add_argument(
        "--resplat_scene_path",
        type=Path,
        default=None,
        help="Existing prepared scene folder to use for ReSplat without running COLMAP.",
    )
    parser.add_argument(
        "--resplat_output_root",
        type=Path,
        default=default_results_root(),
        help="Directory where ReSplat result folders will be created.",
    )
    parser.add_argument(
        "--model_preset",
        type=str,
        default="dl3dv_8v_512x960",
        choices=MODEL_PRESETS,
        help="ReSplat model preset to use when --run_resplat is set.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for ReSplat inference, for example cuda:0 or cpu.",
    )
    parser.add_argument(
        "--render_chunk_size",
        type=int,
        default=10,
        help="Number of target views rendered at once by ReSplat.",
    )
    parser.add_argument(
        "--save_video",
        action="store_true",
        help="Save a smoothed ReSplat render video.",
    )
    parser.add_argument(
        "--save_depth",
        action="store_true",
        help="Save ReSplat depth visualizations.",
    )
    parser.add_argument(
        "--no_save_ply",
        action="store_true",
        help="Do not export gaussians.ply when running ReSplat.",
    )
    parser.add_argument(
        "--no_save_resplat_images",
        action="store_true",
        help="Do not save rendered ReSplat images.",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Compute metrics against held-out target images.",
    )
    return parser.parse_args(argv)


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    root = tk.Tk()
    root.title("COLMAP and ReSplat Processor")
    root.geometry("920x760")

    class Tooltip:
        def __init__(self, parent: tk.Tk) -> None:
            self.parent = parent
            self.tip: tk.Toplevel | None = None
            self.text = ""

        def show(self, text: str, x: int | None = None, y: int | None = None) -> None:
            if not text:
                self.hide()
                return
            self.text = text
            if self.tip is None or not self.tip.winfo_exists():
                self.tip = tk.Toplevel(self.parent)
                self.tip.wm_overrideredirect(True)
                self.tip.wm_attributes("-topmost", True)
                self.tip.transient(self.parent)
                label = tk.Label(
                    self.tip,
                    text=text,
                    justify="left",
                    background="#ffffe0",
                    relief="solid",
                    borderwidth=1,
                    padx=6,
                    pady=4,
                    wraplength=420,
                )
                label.pack()
            else:
                label = self.tip.winfo_children()[0]
                label.configure(text=text)

            if x is None:
                x = self.parent.winfo_pointerx() + 16
            if y is None:
                y = self.parent.winfo_pointery() + 12
            self.tip.wm_geometry(f"+{x}+{y}")
            self.tip.lift()
            self.tip.update_idletasks()

        def hide(self) -> None:
            if self.tip is not None and self.tip.winfo_exists():
                self.tip.destroy()
            self.tip = None

    tooltip = Tooltip(root)

    def add_tooltip(widget: tk.Widget, get_text: Callable[[], str]) -> None:
        def show(event: tk.Event) -> None:
            tooltip.show(get_text(), event.x_root + 16, event.y_root + 12)

        widget.bind("<Enter>", show)
        widget.bind("<Motion>", show)
        widget.bind("<Leave>", lambda _event: tooltip.hide())

    def create_described_dropdown(
        parent: tk.Widget,
        variable: tk.StringVar,
        values: list[str],
        descriptions: dict[str, str],
        width: int = 34,
        indicator_text: Callable[[str], str] | None = None,
        status_text: Callable[[str], str] | None = None,
    ) -> tk.Button:
        popup: tk.Toplevel | None = None
        active_description: tk.Toplevel | None = None

        button = tk.Button(
            parent,
            textvariable=variable,
            anchor="w",
            width=width,
            relief="raised",
            padx=8,
        )

        def close_popup() -> None:
            nonlocal popup, active_description
            if active_description is not None and active_description.winfo_exists():
                active_description.destroy()
            if popup is not None and popup.winfo_exists():
                popup.destroy()
            popup = None
            active_description = None

        def option_description(value: str) -> str:
            description = descriptions.get(value, "")
            if status_text is None:
                return description
            status = status_text(value)
            if not status:
                return description
            return f"{description}\n\n{status}" if description else status

        def set_row_background(row: tk.Widget, color: str) -> None:
            try:
                row.configure(background=color)
            except tk.TclError:
                pass
            for child in row.winfo_children():
                try:
                    child.configure(background=color)
                except tk.TclError:
                    pass

        def show_description(row: tk.Widget, value: str) -> None:
            nonlocal active_description
            if popup is None or not popup.winfo_exists():
                return

            set_row_background(row, "#e6f0ff")
            if active_description is None or not active_description.winfo_exists():
                active_description = tk.Toplevel(root)
                active_description.wm_overrideredirect(True)
                active_description.wm_attributes("-topmost", True)
                active_description.transient(root)
                label = tk.Label(
                    active_description,
                    text=option_description(value),
                    justify="left",
                    background="#ffffe0",
                    relief="solid",
                    borderwidth=1,
                    padx=6,
                    pady=4,
                    wraplength=420,
                )
                label.pack()
            else:
                label = active_description.winfo_children()[0]
                label.configure(text=option_description(value))

            popup.update_idletasks()
            active_description.update_idletasks()
            x = popup.winfo_rootx() + popup.winfo_width() + 8
            y = row.winfo_rooty()
            screen_width = root.winfo_screenwidth()
            screen_height = root.winfo_screenheight()
            if x + active_description.winfo_reqwidth() > screen_width:
                x = max(0, popup.winfo_rootx() - active_description.winfo_reqwidth() - 8)
            if y + active_description.winfo_reqheight() > screen_height:
                y = max(0, screen_height - active_description.winfo_reqheight() - 8)
            active_description.wm_geometry(f"+{x}+{y}")
            active_description.lift()

        def hide_description(row: tk.Widget) -> None:
            nonlocal active_description
            set_row_background(row, "#ffffff")
            if active_description is not None and active_description.winfo_exists():
                active_description.destroy()
            active_description = None

        def open_popup() -> None:
            nonlocal popup
            if popup is not None and popup.winfo_exists():
                close_popup()
                return

            tooltip.hide()
            popup = tk.Toplevel(root)
            popup.wm_overrideredirect(True)
            popup.wm_attributes("-topmost", True)
            popup.transient(root)
            popup.configure(background="#ffffff")

            option_frame = tk.Frame(
                popup,
                background="#ffffff",
                relief="solid",
                borderwidth=1,
            )
            option_frame.grid(row=0, column=0, sticky="nw")

            def bind_row(widget: tk.Widget, row: tk.Widget, value: str) -> None:
                widget.bind("<Enter>", lambda _event: show_description(row, value))
                widget.bind("<Motion>", lambda _event: show_description(row, value))
                widget.bind("<Leave>", lambda _event: hide_description(row))
                widget.bind(
                    "<ButtonRelease-1>",
                    lambda _event: (variable.set(value), close_popup()),
                )

            for row_index, value in enumerate(values):
                row = tk.Frame(
                    option_frame,
                    background="#ffffff",
                )
                row.grid(row=row_index, column=0, sticky="ew")
                row.columnconfigure(0, weight=1)

                text_label = tk.Label(
                    row,
                    text=value,
                    anchor="w",
                    background="#ffffff",
                    padx=10,
                    pady=4,
                    width=width,
                )
                text_label.grid(row=0, column=0, sticky="ew")

                icon_label = tk.Label(
                    row,
                    text=indicator_text(value) if indicator_text is not None else "",
                    anchor="e",
                    background="#ffffff",
                    padx=8,
                    pady=4,
                    width=2,
                )
                icon_label.grid(row=0, column=1, sticky="e")

                bind_row(row, row, value)
                bind_row(text_label, row, value)
                bind_row(icon_label, row, value)

            x = button.winfo_rootx()
            y = button.winfo_rooty() + button.winfo_height()
            popup.wm_geometry(f"+{x}+{y}")
            popup.lift()
            popup.bind("<Escape>", lambda _event: close_popup())
            popup.bind("<FocusOut>", lambda _event: close_popup())
            popup.focus_force()

        button.configure(command=open_popup)
        add_tooltip(button, lambda: option_description(variable.get()))
        return button

    image_dir_var = tk.StringVar()
    output_root_var = tk.StringVar(value=str(default_output_root()))
    scene_name_var = tk.StringVar()
    colmap_var = tk.StringVar(value=shutil.which("colmap") or "colmap")
    camera_model_var = tk.StringVar(value="OPENCV")
    single_camera_var = tk.BooleanVar(value=True)
    max_image_size_var = tk.StringVar(value="2000")
    overwrite_var = tk.BooleanVar(value=False)
    step2_scene_path_var = tk.StringVar()
    run_step2_after_step1_var = tk.BooleanVar(value=False)
    resplat_output_root_var = tk.StringVar(value=str(default_results_root()))
    model_preset_var = tk.StringVar(value="dl3dv_8v_512x960")
    device_var = tk.StringVar(value="cuda:0")
    render_chunk_size_var = tk.StringVar(value="10")
    save_resplat_images_var = tk.BooleanVar(value=True)
    save_ply_var = tk.BooleanVar(value=True)
    save_video_var = tk.BooleanVar(value=False)
    save_depth_var = tk.BooleanVar(value=False)
    eval_var = tk.BooleanVar(value=False)

    def append_log_direct(text: str) -> None:
        log_box.configure(state="normal")
        log_box.insert("end", text + "\n")
        log_box.see("end")
        log_box.configure(state="disabled")
        root.update_idletasks()

    def append_log(text: str) -> None:
        root.after(0, append_log_direct, text)

    def browse_images() -> None:
        selected = filedialog.askdirectory(title="Select folder containing images")
        if selected:
            image_dir_var.set(selected)
            if not scene_name_var.get().strip():
                scene_name_var.set(sanitize_scene_name(Path(selected).name))

    def browse_output() -> None:
        selected = filedialog.askdirectory(title="Select output root folder")
        if selected:
            output_root_var.set(selected)

    def browse_colmap() -> None:
        selected = filedialog.askopenfilename(
            title="Select COLMAP executable",
            filetypes=[("Executable", "*.exe"), ("All files", "*.*")],
        )
        if selected:
            colmap_var.set(selected)

    def browse_resplat_output() -> None:
        selected = filedialog.askdirectory(title="Select ReSplat output root folder")
        if selected:
            resplat_output_root_var.set(selected)

    def browse_prepared_scene() -> None:
        selected = filedialog.askdirectory(
            title="Select prepared scene folder containing images/ and sparse/0/"
        )
        if selected:
            selected_path = Path(selected)
            if selected_path.name == "_colmap_work":
                selected_path = selected_path.parent
            step2_scene_path_var.set(str(selected_path))
            if not scene_name_var.get().strip():
                scene_name_var.set(sanitize_scene_name(selected_path.name))

    def set_widget_tree_enabled(widget: tk.Widget, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        try:
            widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            set_widget_tree_enabled(child, enabled)

    def set_running(active_step: str | None) -> None:
        step1_enabled = active_step is None or active_step == "step1"
        step2_enabled = active_step is None or active_step == "step2"
        set_widget_tree_enabled(step1_frame, step1_enabled)
        set_widget_tree_enabled(step2_frame, step2_enabled)
        run_step1_button.configure(
            state="disabled" if active_step is not None else "normal"
        )
        run_step2_button.configure(
            state="disabled" if active_step is not None else "normal"
        )

    def build_step2_config() -> ReSplatRunConfig | None:
        try:
            render_chunk_size = int(render_chunk_size_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid value", "Render chunk size must be an integer."
            )
            return None

        return ReSplatRunConfig(
            enabled=True,
            model_preset=model_preset_var.get(),
            output_root=Path(resplat_output_root_var.get()),
            device=device_var.get().strip() or "cuda:0",
            save_images=save_resplat_images_var.get(),
            save_ply=save_ply_var.get(),
            save_video=save_video_var.get(),
            save_depth=save_depth_var.get(),
            no_eval=not eval_var.get(),
            render_chunk_size=render_chunk_size,
        )

    def start_step1() -> None:
        try:
            max_image_size = int(max_image_size_var.get())
        except ValueError:
            messagebox.showerror(
                "Invalid value", "Max image size must be an integer."
            )
            return

        scene_name = scene_name_var.get().strip()
        if not scene_name:
            scene_name = sanitize_scene_name(Path(image_dir_var.get()).name)
            scene_name_var.set(scene_name)

        cfg = PrepareConfig(
            image_dir=Path(image_dir_var.get()),
            output_root=Path(output_root_var.get()),
            scene_name=scene_name,
            colmap=colmap_var.get().strip() or "colmap",
            camera_model=camera_model_var.get(),
            single_camera=single_camera_var.get(),
            max_image_size=max_image_size,
            overwrite=overwrite_var.get(),
        )
        resplat_cfg = build_step2_config() if run_step2_after_step1_var.get() else None
        if run_step2_after_step1_var.get() and resplat_cfg is None:
            return

        set_running("step1")
        append_log("Starting Step 1: COLMAP preparation...")

        def worker() -> None:
            try:
                scene_dir = prepare_scene(cfg, append_log)
                root.after(0, lambda scene_dir=scene_dir: step2_scene_path_var.set(str(scene_dir)))
                result_dir = None
                if resplat_cfg is not None:
                    root.after(0, lambda: set_running("step2"))
                    result_dir = run_resplat_inference(
                        scene_dir,
                        cfg.scene_name,
                        resplat_cfg,
                        append_log,
                    )
            except Exception as exc:
                root.after(
                    0,
                    lambda exc=exc: messagebox.showerror("Preparation failed", str(exc)),
                )
                root.after(0, lambda exc=exc: append_log_direct(f"\nERROR: {exc}"))
            else:
                root.after(
                    0,
                    lambda scene_dir=scene_dir, result_dir=result_dir: messagebox.showinfo(
                        "Done",
                        (
                            f"Step 1 complete. Ready for Step 2:\n{scene_dir}"
                            if result_dir is None
                            else f"Step 1 scene:\n{scene_dir}\n\nStep 2 results:\n{result_dir}"
                        ),
                    ),
                )
            finally:
                root.after(0, lambda: set_running(None))

        threading.Thread(target=worker, daemon=True).start()

    def start_step2() -> None:
        resplat_cfg = build_step2_config()
        if resplat_cfg is None:
            return

        scene_dir = Path(step2_scene_path_var.get())
        scene_name = scene_name_var.get().strip() or sanitize_scene_name(scene_dir.name)
        scene_name_var.set(scene_name)

        set_running("step2")
        append_log("Starting Step 2: ReSplat inference...")

        def worker() -> None:
            try:
                result_dir = run_resplat_inference(
                    scene_dir,
                    scene_name,
                    resplat_cfg,
                    append_log,
                )
            except Exception as exc:
                root.after(
                    0,
                    lambda exc=exc: messagebox.showerror("ReSplat failed", str(exc)),
                )
                root.after(0, lambda exc=exc: append_log_direct(f"\nERROR: {exc}"))
            else:
                root.after(
                    0,
                    lambda result_dir=result_dir: messagebox.showinfo(
                        "Step 2 complete", f"ReSplat results:\n{result_dir}"
                    ),
                )
            finally:
                root.after(0, lambda: set_running(None))

        threading.Thread(target=worker, daemon=True).start()

    form = tk.Frame(root, padx=12, pady=12)
    form.pack(fill="x")

    def add_row(parent: tk.Widget, row: int, label: str, widget: tk.Widget, browse: Callable[[], None] | None = None) -> None:
        tk.Label(parent, text=label, anchor="w", width=18).grid(row=row, column=0, sticky="w", pady=4)
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        if browse:
            tk.Button(parent, text="Browse", command=browse).grid(row=row, column=2, padx=(8, 0), pady=4)

    form.columnconfigure(1, weight=1)

    step1_frame = tk.LabelFrame(
        form, text="Step 1: COLMAP Scene Preparation", padx=8, pady=8
    )
    step1_frame.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 10))
    step1_frame.columnconfigure(1, weight=1)

    add_row(step1_frame, 0, "Image folder", tk.Entry(step1_frame, textvariable=image_dir_var), browse_images)
    add_row(step1_frame, 1, "Output root", tk.Entry(step1_frame, textvariable=output_root_var), browse_output)
    add_row(step1_frame, 2, "Scene name", tk.Entry(step1_frame, textvariable=scene_name_var))
    add_row(step1_frame, 3, "COLMAP", tk.Entry(step1_frame, textvariable=colmap_var), browse_colmap)

    camera_menu = create_described_dropdown(
        step1_frame,
        camera_model_var,
        CAMERA_MODELS,
        CAMERA_MODEL_DESCRIPTIONS,
    )
    add_row(step1_frame, 4, "Raw camera model", camera_menu)
    add_row(step1_frame, 5, "Max image size", tk.Entry(step1_frame, textvariable=max_image_size_var))

    options = tk.Frame(step1_frame)
    tk.Checkbutton(options, text="Shared camera calibration", variable=single_camera_var).pack(side="left")
    tk.Checkbutton(options, text="Overwrite existing scene", variable=overwrite_var).pack(
        side="left", padx=(18, 0)
    )
    tk.Checkbutton(
        options, text="Run Step 2 after Step 1", variable=run_step2_after_step1_var
    ).pack(side="left", padx=(18, 0))
    options.grid(row=6, column=1, sticky="w", pady=4)

    run_step1_button = tk.Button(
        step1_frame, text="Run Step 1: COLMAP", command=start_step1
    )
    run_step1_button.grid(row=7, column=1, sticky="w", pady=(8, 4))

    step2_frame = tk.LabelFrame(
        form, text="Step 2: ReSplat Inference", padx=8, pady=8
    )
    step2_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 4))
    step2_frame.columnconfigure(1, weight=1)

    add_row(
        step2_frame,
        0,
        "Prepared scene",
        tk.Entry(step2_frame, textvariable=step2_scene_path_var),
        browse_prepared_scene,
    )
    add_row(
        step2_frame,
        1,
        "Output root",
        tk.Entry(step2_frame, textvariable=resplat_output_root_var),
        browse_resplat_output,
    )
    tk.Label(step2_frame, text="Model preset", anchor="w", width=18).grid(
        row=2, column=0, sticky="w", pady=3
    )
    model_preset_menu = create_described_dropdown(
        step2_frame,
        model_preset_var,
        MODEL_PRESETS,
        MODEL_PRESET_DESCRIPTIONS,
        indicator_text=lambda value: "" if model_checkpoint_exists(value) else "\u2b07",
        status_text=lambda value: (
            f"Downloaded: {model_checkpoint_path(value)}"
            if model_checkpoint_exists(value)
            else (
                "Not downloaded yet. This model will be downloaded into "
                f"{model_checkpoint_path(value).parent} when Step 2 starts."
            )
        ),
    )
    model_preset_menu.grid(
        row=2, column=1, sticky="w", pady=3
    )
    tk.Label(step2_frame, text="Device", anchor="w", width=18).grid(
        row=3, column=0, sticky="w", pady=3
    )
    tk.Entry(step2_frame, textvariable=device_var).grid(
        row=3, column=1, sticky="ew", pady=3
    )
    tk.Label(step2_frame, text="Render chunk size", anchor="w", width=18).grid(
        row=4, column=0, sticky="w", pady=3
    )
    tk.Entry(step2_frame, textvariable=render_chunk_size_var).grid(
        row=4, column=1, sticky="ew", pady=3
    )

    resplat_options = tk.Frame(step2_frame)
    tk.Checkbutton(
        resplat_options, text="Rendered images", variable=save_resplat_images_var
    ).pack(side="left")
    tk.Checkbutton(resplat_options, text="PLY", variable=save_ply_var).pack(
        side="left", padx=(14, 0)
    )
    tk.Checkbutton(resplat_options, text="Video", variable=save_video_var).pack(
        side="left", padx=(14, 0)
    )
    tk.Checkbutton(resplat_options, text="Depth", variable=save_depth_var).pack(
        side="left", padx=(14, 0)
    )
    tk.Checkbutton(resplat_options, text="Metrics", variable=eval_var).pack(
        side="left", padx=(14, 0)
    )
    resplat_options.grid(row=5, column=1, sticky="w", pady=3)

    run_step2_button = tk.Button(
        step2_frame, text="Run Step 2: ReSplat", command=start_step2
    )
    run_step2_button.grid(row=6, column=1, sticky="w", pady=(8, 4))

    log_box = scrolledtext.ScrolledText(root, state="disabled", height=22)
    log_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    append_log(
        "Step 1 creates a ReSplat-compatible COLMAP scene from raw images."
    )
    append_log(
        "Step 2 can run on any existing prepared scene folder with images/ and sparse/0/."
    )

    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.image_dir is None:
        if args.run_resplat and args.resplat_scene_path is not None:
            scene_dir = args.resplat_scene_path.resolve()
            validate_resplat_scene(scene_dir)
            scene_name = args.scene_name or sanitize_scene_name(scene_dir.name)
            resplat_cfg = ReSplatRunConfig(
                enabled=True,
                model_preset=args.model_preset,
                output_root=args.resplat_output_root,
                device=args.device,
                save_images=not args.no_save_resplat_images,
                save_ply=not args.no_save_ply,
                save_video=args.save_video,
                save_depth=args.save_depth,
                no_eval=not args.eval,
                render_chunk_size=args.render_chunk_size,
            )
            run_resplat_inference(scene_dir, scene_name, resplat_cfg)
            return 0
        if args.no_gui:
            raise SystemExit(
                "--image_dir is required unless --run_resplat and "
                "--resplat_scene_path are set"
            )
        launch_gui()
        return 0

    scene_name = args.scene_name or sanitize_scene_name(args.image_dir.name)
    cfg = PrepareConfig(
        image_dir=args.image_dir,
        output_root=args.output_root,
        scene_name=scene_name,
        colmap=args.colmap,
        camera_model=args.camera_model,
        single_camera=not args.per_image_camera,
        max_image_size=args.max_image_size,
        overwrite=args.overwrite,
    )
    scene_dir = prepare_scene(cfg)
    if args.run_resplat:
        resplat_cfg = ReSplatRunConfig(
            enabled=True,
            model_preset=args.model_preset,
            output_root=args.resplat_output_root,
            device=args.device,
            save_images=not args.no_save_resplat_images,
            save_ply=not args.no_save_ply,
            save_video=args.save_video,
            save_depth=args.save_depth,
            no_eval=not args.eval,
            render_chunk_size=args.render_chunk_size,
        )
        run_resplat_inference(scene_dir, cfg.scene_name, resplat_cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
