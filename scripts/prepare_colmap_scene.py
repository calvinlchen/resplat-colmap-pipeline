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


def run_command(command: list[str], log: Callable[[str], None]) -> None:
    log("\n$ " + " ".join(f'"{part}"' if " " in part else part for part in command))
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdout is not None
    for line in process.stdout:
        log(line.rstrip())

    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(f"Command failed with exit code {return_code}: {command[0]}")


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
    return Path.cwd() / "datasets" / "colmap-custom"


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
        default="SIMPLE_RADIAL",
        choices=[
            "SIMPLE_PINHOLE",
            "PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "OPENCV",
            "OPENCV_FISHEYE",
        ],
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
    return parser.parse_args(argv)


def launch_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, scrolledtext

    root = tk.Tk()
    root.title("Prepare ReSplat COLMAP Scene")
    root.geometry("860x620")

    image_dir_var = tk.StringVar()
    output_root_var = tk.StringVar(value=str(default_output_root()))
    scene_name_var = tk.StringVar()
    colmap_var = tk.StringVar(value=shutil.which("colmap") or "colmap")
    camera_model_var = tk.StringVar(value="SIMPLE_RADIAL")
    single_camera_var = tk.BooleanVar(value=True)
    max_image_size_var = tk.StringVar(value="2000")
    overwrite_var = tk.BooleanVar(value=False)

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

    def set_running(running: bool) -> None:
        start_button.configure(state="disabled" if running else "normal")

    def start() -> None:
        try:
            max_image_size = int(max_image_size_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Max image size must be an integer.")
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

        set_running(True)
        append_log("Starting COLMAP preparation...")

        def worker() -> None:
            try:
                scene_dir = prepare_scene(cfg, append_log)
            except Exception as exc:
                root.after(
                    0,
                    lambda exc=exc: messagebox.showerror("Preparation failed", str(exc)),
                )
                root.after(0, lambda exc=exc: append_log_direct(f"\nERROR: {exc}"))
            else:
                root.after(
                    0,
                    lambda scene_dir=scene_dir: messagebox.showinfo(
                        "Scene prepared", f"Ready for ReSplat:\n{scene_dir}"
                    ),
                )
            finally:
                root.after(0, lambda: set_running(False))

        threading.Thread(target=worker, daemon=True).start()

    form = tk.Frame(root, padx=12, pady=12)
    form.pack(fill="x")

    def add_row(row: int, label: str, widget: tk.Widget, browse: Callable[[], None] | None = None) -> None:
        tk.Label(form, text=label, anchor="w", width=18).grid(row=row, column=0, sticky="w", pady=4)
        widget.grid(row=row, column=1, sticky="ew", pady=4)
        if browse:
            tk.Button(form, text="Browse", command=browse).grid(row=row, column=2, padx=(8, 0), pady=4)

    form.columnconfigure(1, weight=1)

    add_row(0, "Image folder", tk.Entry(form, textvariable=image_dir_var), browse_images)
    add_row(1, "Output root", tk.Entry(form, textvariable=output_root_var), browse_output)
    add_row(2, "Scene name", tk.Entry(form, textvariable=scene_name_var))
    add_row(3, "COLMAP", tk.Entry(form, textvariable=colmap_var), browse_colmap)

    camera_menu = tk.OptionMenu(
        form,
        camera_model_var,
        "SIMPLE_RADIAL",
        "RADIAL",
        "OPENCV",
        "SIMPLE_PINHOLE",
        "PINHOLE",
        "OPENCV_FISHEYE",
    )
    add_row(4, "Raw camera model", camera_menu)
    add_row(5, "Max image size", tk.Entry(form, textvariable=max_image_size_var))

    options = tk.Frame(form)
    tk.Checkbutton(options, text="Shared camera calibration", variable=single_camera_var).pack(side="left")
    tk.Checkbutton(options, text="Overwrite existing scene", variable=overwrite_var).pack(
        side="left", padx=(18, 0)
    )
    options.grid(row=6, column=1, sticky="w", pady=4)

    start_button = tk.Button(form, text="Prepare Scene", command=start)
    start_button.grid(row=7, column=1, sticky="w", pady=(8, 4))

    log_box = scrolledtext.ScrolledText(root, state="disabled", height=22)
    log_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    append_log(
        "Select a raw image folder, choose an output root, then click Prepare Scene."
    )
    append_log(
        "The final scene will contain images/ and sparse/0/ for scripts/infer_colmap.py."
    )

    root.mainloop()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if args.image_dir is None:
        if args.no_gui:
            raise SystemExit("--image_dir is required when --no_gui is set")
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
    prepare_scene(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
