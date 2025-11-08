#!/usr/bin/env python3
"""
Entry script for SageMaker training that executes the local Jupyter notebook end-to-end
on the training instance, saving the executed notebook and artifacts to /opt/ml/model.

It also adapts the expected dataset paths in the notebook by creating symlinks
/home/sagemaker-user/imagenet1k/{train,val} -> /opt/ml/input/data/{train,val}
so existing notebook paths continue to work.

Notes:
- This will execute ALL cells in the notebook as-is (plots are rendered off-screen).
- If running on multi-GPU, this script will run once per process. To avoid duplicate
  heavy setup, rank 0 prepares directories/symlinks first; other ranks continue.
- Current notebook code isn’t fully DDP-wrapped, so it may still use a single GPU.
  You can still launch on multi-GPU instances (future work: add DistributedSampler + DDP).
"""

import argparse
import os
import sys
import subprocess
import time
from pathlib import Path


def log(msg: str):
    print(f"[train_from_notebook] {msg}", flush=True)


def pip_install(pkgs):
    # Best-effort install with retries
    for pkg in (pkgs if isinstance(pkgs, (list, tuple)) else [pkgs]):
        for attempt in range(2):
            try:
                log(f"Installing package: {pkg} (attempt {attempt+1})")
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", pkg])
                break
            except subprocess.CalledProcessError as e:
                if attempt == 1:
                    log(f"Failed to install {pkg}: {e}")
                    raise
                time.sleep(2)


def ensure_notebook_deps():
    # Papermill to execute notebooks, nbconvert for HTML export
    try:
        import papermill  # noqa: F401
    except Exception:
        pip_install(["papermill>=2.5.0", "nbconvert>=7.0.0", "jupyter", "ipykernel", "tqdm"])  


def setup_data_symlinks(base_home: Path):
    """Create expected notebook data paths and link them to SageMaker channels."""
    # SageMaker training channels
    ch_train = Path("/opt/ml/input/data/train")
    ch_val = Path("/opt/ml/input/data/val")

    # Notebook expects these paths
    nb_root = base_home / "imagenet1k"
    nb_train = nb_root / "train"
    nb_val = nb_root / "val"

    nb_root.mkdir(parents=True, exist_ok=True)

    def link(src: Path, dst: Path):
        if not src.exists():
            log(f"Channel missing: {src} (ok if you use S3 sync inside the notebook)")
            return
        if dst.is_symlink() or dst.exists():
            # If already correct, skip. If wrong, replace.
            try:
                if dst.resolve() == src.resolve():
                    return
            except Exception:
                pass
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                # non-empty dir — leave it as-is to avoid data loss
                return
        log(f"Linking {dst} -> {src}")
        dst.symlink_to(src, target_is_directory=True)

    # Only rank 0 sets up links to avoid races
    rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", 0)))
    if rank == 0:
        link(ch_train, nb_train)
        link(ch_val, nb_val)


def execute_notebook(input_nb: Path, output_nb: Path, output_html: Path | None = None):
    import papermill as pm

    # Use non-interactive backend for matplotlib
    os.environ.setdefault("MPLBACKEND", "Agg")

    log(f"Executing notebook: {input_nb} -> {output_nb}")
    pm.execute_notebook(
        input_path=str(input_nb),
        output_path=str(output_nb),
        kernel_name="python3",
        parameters={},  # Notebook doesn’t define parameters; executes as-is
        progress_bar=True,
        request_save_on_cell_execute=True,
    )

    if output_html:
        try:
            log(f"Exporting executed notebook to HTML: {output_html}")
            subprocess.check_call([
                sys.executable, "-m", "jupyter", "nbconvert", "--to", "html",
                "--output", str(output_html), str(output_nb)
            ])
        except subprocess.CalledProcessError as e:
            log(f"HTML export failed (continuing): {e}")


def main():
    parser = argparse.ArgumentParser(description="Run notebook on SageMaker training instance")
    parser.add_argument(
        "--notebook",
        type=str,
        default="TSAI_mini_capstone_imagnet1k_resnet50.ipynb",
        help="Notebook filename located in source_dir (/opt/ml/code)",
    )
    parser.add_argument(
        "--output-notebook",
        type=str,
        default="executed_notebook.ipynb",
        help="Filename for the executed notebook saved under /opt/ml/model",
    )
    parser.add_argument(
        "--output-html",
        type=str,
        default="executed_notebook.html",
        help="Filename for the exported HTML saved under /opt/ml/model (optional)",
    )
    args = parser.parse_args()

    code_dir = Path("/opt/ml/code")
    model_dir = Path("/opt/ml/model")
    model_dir.mkdir(parents=True, exist_ok=True)

    input_nb = code_dir / args.notebook
    if not input_nb.exists():
        # Also check working dir in case source_dir layout differs
        alt = Path.cwd() / args.notebook
        if alt.exists():
            input_nb = alt
        else:
            raise FileNotFoundError(f"Notebook not found: {args.notebook}")

    # Ensure deps and paths
    ensure_notebook_deps()
    # Create symlinks so the notebook's hardcoded paths work
    setup_data_symlinks(Path("/home/sagemaker-user"))

    # Execute and save outputs in /opt/ml/model
    output_nb = model_dir / args.output_notebook
    output_html = model_dir / args.output_html if args.output_html else None

    execute_notebook(input_nb, output_nb, output_html)

    log("Notebook execution complete. Artifacts saved to /opt/ml/model")


if __name__ == "__main__":
    main()
