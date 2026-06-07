"""
Auto Trainer - Training model YOLOv8 untuk deteksi kecurangan ujian
Versi Enhanced: lebih banyak augmentasi, hyperparameter yang lebih baik
"""

import os
import sys
import shutil
import logging
import yaml
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    G = lambda t: f"{Fore.GREEN}{t}{Style.RESET_ALL}"
    Y = lambda t: f"{Fore.YELLOW}{t}{Style.RESET_ALL}"
    R = lambda t: f"{Fore.RED}{t}{Style.RESET_ALL}"
    C = lambda t: f"{Fore.CYAN}{t}{Style.RESET_ALL}"
    B = lambda t: f"\033[1m{t}\033[0m"
except:
    G = Y = R = C = B = lambda t: t


DATASETS_DIR  = Path("datasets")
MODELS_DIR    = Path("models")
TRAIN_OUT_DIR = Path("runs/train")


def find_best_dataset() -> Optional[str]:
    """Temukan dataset terbaik untuk training (urutan prioritas)."""
    priority = [
        "exam_unified",      # Dataset gabungan terbaik
        "synthetic_exam",    # Synthetic fallback
        "coco128",
        "coco8",
    ]
    for name in priority:
        d = DATASETS_DIR / name
        for yaml_name in ["dataset.yaml", "data.yaml"]:
            yaml_path = d / yaml_name
            if yaml_path.exists():
                return str(yaml_path)

    # Scan manual
    for yaml_f in DATASETS_DIR.rglob("*.yaml"):
        return str(yaml_f)
    return None


def patch_dataset_yaml(yaml_path: str) -> str:
    """Perbaiki path di dataset.yaml agar absolute."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    base = Path(yaml_path).parent.resolve()

    for key in ["train", "val", "test"]:
        if key in data:
            p = Path(data[key])
            if not p.is_absolute():
                data[key] = str(base / p)

    if "path" not in data:
        data["path"] = str(base)

    patched = yaml_path.replace(".yaml", "_patched.yaml")
    with open(patched, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    return patched


def run_training(
    dataset_yaml:   str   = None,
    model_name:     str   = "yolov8n.pt",
    epochs:         int   = 80,
    batch_size:     int   = 16,
    image_size:     int   = 640,
    device:         str   = "auto",
    output_name:    str   = "exam_detector",
    augment:        bool  = True,
) -> Optional[str]:
    """
    Training YOLOv8 dengan konfigurasi optimal untuk deteksi kecurangan ujian.
    
    Returns:
        Path ke model terbaik (best.pt) atau None jika gagal
    """
    print(f"\n{'='*60}")
    print(f"{B(C('  CV EXAMGUARD - MODEL TRAINING'))}")
    print(f"{'='*60}\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Cari dataset ──────────────────────────────────────────
    if not dataset_yaml:
        dataset_yaml = find_best_dataset()
    if not dataset_yaml or not Path(dataset_yaml).exists():
        print(f"  {R('✗')} Dataset tidak ditemukan!")
        print(f"  Jalankan dulu: python main.py --mode download")
        return None

    print(f"  {G('✓')} Dataset: {dataset_yaml}")

    # Patch path agar absolute
    dataset_yaml = patch_dataset_yaml(dataset_yaml)

    # ── Import Ultralytics ────────────────────────────────────
    try:
        from ultralytics import YOLO
    except ImportError:
        print(f"  {R('✗')} ultralytics tidak terinstall!")
        print(f"  Jalankan: pip install ultralytics")
        return None

    # ── Resolve device ────────────────────────────────────────
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
                print(f"  {G('✓')} GPU CUDA terdeteksi - training akan lebih cepat!")
            else:
                device = "cpu"
                print(f"  {Y('⚠')} CPU mode - training akan lebih lambat")
        except:
            device = "cpu"

    # ── Auto-adjust batch size untuk CPU ─────────────────────
    if device == "cpu" and batch_size > 8:
        batch_size = 8
        print(f"  {Y('⚠')} CPU mode: batch_size diturunkan ke {batch_size}")

    # ── Hyperparameters untuk exam detection ─────────────────
    train_args = {
        "data":          dataset_yaml,
        "model":         model_name,
        "epochs":        epochs,
        "imgsz":         image_size,
        "batch":         batch_size,
        "device":        device,
        "workers":       0,              # Windows: selalu 0
        "project":       str(TRAIN_OUT_DIR),
        "name":          output_name,
        "exist_ok":      True,
        "patience":      20,             # Early stopping
        "save_period":   10,             # Save setiap N epoch
        "val":           True,
        "plots":         True,

        # Augmentasi untuk meningkatkan robustness deteksi HP dari berbagai angle
        "augment":       augment,
        "flipud":        0.0,            # Jangan flip vertikal (orang terbalik aneh)
        "fliplr":        0.5,            # Flip horizontal (cermin) - HP kiri/kanan
        "mosaic":        1.0,            # Mosaic augment
        "mixup":         0.1,            # Mixup augment
        "degrees":       10.0,           # Rotasi -10° s/d +10°
        "translate":     0.1,            # Translasi
        "scale":         0.5,            # Scale augment (zoom)
        "shear":         2.0,            # Shear ringan
        "perspective":   0.0001,         # Perspektif
        "hsv_h":         0.015,          # HSV hue
        "hsv_s":         0.7,            # HSV saturation
        "hsv_v":         0.4,            # HSV value

        # Optimizer
        "optimizer":     "AdamW",
        "lr0":           0.001,
        "lrf":           0.01,
        "momentum":      0.937,
        "weight_decay":  0.0005,
        "warmup_epochs": 3,

        # Confidence / NMS
        "conf":          0.25,
        "iou":           0.45,
        "max_det":       20,
    }

    print(f"\n  {B('Konfigurasi Training:')}")
    print(f"    Model      : {model_name}")
    print(f"    Epochs     : {epochs}")
    print(f"    Batch size : {batch_size}")
    print(f"    Image size : {image_size}x{image_size}")
    print(f"    Device     : {device}")
    print(f"    Augmentasi : {'Ya' if augment else 'Tidak'}")
    print()

    try:
        model = YOLO(model_name)
        start = time.time()
        results = model.train(**train_args)
        elapsed = time.time() - start

        # ── Simpan best.pt ke models/ ─────────────────────────
        best_pt = TRAIN_OUT_DIR / output_name / "weights" / "best.pt"
        if best_pt.exists():
            dest_pt = MODELS_DIR / "exam_detector.pt"
            shutil.copy2(best_pt, dest_pt)
            print(f"\n  {G('✓')} Model terbaik disimpan ke: {dest_pt}")

            # Update config.yaml
            _update_config(str(dest_pt))
            print(f"  {G('✓')} config.yaml diperbarui")

            mins = int(elapsed // 60)
            secs = int(elapsed % 60)
            print(f"\n  {G('✓')} Training selesai dalam {mins}m {secs}s!")
            return str(dest_pt)
        else:
            print(f"  {R('✗')} best.pt tidak ditemukan di {best_pt}")
            return None

    except KeyboardInterrupt:
        print(f"\n  {Y('⚠')} Training dihentikan oleh pengguna")
        # Simpan last checkpoint
        last_pt = TRAIN_OUT_DIR / output_name / "weights" / "last.pt"
        if last_pt.exists():
            dest_pt = MODELS_DIR / "exam_detector_last.pt"
            shutil.copy2(last_pt, dest_pt)
            print(f"  {Y('⚠')} Checkpoint terakhir disimpan ke: {dest_pt}")
            return str(dest_pt)
        return None

    except Exception as e:
        print(f"\n  {R('✗')} Training error: {e}")
        logger.exception("Training error")
        return None


def _update_config(model_path: str):
    """Update config.yaml untuk memakai model custom."""
    config_path = Path("config.yaml")
    if not config_path.exists():
        return
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            content = f.read()
        import re
        content = re.sub(r'use_custom:\s*false', 'use_custom: true', content)
        content = re.sub(
            r'custom_model:\s*"[^"]*"',
            f'custom_model: "{model_path}"',
            content
        )
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        logger.warning(f"Gagal update config.yaml: {e}")


def evaluate_model(model_path: str, dataset_yaml: str = None) -> dict:
    """Evaluasi model pada test set."""
    print(f"\n{B('=== EVALUASI MODEL ===')}")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)

        if not dataset_yaml:
            dataset_yaml = find_best_dataset()
        if not dataset_yaml:
            print(f"  {R('✗')} Dataset tidak ditemukan")
            return {}

        dataset_yaml = patch_dataset_yaml(dataset_yaml)
        metrics = model.val(data=dataset_yaml, imgsz=640, conf=0.25, iou=0.45)

        print(f"\n  {G('Hasil Evaluasi:')}")
        print(f"    mAP50    : {metrics.box.map50:.3f}")
        print(f"    mAP50-95 : {metrics.box.map:.3f}")
        print(f"    Precision: {metrics.box.mp:.3f}")
        print(f"    Recall   : {metrics.box.mr:.3f}")
        return {
            "map50": metrics.box.map50,
            "map":   metrics.box.map,
            "precision": metrics.box.mp,
            "recall":    metrics.box.mr,
        }
    except Exception as e:
        print(f"  {R('✗')} Evaluasi gagal: {e}")
        return {}
