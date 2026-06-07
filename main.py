"""
CV ExamGuard - Sistem Deteksi Kecurangan Ujian
Main Entry Point

Penggunaan:
    python main.py                          # Deteksi real-time (webcam)
    python main.py --mode download          # Download dataset
    python main.py --mode label             # Auto-labeling dataset
    python main.py --mode train             # Training model
    python main.py --mode all               # Download + Label + Train + Deteksi
    python main.py --source video.mp4       # Deteksi dari file video
    python main.py --source 1               # Kamera index 1
    python main.py --mode train --epochs 100 --model yolov8s.pt
"""

import sys
import os
import argparse
import logging
import yaml
from pathlib import Path

# Tambahkan root ke sys.path
sys.path.insert(0, str(Path(__file__).parent))


def setup_logging(level: str = "INFO"):
    """Setup logging ke konsol dan file."""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    from datetime import datetime
    log_file = log_dir / f"examguard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    date_fmt = "%H:%M:%S"

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(fmt, date_fmt))
    console_handler.setLevel(getattr(logging, level, logging.INFO))

    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(fmt, date_fmt))
    file_handler.setLevel(logging.DEBUG)

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # Matikan log bawaan yg terlalu verbose
    logging.getLogger("ultralytics").setLevel(logging.WARNING)
    logging.getLogger("mediapipe").setLevel(logging.WARNING)

    return str(log_file)


def print_banner():
    """Print banner aplikasi."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          CV EXAMGUARD - SISTEM DETEKSI KECURANGAN UJIAN      ║
╠══════════════════════════════════════════════════════════════╣
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)


def check_requirements():
    """Cek apakah semua library yang diperlukan sudah terinstall."""
    required = {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "ultralytics": "ultralytics",
        "numpy": "numpy",
        "yaml": "PyYAML",
    }

    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"\n⚠  LIBRARY BELUM TERINSTALL:")
        for pkg in missing:
            print(f"   pip install {pkg}")
        print(f"\n   Atau jalankan: pip install -r requirements.txt\n")
        return False
    return True


def mode_download(args):
    """Mode: Download dataset."""
    from training.download_datasets import download_all

    config = {}
    if Path("config.yaml").exists():
        with open("config.yaml", 'r') as f:
            config = yaml.safe_load(f) or {}

    api_key = (args.roboflow_key or
               config.get("dataset", {}).get("roboflow_api_key", ""))

    download_all(api_key=api_key, skip_roboflow=args.skip_roboflow)


def mode_label(args):
    """Mode: Auto-labeling dataset."""
    from training.auto_labeler import run_auto_labeling

    run_auto_labeling(
        datasets_dir=args.datasets_dir,
        output_dir=args.labeled_dir,
        model=args.label_model,
        confidence=args.label_confidence,
        visualize=not args.no_visualize
    )


def mode_train(args):
    """Mode: Training model."""
    from training.auto_trainer import run_training

    # Tentukan dataset
    dataset_yaml = args.dataset
    if dataset_yaml is None:
        labeled_yaml = Path(args.labeled_dir) / "dataset.yaml"
        if labeled_yaml.exists():
            dataset_yaml = str(labeled_yaml)

    result = run_training(
        dataset_yaml=dataset_yaml,
        model_name=args.model,
        output_model_path=args.output_model,
        epochs=args.epochs,
        batch_size=args.batch,
        image_size=args.imgsz,
        patience=args.patience,
        device=args.device,
        workers=args.workers,
        resume=args.resume
    )

    if result:
        print(f"\n✓ Training selesai! Model: {result}")
        return result
    else:
        print(f"\n✗ Training gagal")
        return None


def mode_detect(args):
    """Mode: Deteksi real-time."""
    from src.detector import ExamCheatDetector

    # Source video
    source = args.source
    if source.isdigit():
        source = int(source)

    detector = ExamCheatDetector(config_path=args.config)
    stats = detector.run(
        source=source,
        save_video=args.save_video,
        display=not args.headless
    )
    return stats


def mode_all(args):
    """Mode: Full pipeline - download + label + train + detect."""
    print("\n🚀 MENJALANKAN PIPELINE LENGKAP...")
    print("   Tahap 1/4: Download Dataset")
    mode_download(args)

    print("\n   Tahap 2/4: Auto-Labeling")
    mode_label(args)

    print("\n   Tahap 3/4: Training Model")
    model_path = mode_train(args)

    print("\n   Tahap 4/4: Deteksi Real-time")
    if model_path:
        print(f"   Model custom siap: {model_path}")
    mode_detect(args)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="CV ExamGuard - Sistem Deteksi Kecurangan Ujian",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh Penggunaan:
  python main.py                              Deteksi realtime dari webcam
  python main.py --mode download              Download semua dataset
  python main.py --mode label                 Auto-label dataset
  python main.py --mode train                 Training custom model
  python main.py --mode all                   Pipeline lengkap
  python main.py --source video.mp4           Deteksi dari file video
  python main.py --mode train --epochs 100    Training 100 epoch
  python main.py --mode train --model yolov8s.pt  Model lebih akurat
        """
    )

    # Mode
    parser.add_argument(
        "--mode", "-m",
        choices=["detect", "download", "label", "train", "all"],
        default="detect",
        help="Mode operasi (default: detect)"
    )

    # Source
    parser.add_argument(
        "--source", "-s",
        default="0",
        help="Sumber video: 0=webcam, atau path file video (default: 0)"
    )

    # Config
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Path file konfigurasi (default: config.yaml)"
    )

    # Download options
    dl_group = parser.add_argument_group("Opsi Download")
    dl_group.add_argument("--roboflow-key", default="",
                          help="Roboflow API key untuk dataset lebih banyak")
    dl_group.add_argument("--skip-roboflow", action="store_true",
                          help="Skip download dari Roboflow")

    # Labeling options
    lbl_group = parser.add_argument_group("Opsi Labeling")
    lbl_group.add_argument("--datasets-dir", default="datasets",
                           help="Direktori dataset mentah (default: datasets)")
    lbl_group.add_argument("--labeled-dir", default="datasets/labeled",
                           help="Direktori output labeling (default: datasets/labeled)")
    lbl_group.add_argument("--label-model", default="yolov8n.pt",
                           help="Model untuk auto-labeling (default: yolov8n.pt)")
    lbl_group.add_argument("--label-confidence", type=float, default=0.4,
                           help="Confidence threshold labeling (default: 0.4)")
    lbl_group.add_argument("--no-visualize", action="store_true",
                           help="Skip membuat visualisasi label")

    # Training options
    trn_group = parser.add_argument_group("Opsi Training")
    trn_group.add_argument("--dataset", default=None,
                           help="Path ke dataset.yaml untuk training")
    trn_group.add_argument("--model", default="yolov8n.pt",
                           choices=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt",
                                    "yolov8l.pt", "yolov8x.pt"],
                           help="Model base YOLOv8 (default: yolov8n.pt)")
    trn_group.add_argument("--output-model", default="models/exam_detector.pt",
                           help="Path output model (default: models/exam_detector.pt)")
    trn_group.add_argument("--epochs", type=int, default=50,
                           help="Jumlah epoch training (default: 50)")
    trn_group.add_argument("--batch", type=int, default=16,
                           help="Batch size (default: 16)")
    trn_group.add_argument("--imgsz", type=int, default=640,
                           help="Ukuran gambar input (default: 640)")
    trn_group.add_argument("--patience", type=int, default=15,
                           help="Early stopping patience (default: 15)")
    trn_group.add_argument("--device", default="auto",
                           choices=["auto", "cpu", "cuda", "mps"],
                           help="Device training (default: auto)")
    trn_group.add_argument("--workers", type=int, default=0,
                           help="Dataloader workers (default: 0)")
    trn_group.add_argument("--resume", action="store_true",
                           help="Lanjutkan training dari checkpoint")

    # Detection options
    det_group = parser.add_argument_group("Opsi Deteksi")
    det_group.add_argument("--save-video", action="store_true", default=True,
                           help="Simpan video output (default: True)")
    det_group.add_argument("--no-save-video", dest="save_video",
                           action="store_false",
                           help="Jangan simpan video output")
    det_group.add_argument("--headless", action="store_true",
                           help="Jalankan tanpa display (untuk server)")

    # Logging
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Level logging (default: INFO)")

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup
    print_banner()
    log_file = setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info(f"Log file: {log_file}")

    # Cek requirements
    if not check_requirements():
        print("\nInstall requirements terlebih dahulu:")
        print("  pip install -r requirements.txt\n")
        sys.exit(1)

    # Buat direktori yang diperlukan
    for d in ["datasets", "models", "output", "logs", "training/runs"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Jalankan mode yang dipilih
    mode_map = {
        "detect":   mode_detect,
        "download": mode_download,
        "label":    mode_label,
        "train":    mode_train,
        "all":      mode_all,
    }

    mode_fn = mode_map.get(args.mode, mode_detect)

    try:
        logger.info(f"Menjalankan mode: {args.mode.upper()}")
        mode_fn(args)
    except KeyboardInterrupt:
        print("\n\n⚠  Dihentikan oleh pengguna")
    except Exception as e:
        logger.error(f"Error tidak terduga: {e}", exc_info=True)
        print(f"\n✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
