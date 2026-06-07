"""
Auto Labeler - Labelling otomatis menggunakan YOLOv8 pretrained
Mendeteksi objek dalam gambar dan membuat label format YOLO otomatis
"""

import cv2
import os
import json
import shutil
import logging
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)

try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init()
    def GREEN(t): return f"{Fore.GREEN}{t}{Style.RESET_ALL}"
    def YELLOW(t): return f"{Fore.YELLOW}{t}{Style.RESET_ALL}"
    def RED(t): return f"{Fore.RED}{t}{Style.RESET_ALL}"
    def CYAN(t): return f"{Fore.CYAN}{t}{Style.RESET_ALL}"
    def BOLD(t): return f"\033[1m{t}\033[0m"
except:
    def GREEN(t): return t
    def YELLOW(t): return t
    def RED(t): return t
    def CYAN(t): return t
    def BOLD(t): return t


# Mapping COCO class → ExamGuard class index
# ExamGuard classes: 0=person, 1=phone, 2=laptop, 3=book, 4=suspicious
COCO_TO_EXAM_CLASS = {
    0:  0,   # person      → person
    67: 1,   # cell phone  → phone
    63: 2,   # laptop      → laptop
    64: 2,   # mouse       → laptop (suspicious peripheral)
    66: 2,   # keyboard    → laptop
    73: 3,   # book        → book
    84: 3,   # book        → book (duplicate COCO)
    65: 4,   # remote      → suspicious
    76: 4,   # scissors    → suspicious
    74: 4,   # clock       → suspicious
}

EXAM_CLASS_NAMES = ["person", "phone", "laptop", "book", "suspicious"]


class AutoLabeler:
    """
    Auto-labeling menggunakan YOLOv8 pretrained model (COCO weights).
    Menghasilkan label format YOLO untuk training custom model.
    """

    def __init__(self,
                 model_name: str = "yolov8n.pt",
                 confidence: float = 0.4,
                 iou: float = 0.45,
                 device: str = "auto",
                 output_classes: List[str] = None):

        self.confidence = confidence
        self.iou = iou
        self.output_classes = output_classes or EXAM_CLASS_NAMES
        self.model = None
        self.device = self._resolve_device(device)
        self._load_model(model_name)

    def _resolve_device(self, device: str) -> str:
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except:
            pass
        return "cpu"

    def _load_model(self, model_name: str):
        """Load YOLOv8 pretrained model."""
        try:
            from ultralytics import YOLO
            print(f"  ⚙ Loading model {model_name} untuk auto-labeling...")
            self.model = YOLO(model_name)
            print(f"  {GREEN('✓')} Model loaded: {model_name} (COCO pretrained)")
        except Exception as e:
            logger.error(f"Gagal load model: {e}")
            raise

    def label_image(self, image_path: str,
                     min_box_area: float = 0.001) -> List[Dict]:
        """
        Label satu gambar dan return list deteksi dalam format YOLO.
        
        Args:
            image_path: Path ke file gambar
            min_box_area: Area minimum bounding box (rasio dari frame)
            
        Returns:
            List dict: [{class_id, cx, cy, w, h, confidence}]
        """
        if self.model is None:
            return []

        try:
            img = cv2.imread(image_path)
            if img is None:
                return []

            h, w = img.shape[:2]
            frame_area = h * w

            results = self.model(
                image_path,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                verbose=False
            )

            labels = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    coco_class_id = int(box.cls[0])

                    # Hanya proses kelas yang relevan
                    if coco_class_id not in COCO_TO_EXAM_CLASS:
                        continue

                    exam_class_id = COCO_TO_EXAM_CLASS[coco_class_id]
                    conf = float(box.conf[0])

                    # Convert ke YOLO format (normalized)
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    box_w = x2 - x1
                    box_h = y2 - y1
                    box_area = box_w * box_h

                    # Filter kotak terlalu kecil
                    if box_area / frame_area < min_box_area:
                        continue

                    cx = (x1 + x2) / 2 / w
                    cy = (y1 + y2) / 2 / h
                    nw = box_w / w
                    nh = box_h / h

                    # Pastikan nilai dalam rentang valid
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    nw = max(0.001, min(1.0, nw))
                    nh = max(0.001, min(1.0, nh))

                    labels.append({
                        "class_id": exam_class_id,
                        "class_name": self.output_classes[exam_class_id],
                        "cx": cx, "cy": cy,
                        "w": nw, "h": nh,
                        "confidence": conf,
                        "coco_class": coco_class_id
                    })

            return labels

        except Exception as e:
            logger.debug(f"Error labeling {image_path}: {e}")
            return []

    def label_directory(self,
                         image_dir: str,
                         output_label_dir: str,
                         extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
                         visualize: bool = False,
                         viz_dir: str = "") -> Dict:
        """
        Auto-label semua gambar dalam direktori.
        
        Args:
            image_dir: Direktori gambar input
            output_label_dir: Direktori output label (.txt)
            extensions: Ekstensi gambar yang diproses
            visualize: Buat gambar visualisasi label
            viz_dir: Direktori visualisasi output
            
        Returns:
            Dict statistik labeling
        """
        image_dir = Path(image_dir)
        output_label_dir = Path(output_label_dir)
        output_label_dir.mkdir(parents=True, exist_ok=True)

        # Kumpulkan semua gambar
        image_files = []
        for ext in extensions:
            image_files.extend(image_dir.glob(f"*{ext}"))
            image_files.extend(image_dir.glob(f"*{ext.upper()}"))

        if not image_files:
            print(f"  {YELLOW('!')} Tidak ada gambar ditemukan di: {image_dir}")
            return {"total": 0, "labeled": 0, "skipped": 0}

        stats = {
            "total": len(image_files),
            "labeled": 0,
            "skipped": 0,
            "total_labels": 0,
            "class_counts": {name: 0 for name in self.output_classes}
        }

        print(f"  📁 {len(image_files)} gambar ditemukan di: {image_dir.name}")

        for img_path in tqdm(image_files, desc=f"  Labeling {image_dir.name}",
                              ncols=70, leave=False):
            # Deteksi & label
            labels = self.label_image(str(img_path))

            # Tulis label ke file .txt
            label_filename = img_path.stem + ".txt"
            label_path = output_label_dir / label_filename

            if labels:
                with open(label_path, 'w') as f:
                    for label in labels:
                        line = (f"{label['class_id']} "
                                f"{label['cx']:.6f} "
                                f"{label['cy']:.6f} "
                                f"{label['w']:.6f} "
                                f"{label['h']:.6f}\n")
                        f.write(line)
                        stats["class_counts"][label["class_name"]] += 1
                        stats["total_labels"] += 1
                stats["labeled"] += 1
            else:
                # Tulis file kosong (gambar tanpa objek relevan = background)
                label_path.touch()
                stats["skipped"] += 1

            # Visualisasi (opsional)
            if visualize and labels and viz_dir:
                self._visualize_labels(str(img_path), labels, viz_dir)

        return stats

    def _visualize_labels(self, image_path: str, labels: List[Dict], viz_dir: str):
        """Buat gambar visualisasi dengan bounding box."""
        try:
            viz_dir = Path(viz_dir)
            viz_dir.mkdir(parents=True, exist_ok=True)

            img = cv2.imread(image_path)
            if img is None:
                return

            h, w = img.shape[:2]
            colors = [
                (0, 255, 0),    # person - hijau
                (0, 0, 255),    # phone - merah
                (255, 0, 0),    # laptop - biru
                (0, 165, 255),  # book - oranye
                (128, 0, 128),  # suspicious - ungu
            ]

            for label in labels:
                cx, cy = label["cx"] * w, label["cy"] * h
                bw, bh = label["w"] * w, label["h"] * h
                x1 = int(cx - bw / 2)
                y1 = int(cy - bh / 2)
                x2 = int(cx + bw / 2)
                y2 = int(cy + bh / 2)

                color = colors[label["class_id"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                text = f"{label['class_name']} {label['confidence']:.2f}"
                cv2.putText(img, text, (x1, max(y1 - 5, 15)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            out_path = viz_dir / Path(image_path).name
            cv2.imwrite(str(out_path), img)

        except Exception as e:
            logger.debug(f"Error visualizing {image_path}: {e}")


def collect_images_for_labeling(
        source_dirs: List[str],
        output_dir: str,
        max_per_dir: int = 500,
        img_size: int = 640) -> int:
    """
    Kumpulkan dan resize gambar dari berbagai sumber ke satu direktori.
    
    Args:
        source_dirs: List direktori sumber
        output_dir: Direktori output
        max_per_dir: Maksimum gambar per direktori sumber
        img_size: Ukuran resize target
        
    Returns:
        Jumlah gambar yang dikumpulkan
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for src_dir in source_dirs:
        src_dir = Path(src_dir)
        if not src_dir.exists():
            continue

        # Cari semua gambar secara rekursif
        images = []
        for ext in extensions:
            images.extend(src_dir.rglob(f"*{ext}"))
            images.extend(src_dir.rglob(f"*{ext.upper()}"))

        # Batasi jumlah
        images = images[:max_per_dir]

        print(f"  Mengumpulkan dari {src_dir.name}: {len(images)} gambar")

        for img_path in images:
            try:
                img = cv2.imread(str(img_path))
                if img is None or img.size == 0:
                    continue

                # Resize sambil menjaga aspect ratio
                h, w = img.shape[:2]
                if max(h, w) > img_size:
                    scale = img_size / max(h, w)
                    new_w, new_h = int(w * scale), int(h * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # Nama file unik
                out_name = f"{src_dir.name}_{total:05d}{img_path.suffix}"
                out_path = output_dir / out_name
                cv2.imwrite(str(out_path), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
                total += 1

            except Exception as e:
                logger.debug(f"Error processing {img_path}: {e}")

    return total


def run_auto_labeling(datasets_dir: str = "datasets",
                       output_dir: str = "datasets/labeled",
                       model: str = "yolov8n.pt",
                       confidence: float = 0.4,
                       visualize: bool = True) -> Path:
    """
    Jalankan proses auto-labeling pada semua dataset yang tersedia.
    
    Args:
        datasets_dir: Direktori berisi dataset yang didownload
        output_dir: Direktori output dataset yang sudah dilabel
        model: Model YOLOv8 untuk labeling
        confidence: Confidence threshold
        visualize: Buat visualisasi label
        
    Returns:
        Path dataset output yang siap training
    """
    print("\n" + "=" * 65)
    print(BOLD(CYAN("  🏷️  AUTO LABELER - CV ExamGuard")))
    print("  Labelling Otomatis menggunakan YOLOv8 Pretrained")
    print("=" * 65)

    datasets_dir = Path(datasets_dir)
    output_path = Path(output_dir)

    # Direktori split
    splits = ["train", "valid", "test"]
    for split in splits:
        (output_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_path / "labels" / split).mkdir(parents=True, exist_ok=True)

    # Inisialisasi labeler
    labeler = AutoLabeler(
        model_name=model,
        confidence=confidence,
        device="auto"
    )

    # Kumpulkan semua gambar dari dataset yang ada
    print(f"\n  📂 Mencari gambar di: {datasets_dir}")

    all_image_dirs = []
    for item in datasets_dir.rglob("images"):
        if item.is_dir():
            for split_dir in item.iterdir():
                if split_dir.is_dir():
                    all_image_dirs.append(split_dir)

    # Jika tidak ada dataset terstruktur, cari gambar langsung
    if not all_image_dirs:
        for ext in [".jpg", ".jpeg", ".png"]:
            if any(datasets_dir.rglob(f"*{ext}")):
                all_image_dirs.append(datasets_dir)
                break

    if not all_image_dirs:
        print(f"  {YELLOW('!')} Tidak ada gambar ditemukan di {datasets_dir}")
        print(f"     Pastikan sudah menjalankan: python main.py --mode download")

        # Buat dataset sintetis minimal jika belum ada
        from training.download_datasets import create_synthetic_dataset
        synth_dir = create_synthetic_dataset()
        all_image_dirs.append(synth_dir / "images" / "train")

    print(f"  Ditemukan {len(all_image_dirs)} direktori gambar")

    # Proses setiap direktori
    overall_stats = {
        "total_images": 0,
        "labeled_images": 0,
        "total_labels": 0,
        "class_counts": {name: 0 for name in EXAM_CLASS_NAMES}
    }

    for i, img_dir in enumerate(all_image_dirs):
        if not img_dir.exists():
            continue

        # Tentukan split berdasarkan nama direktori
        dir_name = img_dir.name.lower()
        if "valid" in dir_name or "val" in dir_name:
            split = "valid"
        elif "test" in dir_name:
            split = "test"
        else:
            split = "train"

        # Output direktori untuk split ini
        out_img_dir = output_path / "images" / split
        out_lbl_dir = output_path / "labels" / split
        viz_dir = output_path / "visualizations" / split if visualize else ""

        print(f"\n  [{i+1}/{len(all_image_dirs)}] Processing: {img_dir.name} → {split}")

        # Salin gambar ke output direktori
        img_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            img_files.extend(img_dir.glob(f"*{ext}"))
            img_files.extend(img_dir.glob(f"*{ext.upper()}"))

        # Batasi jumlah untuk efisiensi
        max_imgs = 1000
        if len(img_files) > max_imgs:
            import random
            random.shuffle(img_files)
            img_files = img_files[:max_imgs]
            print(f"    (Dibatasi {max_imgs} gambar)")

        # Salin gambar
        for img_file in tqdm(img_files, desc=f"  Copy {split}", ncols=60, leave=False):
            dst = out_img_dir / img_file.name
            if not dst.exists():
                shutil.copy2(str(img_file), str(dst))

        # Auto-label gambar yang disalin
        stats = labeler.label_directory(
            str(out_img_dir),
            str(out_lbl_dir),
            visualize=visualize,
            viz_dir=str(viz_dir) if visualize else ""
        )

        overall_stats["total_images"] += stats["total"]
        overall_stats["labeled_images"] += stats["labeled"]
        overall_stats["total_labels"] += stats.get("total_labels", 0)
        for cls, count in stats.get("class_counts", {}).items():
            if cls in overall_stats["class_counts"]:
                overall_stats["class_counts"][cls] += count

    # Buat dataset.yaml
    yaml_content = f"""# Dataset Auto-Labeled - CV ExamGuard
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

path: {output_path.absolute()}
train: images/train
val: images/valid
test: images/test

# Jumlah kelas
nc: {len(EXAM_CLASS_NAMES)}

# Nama kelas
names: {EXAM_CLASS_NAMES}
"""
    yaml_path = output_path / "dataset.yaml"
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)

    # Simpan statistik labeling
    stats_path = output_path / "labeling_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(overall_stats, f, indent=2)

    # Print ringkasan
    print("\n" + "=" * 65)
    print(BOLD(GREEN("  ✓ AUTO-LABELING SELESAI")))
    print(f"  Total gambar diproses : {overall_stats['total_images']}")
    print(f"  Gambar berhasil dilabel: {overall_stats['labeled_images']}")
    print(f"  Total label dihasilkan: {overall_stats['total_labels']}")
    print(f"\n  Distribusi kelas:")
    for cls, count in overall_stats["class_counts"].items():
        pct = count / max(overall_stats["total_labels"], 1) * 100
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"    {cls:12} [{bar}] {count:4d} ({pct:.1f}%)")
    print(f"\n  Dataset YAML: {yaml_path}")
    print("=" * 65)
    print(f"\n  Selanjutnya jalankan: python main.py --mode train\n")

    return output_path


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets-dir", default="datasets")
    parser.add_argument("--output-dir", default="datasets/labeled")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.4)
    parser.add_argument("--no-visualize", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_auto_labeling(
        datasets_dir=args.datasets_dir,
        output_dir=args.output_dir,
        model=args.model,
        confidence=args.confidence,
        visualize=not args.no_visualize
    )
