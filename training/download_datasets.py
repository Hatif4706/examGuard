"""
Dataset Downloader - Download dataset lebih lengkap untuk deteksi kecurangan ujian
Menggunakan public URLs tanpa API key (Roboflow, GitHub, COCO subset)
"""

import os
import sys
import logging
import requests
import zipfile
import json
import shutil
import random
import math
from pathlib import Path
from typing import List, Optional
import time
import cv2
import numpy as np

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

DATASETS_DIR = Path("datasets")

# ── Class mapping ExamGuard ──────────────────────────────────────────────────
# 0: person, 1: phone, 2: laptop/tablet, 3: book/paper, 4: suspicious
CLASS_NAMES = {0: "person", 1: "phone", 2: "laptop", 3: "book", 4: "suspicious"}
COCO_MAP = {
    0:  0,   # person → person
    67: 1,   # cell phone → phone
    63: 2,   # laptop → laptop
    77: 2,   # laptop (teddy → reuse as laptop proxy)
    73: 3,   # book → book
    65: 4,   # remote → suspicious
    76: 4,   # scissors → suspicious
    74: 4,   # clock → suspicious (bawa-bawa ke ujian)
    64: 4,   # mouse → suspicious
    66: 4,   # keyboard → suspicious
}

# ── Dataset publik yang bisa didownload tanpa API ────────────────────────────
PUBLIC_DATASETS = [
    {
        "name": "COCO8 Sample",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco8.zip",
        "dest": "coco8",
        "desc": "128 gambar COCO sample (person, phone, laptop, dll) - COCO format"
    },
    {
        "name": "COCO128 Sample",
        "url": "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
        "dest": "coco128",
        "desc": "128 gambar COCO (lebih banyak variasi kelas)"
    },
]

# ── Roboflow public datasets (tanpa API key - via export links) ───────────────
ROBOFLOW_DATASETS = [
    {
        "name": "Phone Detection v2",
        "desc": "Deteksi HP dari berbagai angle termasuk belakang",
        "workspace": "roboflow-100",
        "project": "cell-phones-qqhnf",
        "version": "2",
        "priority": "HIGH"
    },
    {
        "name": "Person Detection",
        "desc": "Deteksi peserta ujian dan banyak orang",
        "workspace": "roboflow-100",
        "project": "people-detection-general",
        "version": "1",
        "priority": "HIGH"
    },
    {
        "name": "Face Detection",
        "desc": "Deteksi wajah menghadap berbagai arah",
        "workspace": "roboflow-100",
        "project": "face-detection-mik1i",
        "version": "18",
        "priority": "HIGH"
    },
]


def download_file(url: str, dest_path: str, desc: str = "") -> bool:
    """Download file dengan progress."""
    try:
        print(f"  ↓ {desc or url[:65]}...")
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        dl = 0
        Path(dest_path).parent.mkdir(parents=True, exist_ok=True)
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    dl += len(chunk)
        mb = dl / 1_000_000
        print(f"    {G('✓')} {mb:.1f} MB berhasil didownload")
        return True
    except Exception as e:
        print(f"    {R('✗')} Gagal: {e}")
        return False


def extract_zip(zip_path: str, dest_dir: str) -> bool:
    """Extract zip file."""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)
        os.remove(zip_path)
        return True
    except Exception as e:
        print(f"    {R('✗')} Extract gagal: {e}")
        return False


def download_public_datasets() -> List[str]:
    """Download dataset publik tanpa API key."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []

    print(f"\n{B('=== DOWNLOAD PUBLIC DATASETS ===')}")
    for ds in PUBLIC_DATASETS:
        dest_dir = DATASETS_DIR / ds["dest"]
        if dest_dir.exists() and any(dest_dir.rglob("*.jpg")):
            print(f"  {G('✓')} {ds['name']} sudah ada, skip.")
            downloaded.append(str(dest_dir))
            continue

        print(f"\n  {C(ds['name'])}: {ds['desc']}")
        zip_path = str(DATASETS_DIR / f"{ds['dest']}.zip")
        if download_file(ds["url"], zip_path, ds["name"]):
            dest_dir.mkdir(parents=True, exist_ok=True)
            if extract_zip(zip_path, str(dest_dir)):
                print(f"    {G('✓')} Extracted ke {dest_dir}")
                downloaded.append(str(dest_dir))
    return downloaded


def download_roboflow_datasets(api_key: str = "") -> List[str]:
    """Download dari Roboflow (perlu API key untuk private, publik bisa tanpa)."""
    if not api_key:
        print(f"\n  {Y('INFO')} Roboflow API key tidak ada, skip Roboflow datasets.")
        print(f"  Dapatkan gratis di: https://roboflow.com")
        return []

    downloaded = []
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)

        for ds in ROBOFLOW_DATASETS:
            dest = DATASETS_DIR / ds["project"]
            if dest.exists():
                print(f"  {G('✓')} {ds['name']} sudah ada, skip.")
                downloaded.append(str(dest))
                continue
            try:
                print(f"  {C('↓')} {ds['name']}...")
                proj = rf.workspace(ds["workspace"]).project(ds["project"])
                version = proj.version(int(ds["version"]))
                dataset = version.download("yolov8", location=str(dest))
                print(f"    {G('✓')} Berhasil download {ds['name']}")
                downloaded.append(str(dest))
            except Exception as e:
                print(f"    {Y('⚠')} {ds['name']} gagal: {e}")
    except ImportError:
        print(f"  {Y('⚠')} roboflow tidak terinstall. Jalankan: pip install roboflow")
    return downloaded


def generate_synthetic_dataset(
    num_train: int = 800,
    num_val:   int = 200,
    num_test:  int = 100,
    img_size:  int = 640
) -> str:
    """
    Buat dataset sintetik yang lebih realistis untuk deteksi kecurangan ujian.
    Termasuk: orang (menghadap berbagai arah), HP, laptop, buku, benda lain.
    """
    print(f"\n{B('=== GENERATING SYNTHETIC DATASET ===')}")
    base = DATASETS_DIR / "synthetic_exam"

    splits = {
        "train": num_train,
        "val":   num_val,
        "test":  num_test,
    }

    for split, count in splits.items():
        img_dir   = base / split / "images"
        label_dir = base / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        print(f"  Generating {count} gambar [{split}]...")
        for idx in range(count):
            img = _gen_exam_scene(img_size, idx, split)
            bboxes = _gen_labels_for_scene(img, img_size, idx)

            cv2.imwrite(str(img_dir / f"exam_{split}_{idx:04d}.jpg"), img)
            with open(label_dir / f"exam_{split}_{idx:04d}.txt", "w") as f:
                for b in bboxes:
                    f.write(f"{b['cls']} {b['cx']:.6f} {b['cy']:.6f} "
                            f"{b['bw']:.6f} {b['bh']:.6f}\n")

        print(f"    {G('✓')} {count} gambar & label dibuat")

    # YAML file
    yaml_content = f"""# Dataset Sintetik - CV ExamGuard
path: {base.resolve()}
train: train/images
val: val/images
test: test/images

nc: 5
names: ['person', 'phone', 'laptop', 'book', 'suspicious']
"""
    yaml_path = base / "dataset.yaml"
    yaml_path.write_text(yaml_content)
    print(f"  {G('✓')} Dataset YAML: {yaml_path}")
    print(f"  {G('✓')} Total: {num_train + num_val + num_test} gambar sintetik")
    return str(base)


def _gen_exam_scene(size: int, idx: int, split: str) -> np.ndarray:
    """Buat scene ujian yang realistis secara visual."""
    rng = random.Random(idx * 1337 + (0 if split == "train" else 7777))

    # Background warna ruang kelas (abu/putih/krem)
    bg_choices = [
        (220, 215, 210), (240, 238, 235), (210, 218, 225),
        (230, 225, 218), (245, 242, 238),
    ]
    bg_color = bg_choices[rng.randint(0, len(bg_choices) - 1)]
    img = np.full((size, size, 3), bg_color, dtype=np.uint8)

    # Tambah noise ringan agar tidak terlalu flat
    noise = np.random.randint(-12, 12, (size, size, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Meja (garis horizontal bawah)
    desk_y = int(size * rng.uniform(0.55, 0.70))
    desk_color = (160, 140, 110)
    cv2.rectangle(img, (0, desk_y), (size, size),
                  tuple(int(c * 0.95) for c in bg_color[::-1])[::-1], -1)
    cv2.line(img, (0, desk_y), (size, desk_y), desk_color, 3)

    # Garis meja (kayu) horizontal
    for i in range(1, 4):
        ly = desk_y + int(i * (size - desk_y) * 0.25)
        cv2.line(img, (0, ly), (size, ly), desk_color, 1)

    return img


def _gen_labels_for_scene(img: np.ndarray, size: int, idx: int):
    """Buat bounding box untuk scene ujian."""
    rng = random.Random(idx * 42)
    bboxes = []
    s = size

    # ── Person (hampir selalu ada) ────────────────────────
    if rng.random() < 0.92:
        # Orang menghadap berbagai arah (frontal, kiri, kanan, belakang)
        desk_y = int(s * rng.uniform(0.55, 0.70))
        pw = int(s * rng.uniform(0.28, 0.42))
        ph = int(desk_y * rng.uniform(0.75, 0.95))
        px = rng.randint(s // 4, 3 * s // 4 - pw)
        py = desk_y - ph

        if py < 0:
            ph += py; py = 0
        if py + ph > s:
            ph = s - py

        _draw_person(img, px, py, pw, ph, rng, idx)
        cx = (px + pw / 2) / s
        cy = (py + ph / 2) / s
        bboxes.append({'cls': 0, 'cx': cx, 'cy': cy,
                       'bw': pw / s, 'bh': ph / s})

    # ── HP (30% kemungkinan) ──────────────────────────────
    if rng.random() < 0.30:
        desk_y = int(s * 0.62)
        # HP bisa landscape/portrait, dari depan/belakang
        if rng.random() < 0.5:    # Portrait
            pw, ph = int(s * rng.uniform(0.07, 0.10)), int(s * rng.uniform(0.14, 0.18))
        else:                      # Landscape
            pw, ph = int(s * rng.uniform(0.14, 0.18)), int(s * rng.uniform(0.07, 0.10))
        px = rng.randint(s // 8, 7 * s // 8 - pw)
        py = rng.randint(desk_y + 5, s - ph - 5)
        _draw_phone(img, px, py, pw, ph, rng)
        bboxes.append({'cls': 1,
                       'cx': (px + pw / 2) / s, 'cy': (py + ph / 2) / s,
                       'bw': pw / s, 'bh': ph / s})

    # ── Laptop/Tablet (20% kemungkinan) ──────────────────
    if rng.random() < 0.20:
        desk_y = int(s * 0.60)
        lw = int(s * rng.uniform(0.22, 0.32))
        lh = int(s * rng.uniform(0.16, 0.22))
        lx = rng.randint(5, s - lw - 5)
        ly = rng.randint(desk_y + 5, s - lh - 5)
        _draw_laptop(img, lx, ly, lw, lh, rng)
        bboxes.append({'cls': 2,
                       'cx': (lx + lw / 2) / s, 'cy': (ly + lh / 2) / s,
                       'bw': lw / s, 'bh': lh / s})

    # ── Buku/Lembar ujian (selalu ada di meja) ────────────
    if rng.random() < 0.80:
        desk_y = int(s * 0.62)
        bw = int(s * rng.uniform(0.20, 0.30))
        bh = int(s * rng.uniform(0.14, 0.20))
        bx = rng.randint(5, s - bw - 5)
        by = rng.randint(desk_y + 3, s - bh - 3)
        _draw_book(img, bx, by, bw, bh, rng)
        bboxes.append({'cls': 3,
                       'cx': (bx + bw / 2) / s, 'cy': (by + bh / 2) / s,
                       'bw': bw / s, 'bh': bh / s})

    # ── Benda mencurigakan (10% kemungkinan) ─────────────
    if rng.random() < 0.10:
        desk_y = int(s * 0.62)
        sw = int(s * rng.uniform(0.06, 0.12))
        sh = int(s * rng.uniform(0.06, 0.12))
        sx = rng.randint(5, s - sw - 5)
        sy = rng.randint(desk_y + 3, s - sh - 3)
        col_susp = (rng.randint(80, 200), rng.randint(80, 200), rng.randint(80, 200))
        cv2.rectangle(img, (sx, sy), (sx + sw, sy + sh), col_susp, -1)
        bboxes.append({'cls': 4,
                       'cx': (sx + sw / 2) / s, 'cy': (sy + sh / 2) / s,
                       'bw': sw / s, 'bh': sh / s})

    return bboxes


def _draw_person(img, px, py, pw, ph, rng, idx):
    """Gambar orang dengan berbagai orientasi (frontal, samping, belakang)."""
    # Pilih orientasi
    orientation = idx % 4   # 0:frontal, 1:kiri, 2:kanan, 3:belakang

    skin_tones = [(180, 140, 100), (220, 180, 150), (140, 100, 70), (200, 165, 120)]
    skin = skin_tones[rng.randint(0, len(skin_tones) - 1)]
    shirt_colors = [(50, 100, 180), (180, 50, 50), (50, 160, 50),
                    (100, 100, 100), (220, 180, 50)]
    shirt = shirt_colors[rng.randint(0, len(shirt_colors) - 1)]

    head_r = int(pw * 0.22)
    head_cx = px + pw // 2
    head_cy = py + head_r + 2

    # Kepala
    if orientation == 3:  # Belakang - kepala tidak terlihat wajah
        cv2.circle(img, (head_cx, head_cy), head_r, skin, -1)
        # Rambut (belakang lebih gelap)
        cv2.circle(img, (head_cx, head_cy), head_r,
                   (30, 20, 10), 3)
    elif orientation == 1:  # Melihat kiri
        cv2.circle(img, (head_cx, head_cy), head_r, skin, -1)
        # Muka sedikit ke kiri
        eye_x = head_cx - int(head_r * 0.4)
        cv2.circle(img, (eye_x, head_cy - 3), 3, (30, 30, 30), -1)
    elif orientation == 2:  # Melihat kanan
        cv2.circle(img, (head_cx, head_cy), head_r, skin, -1)
        eye_x = head_cx + int(head_r * 0.4)
        cv2.circle(img, (eye_x, head_cy - 3), 3, (30, 30, 30), -1)
    else:                   # Frontal
        cv2.circle(img, (head_cx, head_cy), head_r, skin, -1)
        # Mata
        for ex in [head_cx - int(head_r * 0.35), head_cx + int(head_r * 0.35)]:
            cv2.circle(img, (ex, head_cy - 3), 3, (30, 30, 30), -1)
        # Mulut
        cv2.ellipse(img, (head_cx, head_cy + int(head_r * 0.35)),
                    (int(head_r * 0.25), int(head_r * 0.12)),
                    0, 0, 180, (150, 90, 70), 1)

    # Badan
    body_top = head_cy + head_r
    body_h = int(ph * 0.55)
    cv2.rectangle(img, (px + int(pw * 0.1), body_top),
                  (px + int(pw * 0.9), body_top + body_h), shirt, -1)

    # Tangan
    arm_y = body_top + int(body_h * 0.25)
    cv2.line(img, (px + int(pw * 0.1), arm_y),
             (px - int(pw * 0.1), arm_y + int(body_h * 0.4)), skin, int(pw * 0.1))
    cv2.line(img, (px + int(pw * 0.9), arm_y),
             (px + pw + int(pw * 0.1), arm_y + int(body_h * 0.4)), skin, int(pw * 0.1))


def _draw_phone(img, px, py, pw, ph, rng):
    """Gambar HP dari berbagai sisi (depan/belakang)."""
    side = rng.choice(["front", "back", "back", "side"])  # Lebih sering belakang
    if side == "back":
        # Belakang HP - warna solid, kamera kecil
        color = tuple(rng.randint(40, 100) for _ in range(3))
        cv2.rectangle(img, (px, py), (px + pw, py + ph), color, -1)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (80, 80, 80), 2)
        # Kamera belakang
        cam_x = px + int(pw * 0.2)
        cam_y = py + int(ph * 0.15)
        cv2.circle(img, (cam_x, cam_y), int(min(pw, ph) * 0.08), (30, 30, 30), -1)
    elif side == "front":
        # Layar HP menyala
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (20, 20, 30), -1)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), (80, 80, 90), 2)
        # Layar bercahaya
        inner = int(min(pw, ph) * 0.08)
        screen_color = (rng.randint(180, 255), rng.randint(180, 255), rng.randint(180, 255))
        cv2.rectangle(img, (px + inner, py + inner),
                      (px + pw - inner, py + ph - inner), screen_color, -1)
    else:
        # Sisi samping - tipis
        color = tuple(rng.randint(60, 120) for _ in range(3))
        cv2.rectangle(img, (px, py), (px + pw, py + ph), color, -1)


def _draw_laptop(img, lx, ly, lw, lh, rng):
    lid_h = int(lh * 0.65)
    base_h = lh - lid_h
    # Layar
    cv2.rectangle(img, (lx, ly), (lx + lw, ly + lid_h), (30, 35, 45), -1)
    cv2.rectangle(img, (lx + 5, ly + 3), (lx + lw - 5, ly + lid_h - 3),
                  (180, 200, 220), -1)
    # Base
    cv2.rectangle(img, (lx - 5, ly + lid_h), (lx + lw + 5, ly + lh),
                  (80, 85, 90), -1)
    cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (60, 65, 70), 2)


def _draw_book(img, bx, by, bw, bh, rng):
    colors = [(240, 235, 220), (255, 250, 240), (245, 240, 230)]
    c = colors[rng.randint(0, len(colors) - 1)]
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), c, -1)
    cv2.rectangle(img, (bx, by), (bx + bw, by + bh), (160, 140, 120), 2)
    # Garis teks
    for i in range(3, bh - 5, 10):
        lx1 = bx + int(bw * 0.08)
        lx2 = bx + int(bw * (0.5 + rng.random() * 0.4))
        cv2.line(img, (lx1, by + i), (lx2, by + i), (180, 170, 155), 1)


def create_unified_dataset(source_dirs: List[str]) -> Optional[str]:
    """Gabungkan berbagai dataset menjadi satu dataset YOLO."""
    print(f"\n{B('=== MENGGABUNGKAN DATASET ===')}")
    out = DATASETS_DIR / "exam_unified"

    for split in ["train", "val", "test"]:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    total_copied = 0
    for src in source_dirs:
        src_path = Path(src)
        for split in ["train", "valid", "val", "test"]:
            img_dir = src_path / split / "images"
            lbl_dir = src_path / split / "labels"
            if not img_dir.exists():
                continue
            out_split = "val" if split == "valid" else split
            for ext in ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]:
                for img_file in img_dir.glob(ext):
                    lbl_file = lbl_dir / (img_file.stem + ".txt")
                    if not lbl_file.exists():
                        continue
                    dest_name = f"{src_path.name}_{img_file.stem}.jpg"
                    dest_lbl  = f"{src_path.name}_{img_file.stem}.txt"
                    # Convert to JPG if needed
                    import cv2 as _cv2
                    _img = _cv2.imread(str(img_file))
                    if _img is None:
                        continue
                    _cv2.imwrite(str(out / out_split / "images" / dest_name), _img,
                                 [_cv2.IMWRITE_JPEG_QUALITY, 95])
                    shutil.copy2(lbl_file, out / out_split / "labels" / dest_lbl)
                    total_copied += 1

    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(f"""# CV ExamGuard Unified Dataset
path: {out.resolve()}
train: train/images
val: val/images
test: test/images

nc: 5
names: ['person', 'phone', 'laptop', 'book', 'suspicious']
""")

    print(f"  {G('✓')} {total_copied} gambar di-copy ke {out}")
    print(f"  {G('✓')} Dataset YAML: {yaml_path}")
    return str(out)


def run_download(roboflow_api_key: str = "", only_synthetic: bool = False) -> Optional[str]:
    """Main: download semua dataset dan buat unified dataset."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"{B(C('  CV EXAMGUARD - DATASET PIPELINE'))}")
    print(f"{'='*60}")

    collected_dirs = []

    # 1. Public datasets (COCO)
    if not only_synthetic:
        public = download_public_datasets()
        collected_dirs.extend(public)

    # 2. Roboflow (if API key)
    if roboflow_api_key:
        rf = download_roboflow_datasets(roboflow_api_key)
        collected_dirs.extend(rf)

    # 3. Custom dataset milik pengguna (auto-detect)
    custom_dir = DATASETS_DIR / "custom"
    if custom_dir.exists():
        has_images = any(custom_dir.rglob("*.jpg")) or any(custom_dir.rglob("*.png"))
        if has_images:
            print(f"\n  {G('✓')} Custom dataset ditemukan: {custom_dir}")
            collected_dirs.append(str(custom_dir))
        else:
            print(f"  {Y('⚠')} Folder datasets/custom/ ada tapi kosong — skip")
    else:
        print(f"  {Y('ℹ')} Belum ada custom dataset. Baca CUSTOM_DATASET_GUIDE.md untuk menambahkan.")

    # 4. Synthetic (selalu generate)
    synth = generate_synthetic_dataset(
        num_train=1000, num_val=250, num_test=125)
    collected_dirs.append(synth)

    # 5. Unified
    if len(collected_dirs) > 1:
        unified = create_unified_dataset(collected_dirs)
    else:
        unified = collected_dirs[0] if collected_dirs else None

    print(f"\n{G('✓ Download selesai!')}")
    if unified:
        print(f"  Dataset siap di: {unified}")
    return unified


def download_all(api_key: str = "", skip_roboflow: bool = False) -> Optional[str]:
    """
    Wrapper untuk main.py — download semua dataset.
    Args:
        api_key: Roboflow API key (opsional)
        skip_roboflow: Lewati download Roboflow
    """
    rf_key = "" if skip_roboflow else api_key
    return run_download(roboflow_api_key=rf_key, only_synthetic=False)
