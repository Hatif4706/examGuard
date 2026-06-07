"""
Helper Script — Tambah Custom Dataset ke CV ExamGuard
Jalankan: python add_custom_dataset.py
"""

import os
import sys
import shutil
import random
import cv2
from pathlib import Path

# ─────────────── KONFIGURASI ────────────────────────────────────────────────
CUSTOM_DIR   = Path("datasets/custom")
SPLIT_RATIO  = (0.70, 0.20, 0.10)   # train / val / test

CLASS_NAMES  = {
    0: "person",
    1: "phone",
    2: "laptop",
    3: "book",
    4: "suspicious",
}

# ─────────────── HELPER ──────────────────────────────────────────────────────
def colored(text, code): return f"\033[{code}m{text}\033[0m"
G  = lambda t: colored(t, "32")
Y  = lambda t: colored(t, "33")
R  = lambda t: colored(t, "31")
C  = lambda t: colored(t, "36")
B  = lambda t: colored(t, "\033[1")


def check_label_file(label_path: Path) -> tuple:
    """Validasi satu file label YOLO. Return (valid, errors)."""
    errors = []
    try:
        lines = label_path.read_text().strip().split("\n")
        for i, line in enumerate(lines):
            if not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) != 5:
                errors.append(f"  Baris {i+1}: harus 5 kolom, dapat {len(parts)}")
                continue
            cls_id = int(parts[0])
            if cls_id not in CLASS_NAMES:
                errors.append(f"  Baris {i+1}: class_id {cls_id} tidak valid (0–4)")
            vals = [float(x) for x in parts[1:]]
            for j, v in enumerate(vals):
                if not (0.0 <= v <= 1.0):
                    errors.append(f"  Baris {i+1}: nilai {v} harus antara 0–1")
    except Exception as e:
        errors.append(f"  Error baca file: {e}")
    return len(errors) == 0, errors


def validate_dataset(source_dir: Path) -> tuple:
    """Validasi seluruh dataset. Return (n_ok, n_error)."""
    imgs   = list(source_dir.rglob("*.jpg")) + \
             list(source_dir.rglob("*.jpeg")) + \
             list(source_dir.rglob("*.png"))
    n_ok   = 0
    n_err  = 0
    n_nolb = 0

    for img in imgs:
        lbl = img.parent.parent / "labels" / (img.stem + ".txt")
        if not lbl.exists():
            # Coba folder yang sama
            lbl2 = img.parent / (img.stem + ".txt")
            if lbl2.exists():
                lbl = lbl2
            else:
                n_nolb += 1
                continue
        ok, errs = check_label_file(lbl)
        if ok:
            n_ok  += 1
        else:
            n_err += 1
            print(f"  {R('✗')} {img.name}: {errs[0]}")

    return n_ok, n_err, n_nolb


def split_and_copy(source_dir: Path, dest_dir: Path) -> int:
    """Bagi dataset ke train/val/test dan copy ke dest_dir."""
    imgs = (list(source_dir.rglob("*.jpg"))  +
            list(source_dir.rglob("*.jpeg")) +
            list(source_dir.rglob("*.png")))
    random.shuffle(imgs)

    n     = len(imgs)
    n_tr  = int(n * SPLIT_RATIO[0])
    n_val = int(n * SPLIT_RATIO[1])

    splits = {
        "train": imgs[:n_tr],
        "val":   imgs[n_tr:n_tr + n_val],
        "test":  imgs[n_tr + n_val:],
    }

    for split_name, split_imgs in splits.items():
        (dest_dir / split_name / "images").mkdir(parents=True, exist_ok=True)
        (dest_dir / split_name / "labels").mkdir(parents=True, exist_ok=True)

    copied = 0
    for split_name, split_imgs in splits.items():
        for img in split_imgs:
            # Cari label
            lbl = img.parent.parent / "labels" / (img.stem + ".txt")
            if not lbl.exists():
                lbl = img.parent / (img.stem + ".txt")
            if not lbl.exists():
                continue

            # Copy/convert image → JPG
            dest_img = dest_dir / split_name / "images" / (img.stem + ".jpg")
            src_img  = cv2.imread(str(img))
            if src_img is None:
                continue
            cv2.imwrite(str(dest_img), src_img, [cv2.IMWRITE_JPEG_QUALITY, 95])

            # Copy label
            dest_lbl = dest_dir / split_name / "labels" / (img.stem + ".txt")
            shutil.copy2(lbl, dest_lbl)
            copied += 1

    return copied


def write_yaml(dest_dir: Path):
    """Buat dataset.yaml untuk custom dataset."""
    yaml_content = f"""# CV ExamGuard — Custom Dataset
# Dibuat oleh: add_custom_dataset.py

path: {dest_dir.resolve()}
train: train/images
val: val/images
test: test/images

nc: 5
names: ['person', 'phone', 'laptop', 'book', 'suspicious']
"""
    (dest_dir / "dataset.yaml").write_text(yaml_content)


# ─────────────── MAIN ────────────────────────────────────────────────────────
def main():
    print()
    print("=" * 58)
    print(f"  {C('CV EXAMGUARD — TAMBAH CUSTOM DATASET')}")
    print("=" * 58)
    print()

    print(f"{B('Kelas yang tersedia:')}")
    for cid, name in CLASS_NAMES.items():
        print(f"  {cid} = {name}")
    print()

    # ── Pilih mode ─────────────────────────────────────────────────────────
    print(f"{B('Pilih mode:')}")
    print("  1. Dataset sudah terstruktur (punya folder train/val/test)")
    print("  2. Dataset flat (semua gambar + label dalam 1 folder)")
    print("  3. Hanya gambar (akan di-auto-label dulu)")
    print("  4. Validasi dataset yang sudah ada di datasets/custom/")
    print()

    choice = input("Pilihan (1/2/3/4): ").strip()

    if choice == "1":
        _mode_structured()
    elif choice == "2":
        _mode_flat()
    elif choice == "3":
        _mode_images_only()
    elif choice == "4":
        _mode_validate()
    else:
        print(f"{R('Pilihan tidak valid.')}")
        sys.exit(1)


def _mode_structured():
    """Dataset sudah ada train/val/test."""
    print()
    src = input("Path folder dataset Anda (contoh: C:\\dataset\\exam): ").strip().strip('"')
    src_path = Path(src)

    if not src_path.exists():
        print(f"{R('✗ Folder tidak ditemukan:')} {src_path}")
        sys.exit(1)

    # Validasi
    print(f"\n  Memvalidasi dataset dari {src_path}...")
    n_ok, n_err, n_nolb = validate_dataset(src_path)
    print(f"  ✓ Valid : {n_ok} gambar")
    if n_err:   print(f"  {R('✗')} Error  : {n_err} label bermasalah")
    if n_nolb:  print(f"  {Y('⚠')} No label: {n_nolb} gambar tanpa label (dilewati)")

    if n_ok == 0:
        print(f"\n{R('Tidak ada gambar valid. Periksa struktur folder Anda.')}")
        sys.exit(1)

    # Copy ke datasets/custom/
    print(f"\n  Meng-copy {n_ok} gambar ke datasets/custom/ ...")
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test", "valid"]:
        img_dir = src_path / split / "images"
        lbl_dir = src_path / split / "labels"
        if not img_dir.exists():
            continue
        out_split = "val" if split == "valid" else split
        (CUSTOM_DIR / out_split / "images").mkdir(parents=True, exist_ok=True)
        (CUSTOM_DIR / out_split / "labels").mkdir(parents=True, exist_ok=True)

        for ext in ["*.jpg", "*.jpeg", "*.png"]:
            for img in img_dir.glob(ext):
                lbl = lbl_dir / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                dest_img = CUSTOM_DIR / out_split / "images" / (img.stem + ".jpg")
                _img = cv2.imread(str(img))
                if _img is not None:
                    cv2.imwrite(str(dest_img), _img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                shutil.copy2(lbl, CUSTOM_DIR / out_split / "labels" / (img.stem + ".txt"))

    write_yaml(CUSTOM_DIR)
    _print_success()


def _mode_flat():
    """Semua gambar dan label dalam 1 folder."""
    print()
    src = input("Path folder gambar + label Anda: ").strip().strip('"')
    src_path = Path(src)

    if not src_path.exists():
        print(f"{R('✗ Folder tidak ditemukan:')} {src_path}")
        sys.exit(1)

    print(f"\n  Memvalidasi {src_path}...")
    n_ok, n_err, n_nolb = validate_dataset(src_path)
    print(f"  ✓ Valid : {n_ok} gambar")
    if n_err:  print(f"  {R('✗')} Error : {n_err} label bermasalah")

    if n_ok == 0:
        print(f"\n{R('Tidak ada gambar valid.')}")
        sys.exit(1)

    print(f"\n  Split {n_ok} gambar (70% train / 20% val / 10% test)...")
    CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
    copied = split_and_copy(src_path, CUSTOM_DIR)

    write_yaml(CUSTOM_DIR)
    print(f"  {G('✓')} {copied} gambar berhasil di-split dan dicopy")
    _print_success()


def _mode_images_only():
    """Hanya gambar — perlu auto-label dulu."""
    print()
    src = input("Path folder gambar (JPG/PNG): ").strip().strip('"')
    src_path = Path(src)

    if not src_path.exists():
        print(f"{R('✗ Folder tidak ditemukan:')} {src_path}")
        sys.exit(1)

    imgs = list(src_path.glob("*.jpg")) + list(src_path.glob("*.png"))
    if not imgs:
        print(f"{R('✗ Tidak ada gambar ditemukan')}")
        sys.exit(1)

    print(f"  Ditemukan {len(imgs)} gambar")
    print(f"\n  {Y('Auto-labeling dengan YOLOv8n...')}")

    try:
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")

        COCO_TO_EXAM = {0:0, 67:1, 63:2, 73:3, 65:4, 74:4, 64:4, 66:4}

        label_dir = src_path / "labels_auto"
        label_dir.mkdir(exist_ok=True)

        n_labeled = 0
        for img_f in imgs:
            res = model(str(img_f), verbose=False)[0]
            h, w = res.orig_shape
            lines = []
            for box in res.boxes:
                coco = int(box.cls[0])
                if coco not in COCO_TO_EXAM:
                    continue
                exam_cls = COCO_TO_EXAM[coco]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                cx = ((x1+x2)/2) / w
                cy = ((y1+y2)/2) / h
                bw = (x2-x1) / w
                bh = (y2-y1) / h
                lines.append(f"{exam_cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")

            if lines:
                (label_dir / (img_f.stem + ".txt")).write_text("\n".join(lines))
                n_labeled += 1

        print(f"  {G('✓')} {n_labeled}/{len(imgs)} gambar berhasil dilabel otomatis")

        # Pindah label ke folder images
        for lbl in label_dir.glob("*.txt"):
            shutil.copy2(lbl, src_path / lbl.name)

        print(f"\n  {Y('⚠ PENTING: Periksa hasil auto-label sebelum training!')}")
        print(f"  Buka: labelImg {src_path}")

        confirm = input("\n  Lanjutkan split dan copy ke datasets/custom/? (y/n): ")
        if confirm.lower() != 'y':
            print("  Dibatalkan. Periksa dulu label di labelImg.")
            return

        CUSTOM_DIR.mkdir(parents=True, exist_ok=True)
        copied = split_and_copy(src_path, CUSTOM_DIR)
        write_yaml(CUSTOM_DIR)
        print(f"  {G('✓')} {copied} gambar dicopy ke datasets/custom/")
        _print_success()

    except ImportError:
        print(f"{R('✗ ultralytics tidak terinstall:')} pip install ultralytics")
        sys.exit(1)


def _mode_validate():
    """Validasi dataset yang sudah ada."""
    if not CUSTOM_DIR.exists():
        print(f"\n{Y('datasets/custom/ belum ada. Buat dulu dengan mode 1/2/3.')}")
        return

    imgs = list(CUSTOM_DIR.rglob("*.jpg")) + list(CUSTOM_DIR.rglob("*.png"))
    print(f"\n  Validasi {CUSTOM_DIR}...")
    print(f"  Total gambar: {len(imgs)}")

    n_ok, n_err, n_nolb = validate_dataset(CUSTOM_DIR)

    print()
    print(f"  {G('✓')} Valid     : {n_ok}")
    print(f"  {R('✗')} Bermasalah: {n_err}")
    print(f"  {Y('⚠')} Tanpa label: {n_nolb}")

    # Count per split
    for split in ["train", "val", "test"]:
        n = len(list((CUSTOM_DIR / split / "images").glob("*.jpg"))) if \
            (CUSTOM_DIR / split / "images").exists() else 0
        print(f"  {split:6}: {n} gambar")

    if n_err == 0:
        print(f"\n  {G('Dataset siap dipakai!')}")
    else:
        print(f"\n  {Y('Perbaiki label bermasalah sebelum training.')}")


def _print_success():
    print()
    print("=" * 58)
    print(f"  {G('✓ Custom dataset berhasil ditambahkan!')}")
    print("=" * 58)
    print()
    print("  Dataset disimpan di: datasets/custom/")
    print()
    print("  Langkah selanjutnya:")
    print("  1. (Opsional) Periksa label: labelImg datasets/custom/train/images")
    print("  2. Training + custom dataset:")
    print()
    print("     python main.py --mode train --epochs 100")
    print()
    print("  Dataset custom akan OTOMATIS digabung dengan dataset yang")
    print("  sudah ada (COCO, Synthetic, dll) — tidak ada yang dihapus.")
    print()


if __name__ == "__main__":
    main()
