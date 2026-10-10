# 🎓 CV ExamGuard (Application Core)

> Direktori ini berisi source code inti, script peluncur, model, dan pipeline training dari **CV ExamGuard**.  
> Untuk dokumentasi lengkap, arsitektur sistem, dan tolok ukur performa, silakan baca **[README Utama](../README.md)** di root repositori.

---

## ⚡ Quick Start

### 1. Prasyarat
- **Python**: 3.9, 3.10, atau **3.11** (Sangat disarankan 3.11)
- **Webcam**: Kamera laptop atau USB webcam

### 2. Jalankan Otomatis (Windows)
Cukup klik dua kali:
- **`JALANKAN.bat`** — Menu interaktif untuk instalasi, training, dan peluncuran deteksi.

### 3. Jalankan via Terminal / CLI
```bash
# Buat Virtual Environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux/Mac

# Install Dependensi
pip install -r requirements.txt

# Verifikasi Kesiapan Sistem
python tests/quick_test.py

# Mulai Deteksi Real-Time (Webcam)
python main.py
```

---

## 🎮 Mode Perintah (`main.py`)

| Perintah | Deskripsi |
| :--- | :--- |
| `python main.py` | Menjalankan deteksi real-time dengan webcam default (`source: 0`) |
| `python main.py --source video.mp4` | Menganalisis file rekaman video |
| `python main.py --mode download` | Mengunduh dataset publik & membuat dataset sintetis |
| `python main.py --mode label` | Auto-labeling gambar unannotated ke format YOLO |
| `python main.py --mode train` | Melatih custom model YOLOv8 |
| `python main.py --mode all` | Menjalankan seluruh pipeline: Download → Label → Train → Detect |
| `python add_custom_dataset.py` | Memvalidasi dan memasukkan custom dataset lokal ke sistem |

---

## ⌨️ Kontrol Tombol saat Monitoring

- **`Q`** : Keluar dari sesi & simpan laporan otomatis (HTML/Video/Screenshot)
- **`R`** : Reset analitik dan riwayat event sesi
- **`S`** : Ambil screenshot manual seketika
- **`H`** : Tampilkan / sembunyikan 468 titik Face Mesh
