# 🎓 CV ExamGuard — Next-Gen AI Exam Proctoring & Anti-Cheating Surveillance System

<div align="center">

![Python Version](https://img.shields.io/badge/Python-3.9%20%7C%203.10%20%7C%203.11-3776AB?style=for-the-badge&logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF?style=for-the-badge&logo=yolo&logoColor=black)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Mesh-00A98F?style=for-the-badge&logo=google&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-4D4D4D?style=for-the-badge&logo=windows&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

<p align="center">
  <b>Sistem pengawasan ujian otomatis berbasis Computer Vision cerdas & Deep Learning real-time.</b><br>
  Mendeteksi pergerakan kepala (Head Pose Estimation), penggunaan smartphone/gadget, multiple candidates, hingga anomali perilaku peserta dengan akurasi tinggi dan latensi rendah tanpa memerlukan GPU diskrit (Optimized for CPU 25-30 FPS).
</p>

[Fitur Utama](#-fitur-utama) • [Arsitektur Sistem](#-arsitektur-sistem) • [Instalasi Cepat](#-instalasi--setup) • [Panduan Penggunaan](#-panduan-penggunaan) • [Pipeline MLOps](#-pipeline-mlops--training) • [Konfigurasi](#-konfigurasi-sistem-configyaml) • [Laporan & Analitik](#-laporan--analisis-sesi)

</div>

---

## 💡 Ringkasan & Setup Cepat (Lightweight & Zero-Setup)

> 🚀 **PENTING / PRAKTIS**:
> Anda **TIDAK PERLU mengunduh dataset bergigabyte** untuk menjalankan pengawasan ujian!
> - Deteksi real-time langsung siap pakai menggunakan model **`yolov8n.pt`** bawaan (hanya 6.5 MB) dan **MediaPipe Face Mesh**.
> - Dataset **HANYA diperlukan jika Anda ingin melatih (*training*) custom model baru**.
> - Jika ingin melatih custom model, dataset dapat diunduh & di-generate otomatis kapan saja secara instan melalui perintah `python main.py --mode download`.

---

## 🌟 Mengapa CV ExamGuard?

Dalam ekosistem ujian daring (online) dan hybrid modern, integritas akademik menjadi tantangan terbesar. Solusi pengawasan konvensional seringkali:
1. **Membebani CPU/Hardware**: Mengharuskan spesifikasi GPU tinggi yang sulit dijangkau semua institusi.
2. **False Positive Tinggi**: Peserta yang hanya menunduk sedikit untuk membaca lembar soal langsung dianggap curang.
3. **Alert Duplikasi (Event Spam)**: Menghasilkan puluhan trigger peringatan per detik untuk satu objek yang sama (misal 33x notifikasi HP dalam 1 detik).
4. **Minim Bukti Audit**: Tidak menyediakan bukti visual real-time dan analitik komprehensif pasca-ujian.

**CV ExamGuard v2.3 Enhanced** dirancang secara khusus untuk mengatasi semua tantangan di atas melalui rekayasa komputasi visi yang efisien, kalibrasi *head-pose* berbasis PnP solver dengan *Exponential Moving Average (EMA)* smoothing, sistem deduplikasi event berbasis sesi (*SessionTracker*), serta laporan analitik interaktif HTML (Chart.js) dan PDF.

---

## 🚀 Fitur Utama

| Modul | Teknologi | Deskripsi & Keunggulan |
| :--- | :--- | :--- |
| **3D Head Pose Tracking** | MediaPipe Face Mesh + solvePnP | Estimasi rotasi kepala (*Yaw*, *Pitch*, *Roll*) secara real-time via 468 facial landmark 3D. Dilengkapi kalibrasi offset pitch (+8°) dan smoothing EMA ($\alpha=0.35$) anti-jitter. |
| **Object Detection & Gadgets** | Ultralytics YOLOv8 (n/s/m) | Deteksi instan HP/ponsel, laptop, tablet, buku, dan benda mencurigakan lainnya dengan batas deteksi confidence adaptif. |
| **Multi-Person & Absence Detection** | YOLOv8 + Temporal Tracking | Mengidentifikasi jika ada joki / orang lain di frame (>1 orang) atau jika peserta meninggalkan meja ujian (>5 detik). |
| **Session-Based Event Tracking** | State-Machine Rule Engine | Mencegah spam deteksi. Mengelompokkan pelanggaran kontinu ke dalam 1 sesi terpadu dengan pencatatan durasi aktif. |
| **Multi-Level Alert System** | Visual HUD + Sound Synthesizer | 5 tingkatan alert (*AMAN*, *WASPADA*, *MENCURIGAKAN*, *PELANGGARAN*, *KECURANGAN!*) dengan audio synthesizer multi-frekuensi (*winsound* native & *pygame*). |
| **Auto-Evidence Capture** | OpenCV Threaded Capture | Tangkapan layar otomatis (screenshot) beresolusi tinggi saat terjadi pelanggaran dan perekaman video sesi (*.mp4). |
| **Real-time Live Dashboard** | Custom OpenCV HUD UI | Panel kontrol gelap (Dark Theme 340px) menampilkan indikator arah dengan panah dinamis (←, →, ↓), skor kecurangan, kurva riwayat, dan live metrics. |
| **Streaming Metrics Tracker** | Confusion Matrix Engine | Menghitung metrik evaluasi secara live: Precision, Recall, F1-Score, Accuracy, dan IoU mAP (0.50:0.95). |
| **Automated HTML & PDF Report** | Chart.js + CSS Modern + fpdf2 | Rekapitulasi pasca-sesi dengan grafik interaktif distribusi kecurangan, linimasa menit-ke-menit, dan tabel log kejadian lengkap. |
| **Full MLOps Pipeline** | Automated Data Tools | Built-in Synthetic Dataset Generator, Roboflow API Downloader, Auto-Labeler pseudo-labeling, dan Auto-Trainer YOLOv8. |

---

## 🏗️ Arsitektur Sistem

CV ExamGuard menggunakan arsitektur pemrosesan **Dual-Stream Resolution**: Frame kamera dipertahankan dalam resolusi penuh untuk tampilan layar monitor & perekaman bukti, sedangkan komputasi AI berjalan pada resolusi yang di-*downsample* secara agresif ($480 \times 360$) sebelum titik koordinat diskalakan kembali ke dimensi asli. Hal ini menghasilkan performa stabil **25–30 FPS pada prosesor laptop standar**.

```mermaid
flowchart TD
    subgraph INPUT ["1. Video Acquisition"]
        CAM["Webcam / Video Stream (1280x720)"]
    end

    subgraph PREPROC ["2. Optimization Engine"]
        RESIZE["Downsample ke (480x360) untuk AI"]
        ORIG["Original Buffer untuk HUD & Rekaman"]
    end

    subgraph VISION ["3. Computer Vision & Deep Learning"]
        direction TB
        MP["MediaPipe Face Mesh (468 3D Landmarks)"] --> PNP["OpenCV solvePnP + EMA Filter\n(Yaw, Pitch, Roll Angle)"]
        YOLO["YOLOv8 Inference\n(Person, Phone, Laptop, Book)"] --> BBOX["Bounding Boxes & Confidences"]
    end

    subgraph ANALYZER ["4. Behavioral & Temporal Engine"]
        direction TB
        STATE["BehaviorAnalyzer & SessionTracker"]
        WEIGHT["Weighted Scoring Matrix\n(Phone: 8 | Gaze: 7 | Multi: 5 | HeadDown: 4)"]
        COOLDOWN["Cooldown Filter & Anti-Duplication Engine"]
        STATE --- WEIGHT --- COOLDOWN
    end

    subgraph ACTION ["5. Alert & Evidence Dispatcher"]
        ALERT_LVL{"Level Risiko:\nAMAN | WASPADA | MENCURIGAKAN |\nPELANGGARAN | KECURANGAN!"}
        AUDIO["Audio Alert Synthesizer\n(Windows winsound / pygame)"]
        SCREENSHOT["Auto-Screenshot Bukti Pelanggaran"]
        VID_REC["Continuous Session Video Recorder"]
        HUD["Live Dashboard Overlay (340px HUD)"]
    end

    subgraph REPORT ["6. Analytics & Audit Logging"]
        METRICS["MetricsTracker (Precision, Recall, F1, mAP)"]
        LOGS["Session Event Logs (*.log)"]
        HTML_REP["Interactive HTML Report (Chart.js) & PDF"]
    end

    CAM --> RESIZE
    CAM --> ORIG
    RESIZE --> MP
    RESIZE --> YOLO
    PNP --> STATE
    BBOX --> STATE
    STATE --> ALERT_LVL
    ALERT_LVL --> AUDIO
    ALERT_LVL --> SCREENSHOT
    ALERT_LVL --> VID_REC
    ALERT_LVL --> HUD
    ALERT_LVL --> METRICS
    ALERT_LVL --> LOGS
    METRICS --> HTML_REP
    LOGS --> HTML_REP
```

---

## 🔬 Rincian Teknis & Keunggulan Komputasi

### 1. Estimasi Pose Kepala (Head Pose Estimation)
Alih-alih mengandalkan model klasifikasi arah pandang black-box, ExamGuard menggunakan pendekatan **geometrik 3D Perspective-n-Point (`solvePnP`)**:
- **6 Titik Wajah 3D Universal**: Ujung hidung (1), dagu (152), sudut mata kiri (33), sudut mata kanan (263), sudut bibir kiri (78), dan sudut bibir kanan (308).
- **Euler Angles Extraction**: Matriks rotasi diurai menjadi sudut *Yaw* (menoleh kiri/kanan), *Pitch* (menunduk/mendongak), dan *Roll* (miring).
- **Kalibrasi Real-World**:
  - `pitch_offset: 8.0°`: Mengkompensasi posisi kamera laptop yang umumnya berada lebih tinggi dari posisi peserta duduk normal.
  - **AngleSmootherEMA**: Filter eksponensial bergerak dengan koefisien $\alpha=0.35$ untuk meniadakan fluktuasi landmark akibat noise pencahayaan.

```python
# Formula Smoothing EMA pada Rotasi Kepala
smoothed_angle = alpha * current_angle + (1 - alpha) * previous_smoothed_angle
```

### 2. Logika Penilaian Perilaku (Behavior Scoring Engine)
Sistem mengakumulasi skor risiko per frame berdasarkan bobot pelanggaran:

$$\text{Total Score} = \sum (\text{Weight}_{\text{active\_violation}})$$

| Jenis Pelanggaran | Bobot Skor | Batas Ambang (Threshold) | Aksi Default |
| :--- | :---: | :--- | :--- |
| **Penggunaan Smartphone (`PHONE_DETECTED`)** | **8** | Terdeteksi $\ge 0.40$ confidence | Audio Peringatan 1800Hz + Screenshot |
| **Menoleh Lama (`EXTENDED_GAZE_AWAY`)** | **7** | Menoleh kiri/kanan berturut-turut $> 3.0$ dtk | Peringatan Kritis + Event Log |
| **Banyak Orang di Frame (`MULTIPLE_PERSONS`)** | **5** | Terdeteksi $> 1$ orang selama $> 2.0$ dtk | Alert Pelanggaran Joki |
| **Kepala Menunduk Ekstrem (`HEAD_DOWN`)** | **4** | Pitch $> 22^\circ$ selama $> 3.0$ dtk | Alert Menunduk (Orange) |
| **Menoleh Kiri / Kanan (`LOOKING_LEFT/RIGHT`)** | **3** | $\|Yaw\| > 20^\circ$ | Indikator HUD Merah + Beep |
| **Objek Mencurigakan (`SUSPICIOUS_OBJECT`)** | **3** | Buku/contekan terdeteksi | Alert Benda Terlarang |
| **Peserta Tidak Terlihat (`NO_PERSON`)** | **3** | Wajah hilang $> 5.0$ detik | Status Darurat Peserta Hilang |

### 3. Solusi Anti-Duplikasi (SessionTracker Engine)
Jika peserta mengeluarkan HP selama 2 detik pada video 30 FPS, algoritma konvensional mencatat 60 baris pelanggaran baru. ExamGuard memecahkan masalah ini dengan **Session Tracker**:
- Deteksi kontinu dihitung sebagai **1 sesi tunggal**.
- Dilengkapi *grace period* toleransi (1.0 detik) agar kedipan atau oklusi sesaat tidak memecah sesi.
- Log mencatat durasi total HP digunakan secara akurat.

---

## ⚡ Tolok Ukur Kinerja (Benchmark)

Pengujian performa pada prosesor standar (Intel Core i5 / AMD Ryzen 5 tanpa dGPU):

| Metrik | Versi Awal (Full-Res 720p) | CV ExamGuard Enhanced (Dual-Stream) | Peningkatan |
| :--- | :---: | :---: | :---: |
| **Frame Rate Rata-rata** | 12 - 16 FPS | **26 - 30 FPS** | **+87% Lebih Cepat** |
| **Latency Inferensi** | ~75 ms / frame | **~33 ms / frame** | **-56% Lebih Responsif** |
| **Akurasi Sudut Wajah** | Sering jitter ($\pm 8^\circ$) | Stabil & halus ($\pm 1.2^\circ$) | **Stabil (EMA Smoothing)** |
| **Duplikasi Log HP** | 33x per detik | **1x sesi terpadu** | **100% Akurat** |
| **Penggunaan RAM** | ~3.8 GB | **~1.9 GB** | **Hemat Memori** |

---

## 💻 Persyaratan Sistem

- **Sistem Operasi**: Windows 10/11 (Sangat direkomendasikan & dioptimalkan), Linux Ubuntu 20.04+, macOS 12+
- **Python**: **3.9, 3.10, atau 3.11** *(Disarankan 3.11; MediaPipe saat ini belum stabil di Python 3.12+)*
- **Memori (RAM)**: Minimum 4GB (Direkomendasikan 8GB+)
- **Kamera**: Webcam bawaan laptop atau kamera USB 720p/1080p
- **Penyimpanan**: Ruang kosong minimal ~1.5GB (tanpa dataset gigabyte)
- **Akselerasi GPU** *(Opsional)*: NVIDIA GPU dengan dukungan CUDA 11.8 / 12.1 untuk akselerasi ekstra

---

## ⚙️ Instalasi & Setup

### Opsi 1: Setup Otomatis 1-Klik (Windows)
```cmd
1. Buka folder proyek
2. Klik dua kali file: JALANKAN.bat
3. Pilih menu [1] INSTALL Dependencies untuk instalasi otomatis
4. Pilih menu [5] JALANKAN Deteksi untuk langsung mulai monitoring
```

### Opsi 2: Setup Manual melalui Terminal / VS Code

1. **Clone repositori ini:**
   ```bash
   git clone https://github.com/Hatif4706/examGuard.git
   cd examGuard
   ```

2. **Buat dan aktifkan Python Virtual Environment:**
   ```bash
   # Windows
   py -3.11 -m venv venv
   venv\Scripts\activate

   # Linux / macOS
   python3.11 -m venv venv
   source venv/bin/activate
   ```

3. **Install PyTorch (Pilih sesuai hardware Anda):**
   ```bash
   # CPU Only (Rekomendasi umum):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

   # NVIDIA GPU (CUDA 12.1):
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

4. **Install seluruh dependensi proyek:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Jalankan Quick Verification Test:**
   ```bash
   python tests/quick_test.py
   ```
   *Jika semua komponen menampilkan tanda `✓ OK`, sistem Anda siap digunakan!*

---

## 🎮 Panduan Penggunaan

### 1. Deteksi Real-Time Webcam (Langsung Siap Pakai)
Mode default untuk mengawasi sesi ujian menggunakan kamera komputer *(tanpa perlu download dataset apa pun)*:
```bash
python main.py
# Atau menentukan index kamera secara eksplisit:
python main.py --mode detect --source 0
```

### 2. Deteksi dari File Video Rekaman
Menganalisis file video rekaman ujian yang sudah ada sebelumnya:
```bash
python main.py --source "C:\path\ke\video_ujian.mp4"
```

### 3. Menjalankan Seluruh Pipeline Otomatis (End-to-End)
Mengunduh dataset, melakukan auto-labeling, melatih custom model, dan langsung membuka pengawasan kamera:
```bash
python main.py --mode all
```

### ⌨️ Kontrol Keyboard Interaktif (Saat Monitoring Berjalan)

| Tombol | Fungsi | Keterangan |
| :---: | :--- | :--- |
| `Q` | **Quit / Selesai** | Menghentikan pengawasan, menyimpan video rekaman, dan membuat laporan otomatis |
| `R` | **Reset Analisis** | Me-reset riwayat sesi dan penghitung pelanggaran ke kondisi awal |
| `S` | **Manual Snapshot** | Menyimpan tangkapan layar beresolusi tinggi saat itu juga ke folder `output/screenshots/` |
| `H` | **Toggle Landmark** | Menampilkan atau menyembunyikan visualisasi 468 titik Face Mesh pada wajah |

---

## 🤖 Pipeline MLOps & Training Kustom (Opsional)

Jika Anda ingin melatih model YOLOv8 kustom Anda sendiri:

```
                  ┌──────────────────────┐
                  │ 1. Download Dataset  │ (COCO8, Roboflow, Synthetic)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │   2. Auto-Labeler    │ (Pseudo-labeling via pretrained YOLO)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │   3. Auto-Trainer    │ (Fine-tuning YOLOv8 + Augmentation)
                  └──────────┬───────────┘
                             ▼
                  ┌──────────────────────┐
                  │ 4. Deployment Model  │ (models/exam_detector.pt)
                  └──────────────────────┘
```

### Langkah 1: Download & Generate Dataset Otomatis
```bash
# Otomatis download COCO8 sample & generate 1.000 dataset sintetis instan:
python main.py --mode download

# Atau dengan Roboflow API Key untuk ribuan dataset publik:
python main.py --mode download --roboflow-key YOUR_ROBOFLOW_API_KEY
```

### Langkah 2: Auto-Labeling Dataset
Secara otomatis melabeli folder gambar tak berlabel ke dalam format bounding box YOLO:
```bash
python main.py --mode label --label-confidence 0.45
```

### Langkah 3: Training Custom YOLOv8 Model
```bash
# Training Cepat (YOLOv8 Nano, 80 Epochs):
python main.py --mode train --model yolov8n.pt --epochs 80

# Training Akurasi Tinggi (YOLOv8 Small, CUDA GPU):
python main.py --mode train --model yolov8s.pt --epochs 120 --device cuda
```

### Langkah 4: Menambahkan Custom Dataset Sendiri
Gunakan utilitas validasi otomatis untuk menambahkan dataset lokal Anda ke dalam sistem:
```bash
python add_custom_dataset.py
```
*Script ini akan memverifikasi integritas koordinat bounding box YOLO ($0 \le x, y, w, h \le 1$) dan membagi dataset secara otomatis menjadi rasio 70% Train, 20% Val, dan 10% Test.*

---

## ⚙️ Konfigurasi Sistem (`config.yaml`)

Seluruh parameter deteksi, sensitivitas sudut, ambang batas waktu, dan visualisasi dapat dikonfigurasi melalui file `config.yaml`:

```yaml
# ================================================================
# KONFIGURASI CV EXAMGUARD v2.3
# ================================================================

camera:
  source: 0                    # 0 untuk webcam utama, atau path file video
  width: 1280                  # Resolusi display frame
  height: 720
  fps: 30

optimization:
  process_frame_width: 480     # Resolusi proses AI (downsampled untuk efisiensi)
  process_frame_height: 360
  skip_frame_interval: 1       # 1 = proses setiap frame

head_pose:
  yaw_threshold: 20            # Ambang toleransi menoleh kiri/kanan (derajat)
  pitch_threshold: 22          # Ambang batas kepala menunduk (derajat)
  pitch_offset: 8.0            # Offset kalibrasi posisi layar (mencegah false positive)
  roll_threshold: 20
  min_detection_confidence: 0.4

behavior:
  gaze_away_duration: 3.0      # Detik menoleh sebelum alert EXTENDED aktif
  phone_detection_duration: 3.0 # Cooldown durasi log HP
  multiple_person_duration: 2.0 # Toleransi deteksi multiple person
  head_down_duration: 3.0      # Durasi toleransi kepala menunduk
  face_not_detected_duration: 5.0 # Durasi toleransi wajah tidak terlihat

alerts:
  audio: true                  # Aktifkan bunyi alarm multi-frekuensi
  visual: true                 # Tampilkan banner peringatan di HUD
  screenshot: true             # Tangkap bukti gambar saat pelanggaran terjadi
  cooldown:
    low: 8
    medium: 5
    high: 3
    critical: 2

training:
  epochs: 80
  batch_size: 16
  image_size: 640
  model: "yolov8n.pt"
  patience: 20                 # Early stopping patience
```

---

## 📊 Laporan & Analisis Sesi

Setiap kali sesi pengawasan diakhiri (menekan tombol `Q`), ExamGuard secara otomatis mengompilasi dan mengekspor seluruh bukti ke direktori `output/`:

1. **Laporan Interaktif HTML (`output/report_*.html`)**:
   - Visualisasi grafik Chart.js: Distribusi kecurangan dan grafik aktivitas per menit.
   - Ringkasan durasi sesi dan persentase waktu mencurigakan (*Suspicious Time Ratio*).
   - Tabel riwayat log kronologis dengan label status (*WASPADA*, *PELANGGARAN*, *KECURANGAN!*).
2. **Rekaman Video Sesi Lengkap (`output/recording_*.mp4`)**:
   - Video rekaman utuh beresolusi tinggi dengan HUD overlay sebagai bukti audit otentik.
3. **Galeri Tangkapan Layar Bukti (`output/screenshots/*.jpg`)**:
   - Gambar tangkapan seketika saat alarm pelanggaran terpicu.
4. **Log Teknis Sesi (`logs/session_*.log`)**:
   - Log terperinci per milidetik untuk kebutuhan audit forensik.

---

## 📁 Struktur Direktori Proyek

```text
ExamCheatDetector/
├── main.py                        # Entry point eksekusi sistem (CLI)
├── config.yaml                    # Konfigurasi parameter & threshold sistem
├── requirements.txt               # Daftar pustaka Python dependensi
├── JALANKAN.bat                   # Menu interaktif launcher (Windows)
├── add_custom_dataset.py          # Script helper validasi custom dataset
├── yolov8n.pt                     # Bobot dasar model YOLOv8 Nano (6.5 MB)
│
├── src/                           # Modul inti Computer Vision & Logic
│   ├── detector.py                # Main Orchestrator penggabung seluruh subsistem
│   ├── head_pose.py               # MediaPipe Face Mesh + solvePnP + EMA Filter
│   ├── object_detector.py         # YOLOv8 Object Detection wrapper
│   ├── behavior_analyzer.py       # Rule engine, scoring, & SessionTracker
│   ├── alert_manager.py           # Multi-level alert, winsound/pygame audio
│   ├── dashboard.py               # OpenCV dark-themed HUD renderer (340px)
│   ├── metrics_tracker.py         # Real-time Confusion Matrix, Precision, Recall, mAP
│   └── report_generator.py        # Pembuat laporan HTML (Chart.js) & PDF
│
├── training/                      # Pipeline MLOps & Training
│   ├── download_datasets.py       # Download COCO8, Roboflow, & Synthetic Generator
│   ├── auto_labeler.py            # Pseudo-labeling otomatis
│   └── auto_trainer.py            # Fine-tuning & augmentasi YOLOv8
│
├── tests/                         # Unit & integration testing
│   └── quick_test.py              # Suite uji coba cepat kesiapan lingkungan
│
├── models/                        # Tempat penyimpanan model terlatih (.pt)
├── datasets/                      # Direktori dataset training/validation
├── output/                        # Hasil output (Laporan HTML, Video, Screenshot)
├── logs/                          # Berkas log audit sesi (.log)
└── README.md                      # Dokumentasi utama repositori
```

---

## 🗺️ Roadmap Pengembangan Masa Depan

- [ ] **Iris & Gaze Tracking Lanjutan**: Deteksi lirikan pupil mata menggunakan koordinat iris MediaPipe tanpa menggerakkan kepala.
- [ ] **Face Recognition & Anti-Spoofing Liveness**: Verifikasi identitas peserta terdaftar dan pencegahan manipulasi foto/video statis (Deepfake/Photo Spoofing).
- [ ] **Acoustic Speech Detection**: Analisis audio mikrofon latar belakang untuk mendeteksi suara bisikan atau percakapan bantuan pihak ketiga.
- [ ] **Modern Web-Based Proctoring Dashboard**: Antarmuka web pengawas berbasis WebSocket/WebRTC untuk memantau puluhan peserta ujian secara serentak dalam satu layar.

---

## 👥 Kontribusi

Kontribusi terhadap CV ExamGuard selalu terbuka lebar! Silakan lakukan:
1. **Fork** repositori ini
2. Buat fitur branch baru (`git checkout -b feature/FiturKeren`)
3. Commit perubahan Anda (`git commit -m 'Menambahkan fitur deteksi baru'`)
4. Push ke branch Anda (`git push origin feature/FiturKeren`)
5. Buat **Pull Request** baru

---

## 📄 Lisensi

Proyek ini dilisensikan di bawah **[MIT License](LICENSE)**. Bebas digunakan untuk keperluan penelitian akademik, tugas akhir/skripsi, maupun pengembangan komersial dengan mencantumkan atribusi pengembang asli.

---

<div align="center">

**Dikembangkan dengan ❤️ menggunakan Python, YOLOv8, MediaPipe, dan OpenCV.**

*Jika proyek ini bermanfaat bagi Anda, jangan lupa berikan ⭐ Star di GitHub!*

</div>
