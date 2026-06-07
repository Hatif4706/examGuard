@echo off
title CV ExamGuard - Sistem Deteksi Kecurangan Ujian

echo.
echo ================================================================
echo           CV EXAMGUARD - SISTEM DETEKSI KECURANGAN UJIAN
echo              Berbasis Computer Vision AND AI (YOLOv8)
echo ================================================================
echo.

:: Cek Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python tidak ditemukan!
    echo          Install Python 3.9+ dari https://python.org
    echo          Pastikan centang "Add Python to PATH" saat install
    pause
    exit /b 1
)

echo [OK] Python ditemukan
python --version

:: Cek pip
pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip tidak ditemukan!
    pause
    exit /b 1
)

:: Pindah ke direktori script
cd /d "%~dp0"

:start
echo.
echo ================================================================
echo   MENU UTAMA
echo ================================================================
echo.
echo   [1] INSTALL Dependencies (wajib pertama kali)
echo   [2] DOWNLOAD Dataset Otomatis
echo   [3] AUTO-LABEL Dataset
echo   [4] TRAINING Model Custom
echo   [5] JALANKAN Deteksi (Webcam)
echo   [6] JALANKAN Deteksi (File Video)
echo   [7] PIPELINE LENGKAP (Download+Label+Train+Detect)
echo   [8] Keluar
echo.
set /p choice="Pilih [1-8]: "

if "%choice%"=="1" goto install
if "%choice%"=="2" goto download
if "%choice%"=="3" goto label
if "%choice%"=="4" goto train
if "%choice%"=="5" goto detect_cam
if "%choice%"=="6" goto detect_video
if "%choice%"=="7" goto pipeline
if "%choice%"=="8" goto end

echo Pilihan tidak valid
pause
goto start

:install
echo.
echo ================================================================
echo   INSTALASI DEPENDENCIES
echo ================================================================
echo.
echo Menginstall semua library yang diperlukan...
echo Proses ini mungkin memakan waktu 10-15 menit...
echo.

python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
pip install mediapipe
pip install -r requirements.txt

echo.
echo [OK] Instalasi selesai!
echo.
pause
goto start

:download
echo.
echo ================================================================
echo   DOWNLOAD DATASET
echo ================================================================
echo.
python main.py --mode download
echo.
pause
goto start

:label
echo.
echo ================================================================
echo   AUTO-LABELING DATASET
echo ================================================================
echo.
python main.py --mode label
echo.
pause
goto start

:train
echo.
echo ================================================================
echo   TRAINING MODEL CUSTOM
echo ================================================================
echo.
set /p epochs="Jumlah epoch [50]: "
if "%epochs%"=="" set epochs=50

set /p model="Model YOLOv8 [yolov8n.pt / yolov8s.pt / yolov8m.pt]: "
if "%model%"=="" set model=yolov8n.pt

python main.py --mode train --epochs %epochs% --model %model%
echo.
pause
goto start

:detect_cam
echo.
echo ================================================================
echo   DETEKSI REALTIME - WEBCAM
echo ================================================================
echo.
set /p cam_id="Index kamera [0]: "
if "%cam_id%"=="" set cam_id=0

python main.py --mode detect --source %cam_id%
echo.
pause
goto start

:detect_video
echo.
echo ================================================================
echo   DETEKSI - FILE VIDEO
echo ================================================================
echo.
set /p video_path="Path file video (contoh: C:\video\ujian.mp4): "
if "%video_path%"=="" (
    echo Path video tidak boleh kosong
    pause
    goto start
)

python main.py --mode detect --source "%video_path%"
echo.
pause
goto start

:pipeline
echo.
echo ================================================================
echo   PIPELINE LENGKAP
echo   Download + Label + Train + Detect
echo ================================================================
echo.
echo Proses ini akan:
echo   1. Download dataset dari internet
echo   2. Auto-label semua gambar
echo   3. Training model custom (50 epoch)
echo   4. Memulai deteksi real-time
echo.
echo Estimasi waktu: 30-90 menit tergantung hardware
echo.
set /p confirm="Lanjutkan? [y/N]: "
if /i "%confirm%"=="y" (
    python main.py --mode all --epochs 50
)
echo.
pause
goto start

:end
echo.
echo Terima kasih telah menggunakan CV ExamGuard!
echo.
exit /b 0
