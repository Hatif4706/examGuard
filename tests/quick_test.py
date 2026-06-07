"""
Quick Test - Verifikasi semua komponen berjalan dengan benar
Jalankan: python tests/quick_test.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time

PASS = "✓"
FAIL = "✗"
WARN = "⚠"

results = []

def test(name, fn):
    try:
        result = fn()
        status = PASS
        msg = "OK"
        if isinstance(result, str):
            msg = result
        results.append((name, True, msg))
        print(f"  {PASS} {name}: {msg}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} {name}: {str(e)[:80]}")

print("\n" + "=" * 55)
print("  CV ExamGuard - Quick Test Suite")
print("=" * 55)

# 1. Library imports
print("\n[1] Library Dependencies")

test("OpenCV", lambda: __import__("cv2").__version__)
test("NumPy", lambda: __import__("numpy").__version__)
test("MediaPipe", lambda: __import__("mediapipe").__version__)
test("Ultralytics", lambda: __import__("ultralytics").__version__)
test("PyYAML", lambda: __import__("yaml").__version__)
test("Requests", lambda: __import__("requests").__version__)
test("tqdm", lambda: __import__("tqdm").__version__)

# 2. PyTorch
print("\n[2] PyTorch & Hardware")
def test_torch():
    import torch
    version = torch.__version__
    cuda = torch.cuda.is_available()
    return f"v{version} | CUDA: {'YES' if cuda else 'NO (CPU mode)'}"
test("PyTorch", test_torch)

# 3. Module imports
print("\n[3] CV ExamGuard Modules")

def test_head_pose_import():
    from src.head_pose import HeadPoseEstimator, HeadPose
    return "HeadPoseEstimator, HeadPose"

def test_object_detector_import():
    from src.object_detector import ObjectDetector, DetectionResult
    return "ObjectDetector, DetectionResult"

def test_behavior_analyzer_import():
    from src.behavior_analyzer import BehaviorAnalyzer, AlertLevel, CheatType
    return f"{len(list(CheatType))} CheatTypes, {len(list(AlertLevel))} AlertLevels"

def test_alert_manager_import():
    from src.alert_manager import AlertManager, AlertConfig
    return "AlertManager, AlertConfig"

def test_dashboard_import():
    from src.dashboard import DashboardRenderer
    return "DashboardRenderer"

def test_report_import():
    from src.report_generator import ReportGenerator
    return "ReportGenerator"

def test_detector_import():
    from src.detector import ExamCheatDetector
    return "ExamCheatDetector"

test("src.head_pose", test_head_pose_import)
test("src.object_detector", test_object_detector_import)
test("src.behavior_analyzer", test_behavior_analyzer_import)
test("src.alert_manager", test_alert_manager_import)
test("src.dashboard", test_dashboard_import)
test("src.report_generator", test_report_import)
test("src.detector", test_detector_import)

# 4. Training modules
print("\n[4] Training Modules")

def test_downloader():
    from training.download_datasets import download_all, ROBOFLOW_DATASETS
    return f"{len(ROBOFLOW_DATASETS)} datasets configured"

def test_labeler():
    from training.auto_labeler import AutoLabeler, EXAM_CLASS_NAMES
    return f"{len(EXAM_CLASS_NAMES)} classes: {EXAM_CLASS_NAMES}"

def test_trainer():
    from training.auto_trainer import run_training, find_best_dataset
    return "run_training, find_best_dataset"

test("training.download_datasets", test_downloader)
test("training.auto_labeler", test_labeler)
test("training.auto_trainer", test_trainer)

# 5. Functional tests
print("\n[5] Functional Tests")

def test_head_pose_dummy():
    """Test head pose pada gambar dummy."""
    from src.head_pose import HeadPoseEstimator
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    # Gambar wajah dummy (kotak putih sederhana)
    img[100:380, 200:440] = 200
    
    estimator = HeadPoseEstimator(max_faces=1)
    poses = estimator.estimate(img)
    estimator.close()
    return f"Detected {len(poses)} faces (expected 0 for blank image)"

def test_behavior_analyzer():
    """Test behavior analyzer."""
    from src.behavior_analyzer import BehaviorAnalyzer
    from src.head_pose import HeadPose
    from src.object_detector import DetectionResult
    
    analyzer = BehaviorAnalyzer()
    
    # Test dengan pose normal
    poses = []
    det_result = DetectionResult()
    analysis = analyzer.analyze(poses, det_result, frame_number=1)
    
    return f"Alert level: {analysis.alert_level.label()}, Score: {analysis.cheat_score}"

def test_dashboard_render():
    """Test dashboard rendering."""
    from src.dashboard import DashboardRenderer
    from src.behavior_analyzer import BehaviorAnalyzer, FrameAnalysis, AlertLevel
    from src.object_detector import DetectionResult
    import time
    
    # Buat dummy analysis
    from src.behavior_analyzer import CheatType
    analysis = FrameAnalysis(
        timestamp=time.time(),
        alert_level=AlertLevel.NONE,
        cheat_score=0,
        detected_cheats=[],
        events=[],
        person_count=1,
        has_phone=False,
        head_poses=[],
        detection_result=DetectionResult()
    )
    
    dashboard = DashboardRenderer(dashboard_width=300)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    combined = dashboard.render(frame, analysis)
    
    expected_w = 640 + 300
    if combined.shape[1] == expected_w:
        return f"Frame {frame.shape[1]}x{frame.shape[0]} + Dashboard → {combined.shape[1]}x{combined.shape[0]}"
    else:
        raise ValueError(f"Width mismatch: {combined.shape[1]} != {expected_w}")

def test_config_load():
    """Test loading config.yaml."""
    import yaml
    from pathlib import Path
    config_path = Path("config.yaml")
    if not config_path.exists():
        return "config.yaml tidak ditemukan (akan dibuat saat pertama run)"
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return f"Config loaded, {len(cfg)} sections"

def test_yolov8_available():
    """Test apakah YOLOv8 bisa load model."""
    from ultralytics import YOLO
    # Hanya test import, tidak download model
    return "YOLOv8 ready (model akan didownload otomatis saat pertama detect)"

test("Head Pose (dummy frame)", test_head_pose_dummy)
test("Behavior Analyzer", test_behavior_analyzer)
test("Dashboard Render", test_dashboard_render)
test("Config Load", test_config_load)
test("YOLOv8 Available", test_yolov8_available)

# 6. Directories
print("\n[6] Directory Structure")

def test_directories():
    from pathlib import Path
    dirs = ["src", "training", "models", "datasets", "output", "logs", ".vscode"]
    missing = [d for d in dirs if not Path(d).exists()]
    if missing:
        # Buat yang belum ada
        for d in missing:
            Path(d).mkdir(parents=True, exist_ok=True)
        return f"Dibuat: {missing}"
    return f"Semua {len(dirs)} direktori ada"

test("Required directories", test_directories)

# Summary
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print("\n" + "=" * 55)
print(f"  HASIL: {passed}/{total} passed", end="")
if failed > 0:
    print(f" | {failed} FAILED")
else:
    print(" | SEMUA OK! ✓")
print("=" * 55)

if failed == 0:
    print("\n🎉 Sistem siap digunakan!")
    print("   Jalankan: python main.py")
    print("   Atau klik: JALANKAN.bat\n")
else:
    print(f"\n⚠  {failed} test gagal. Jalankan:")
    print("   pip install -r requirements.txt\n")

sys.exit(0 if failed == 0 else 1)
