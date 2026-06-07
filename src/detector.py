"""
Exam Cheat Detector - Orchestrator Utama
Mengintegrasikan semua komponen: head pose, object detection, behavior analysis, alerts
"""

import cv2
import time
import logging
import yaml
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

from .head_pose import HeadPoseEstimator
from .object_detector import ObjectDetector, DetectionResult
from .behavior_analyzer import BehaviorAnalyzer
from .alert_manager import AlertManager, AlertConfig
from .report_generator import ReportGenerator
from .dashboard import DashboardRenderer
from .metrics_tracker import MetricsTracker

logger = logging.getLogger(__name__)


class ExamCheatDetector:
    """
    Kelas utama yang mengintegrasikan semua komponen deteksi kecurangan.
    """

    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.session_name = self._get_session_name()

        logger.info("=" * 50)
        logger.info("  SISTEM DETEKSI KECURANGAN UJIAN")
        logger.info("  CV ExamGuard v1.0 ENHANCED")
        logger.info("=" * 50)
        logger.info(f"Sesi: {self.session_name}")
        logger.info("Inisialisasi komponen...")

        # Inisialisasi state untuk optimization
        self._init_process_frame_state()
        
        # Inisialisasi semua komponen
        self._init_components()
        # Test audio singkat saat startup
        try:
            if self.alert_manager.audio and self.alert_manager.audio.enabled:
                import threading
                threading.Thread(
                    target=self.alert_manager.audio._play_pattern,
                    args=([(880, 120), (0, 60), (1100, 120)],),
                    daemon=True
                ).start()
                logger.info(f"Audio ready [{self.alert_manager.audio._backend}]")
        except Exception:
            pass
        # Hubungkan metrics ke dashboard untuk tampilan real-time
        self.dashboard._metrics_ref = self.metrics
        logger.info("Semua komponen siap!")

    def _load_config(self, config_path: str) -> dict:
        """Load konfigurasi dari YAML."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Konfigurasi loaded dari: {config_path}")
            return config
        except FileNotFoundError:
            logger.warning(f"Config tidak ditemukan: {config_path}, pakai default")
            return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return {}

    def _get_session_name(self) -> str:
        """Generate nama sesi."""
        session_cfg = self.config.get("logging", {}).get("session_name", "auto")
        if session_cfg == "auto":
            return datetime.now().strftime("sesi_%Y%m%d_%H%M%S")
        return session_cfg

    def _init_process_frame_state(self):
        """Inisialisasi state untuk frame processing optimization."""
        self._last_detection_result = None
        
    def _init_components(self):
        """Inisialisasi semua komponen deteksi."""
        cfg = self.config

        # Head Pose Estimator
        hp_cfg = cfg.get("head_pose", {})
        logger.info("Loading Head Pose Estimator (MediaPipe)...")
        # pitch_threshold untuk HeadPoseEstimator = threshold + offset
        # agar looking_straight akurat (tidak false positive saat duduk tegak)
        raw_pitch_thresh = hp_cfg.get("pitch_threshold", 22.0)
        pitch_off        = hp_cfg.get("pitch_offset", 8.0)
        self.head_pose_estimator = HeadPoseEstimator(
            yaw_threshold=hp_cfg.get("yaw_threshold", 20.0),
            pitch_threshold=raw_pitch_thresh + pitch_off,  # e.g. 22+8=30°
            roll_threshold=hp_cfg.get("roll_threshold", 20.0),
            max_faces=hp_cfg.get("max_faces", 4),
            min_detection_confidence=hp_cfg.get("min_detection_confidence", 0.4),
            min_tracking_confidence=hp_cfg.get("min_tracking_confidence", 0.4)
        )

        # Object Detector (YOLOv8)
        model_cfg = cfg.get("models", {})
        use_custom = model_cfg.get("use_custom", False)
        model_path = (model_cfg.get("custom_model", "models/exam_detector.pt")
                      if use_custom else model_cfg.get("yolo_model", "yolov8n.pt"))

        # Cek apakah custom model tersedia
        if use_custom and not Path(model_path).exists():
            logger.warning(f"Custom model tidak ditemukan di {model_path}, pakai YOLOv8 pretrained")
            model_path = model_cfg.get("yolo_model", "yolov8n.pt")

        logger.info(f"Loading Object Detector (YOLOv8): {model_path}...")
        self.object_detector = ObjectDetector(
            model_path=model_path,
            confidence_threshold=model_cfg.get("confidence_threshold", 0.5),
            iou_threshold=model_cfg.get("iou_threshold", 0.45),
            device=model_cfg.get("device", "auto")
        )

        # Metrics Tracker
        self.metrics = MetricsTracker(
            confidence_threshold=model_cfg.get("confidence_threshold", 0.40)
        )

        # Behavior Analyzer
        beh_cfg = cfg.get("behavior", {})
        logger.info("Inisialisasi Behavior Analyzer...")
        self.behavior_analyzer = BehaviorAnalyzer(
            gaze_away_duration=beh_cfg.get("gaze_away_duration", 3.0),
            phone_duration=beh_cfg.get("phone_detection_duration", 3.0),
            multiple_person_duration=beh_cfg.get("multiple_person_duration", 2.0),
            head_down_duration=beh_cfg.get("head_down_duration", 3.0),
            yaw_threshold=hp_cfg.get("yaw_threshold", 20.0),
            pitch_threshold=hp_cfg.get("pitch_threshold", 22.0),
            pitch_offset=hp_cfg.get("pitch_offset", 8.0),
        )

        # Alert Manager
        alert_cfg = cfg.get("alerts", {})
        cooldowns = alert_cfg.get("cooldown", {})
        log_cfg = cfg.get("logging", {})
        alert_config = AlertConfig(
            visual=alert_cfg.get("visual", True),
            audio=alert_cfg.get("audio", True),
            log_to_file=alert_cfg.get("log_to_file", True),
            screenshot_on_alert=alert_cfg.get("screenshot_on_alert", True),
            audio_volume=alert_cfg.get("audio_volume", 0.7),
            cooldowns={
                "low":      float(cooldowns.get("low", 10.0)),
                "medium":   float(cooldowns.get("medium", 5.0)),
                "high":     float(cooldowns.get("high", 3.0)),
                "critical": float(cooldowns.get("critical", 1.0)),
            },
            output_dir=log_cfg.get("output_dir", "output"),
            log_dir=log_cfg.get("log_dir", "logs")
        )
        logger.info("Inisialisasi Alert Manager...")
        self.alert_manager = AlertManager(config=alert_config)

        # Report Generator
        self.report_generator = ReportGenerator(
            output_dir=log_cfg.get("output_dir", "output")
        )

        # Dashboard
        display_cfg = cfg.get("display", {})
        self.dashboard = DashboardRenderer(
            dashboard_width=display_cfg.get("dashboard_width", 360)
        )

        # Settings
        self.show_landmarks = display_cfg.get("show_landmarks", False)
        self.show_fps = display_cfg.get("show_fps", True)
        self.show_angles = display_cfg.get("show_angles", True)
        self.save_video = log_cfg.get("save_detection_video", True)
        self.generate_report = log_cfg.get("generate_report", True)
        self.report_format = log_cfg.get("report_format", "html")

        # FPS Optimization Settings
        opt_cfg = cfg.get("optimization", {})
        self.process_frame_width = opt_cfg.get("process_frame_width", 640)
        self.process_frame_height = opt_cfg.get("process_frame_height", 480)
        self.target_fps = opt_cfg.get("target_fps", 30)
        self.skip_frame_interval = opt_cfg.get("skip_frame_interval", 1)

        # Video writer (inisialisasi saat run)
        self.video_writer = None
        self.output_dir = Path(log_cfg.get("output_dir", "output"))

    def _init_video_writer(self, frame_shape, fps: float):
        """Inisialisasi video writer untuk merekam sesi."""
        if not self.save_video:
            return

        h, w_with_dash = frame_shape[:2]
        filename = f"recording_{self.session_name}.mp4"
        filepath = self.output_dir / filename

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            str(filepath), fourcc, fps, (w_with_dash, h)
        )
        if self.video_writer.isOpened():
            logger.info(f"Merekam video ke: {filepath}")
        else:
            logger.warning("Gagal inisialisasi video writer")
            self.video_writer = None

    def _process_frame(self, frame, frame_number: int):
        """Proses satu frame dengan aggressive FPS optimization."""
        annotated = frame.copy()
        
        # ========== AGGRESSIVE FPS OPTIMIZATION ==========
        # Resize frame untuk inference (buat JAUH lebih kecil)
        process_frame = cv2.resize(frame, (480, 360))  # ↓ Turun dari 640x480
        scale_x = frame.shape[1] / 480
        scale_y = frame.shape[0] / 360

        # 1. Object Detection (skip frame untuk speed)
        skip_interval = getattr(self, 'skip_frame_interval', 1)
        if frame_number % skip_interval == 0:
            detection_result = self.object_detector.detect(process_frame)
            
            # Scale kembali bounding box
            for detection in detection_result.all_detections:
                x1, y1, x2, y2 = detection.bbox
                detection.bbox = (
                    int(x1 * scale_x), int(y1 * scale_y),
                    int(x2 * scale_x), int(y2 * scale_y)
                )
                cx, cy = detection.center
                detection.center = (int(cx * scale_x), int(cy * scale_y))
            
            self._last_detection_result = detection_result
        else:
            # Reuse hasil deteksi sebelumnya untuk frame yang di-skip
            detection_result = self._last_detection_result if self._last_detection_result else DetectionResult()
        
        self.object_detector.draw_detections(annotated, detection_result)

        # Update metrics tracker
        self.metrics.update_from_detection(detection_result)

        # 2. Head Pose Estimation (dari frame original)
        head_poses = self.head_pose_estimator.estimate(annotated)
        for hp in head_poses:
            self.head_pose_estimator.draw_head_direction(
                annotated, hp,
                draw_landmarks=self.show_landmarks
            )

        # 3. Behavior Analysis
        analysis = self.behavior_analyzer.analyze(
            head_poses, detection_result, frame_number
        )

        # 4. Alert Manager
        self.alert_manager.process_frame(annotated, analysis, self.session_name)

        # 5. Render Dashboard
        combined = self.dashboard.render(annotated, analysis)

        return combined, analysis

    def _draw_info_overlay(self, frame, fps: float):
        """Gambar informasi tambahan di sudut frame."""
        if self.show_fps:
            cv2.putText(frame, f"FPS: {fps:.1f}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        # Instruksi
        help_text = "Tekan 'Q' untuk keluar | 'R' untuk reset | 'S' untuk screenshot"
        h, w = frame.shape[:2]
        cv2.putText(frame, help_text,
                    (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (100, 116, 139), 1)

    def run(self, source=0, save_video: bool = True,
             display: bool = True) -> dict:
        """
        Jalankan deteksi real-time.
        
        Args:
            source: Sumber video (0=webcam, atau path file)
            save_video: Apakah simpan video output
            display: Apakah tampilkan window
            
        Returns:
            Dict statistik sesi
        """
        self.save_video = save_video

        # Buka sumber video
        logger.info(f"Membuka sumber video: {source}")
        cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)

        if not cap.isOpened():
            logger.error(f"Tidak bisa membuka sumber video: {source}")
            return {}

        # Set resolusi kamera
        cam_cfg = self.config.get("camera", {})
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_cfg.get("width", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_cfg.get("height", 720))
        cap.set(cv2.CAP_PROP_FPS, cam_cfg.get("fps", 30))

        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        logger.info(f"Kamera berjalan pada {actual_fps:.0f} FPS")

        frame_number = 0
        fps_counter = []
        video_writer_initialized = False

        # Window setup
        window_name = "CV ExamGuard - Sistem Deteksi Kecurangan Ujian"
        if display:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 1400, 720)

        logger.info("Memulai deteksi... Tekan 'Q' untuk keluar")
        print("\n" + "=" * 60)
        print("  DETEKSI DIMULAI")
        print("  Tekan Q untuk keluar")
        print("  Tekan R untuk reset analisis")
        print("  Tekan S untuk screenshot manual")
        print("  Tekan H untuk toggle landmark")
        print("=" * 60 + "\n")

        try:
            while True:
                t_start = time.time()
                ret, frame = cap.read()

                if not ret:
                    logger.warning("Tidak bisa membaca frame, mencoba lagi...")
                    if isinstance(source, str) and source != "0":
                        break  # Video file selesai
                    time.sleep(0.1)
                    continue

                frame_number += 1

                # Proses frame
                combined_frame, analysis = self._process_frame(frame, frame_number)

                # Hitung FPS
                t_end = time.time()
                fps_counter.append(1.0 / max(t_end - t_start, 0.001))
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                current_fps = sum(fps_counter) / len(fps_counter)

                # Overlay info
                self._draw_info_overlay(combined_frame, current_fps)

                # Inisialisasi video writer di frame pertama
                if not video_writer_initialized and self.save_video:
                    self._init_video_writer(combined_frame.shape, actual_fps)
                    video_writer_initialized = True

                # Tulis frame ke video
                if self.video_writer and self.video_writer.isOpened():
                    self.video_writer.write(combined_frame)

                # Tampilkan frame
                if display:
                    cv2.imshow(window_name, combined_frame)

                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == ord('Q'):
                        logger.info("Keluar dari deteksi (tombol Q)")
                        break
                    elif key == ord('r') or key == ord('R'):
                        self.behavior_analyzer.reset()
                        self.metrics.reset()
                        print("✓ Analisis perilaku di-reset")
                    elif key == ord('s') or key == ord('S'):
                        # Screenshot manual
                        ts = datetime.now().strftime("%H%M%S")
                        ss_path = self.output_dir / f"manual_screenshot_{ts}.jpg"
                        cv2.imwrite(str(ss_path), combined_frame)
                        print(f"✓ Screenshot disimpan: {ss_path}")
                    elif key == ord('h') or key == ord('H'):
                        self.show_landmarks = not self.show_landmarks
                        print(f"Landmark: {'ON' if self.show_landmarks else 'OFF'}")
                else:
                    # Headless mode - berhenti hanya jika video file selesai
                    if analysis and frame_number % 100 == 0:
                        logger.info(f"Frame {frame_number} diproses | "
                                   f"Alert: {analysis.alert_level.label()}")

        except KeyboardInterrupt:
            logger.info("Dihentikan oleh pengguna (Ctrl+C)")
        except Exception as e:
            logger.error(f"Error saat deteksi: {e}", exc_info=True)
        finally:
            # Cleanup
            logger.info("Membersihkan resource...")
            cap.release()

            if self.video_writer:
                self.video_writer.release()
                logger.info("Video recording disimpan")

            if display:
                cv2.destroyAllWindows()

            self.head_pose_estimator.close()
            self.alert_manager.close()

            # Generate laporan
            stats  = self.behavior_analyzer.get_session_stats()
            events = self.behavior_analyzer.event_history

            # Print metrik performa
            print("\n" + self.metrics.format_report())

            if self.generate_report and frame_number > 10:
                logger.info("Membuat laporan sesi...")
                if self.report_format == "html" or self.report_format == "both":
                    report_path = self.report_generator.generate_html_report(
                        stats, events, self.session_name)
                    print(f"\n✓ Laporan HTML: {report_path}")

                if self.report_format == "pdf" or self.report_format == "both":
                    pdf_path = self.report_generator.generate_pdf_report(
                        stats, events, self.session_name)
                    if pdf_path:
                        print(f"✓ Laporan PDF:  {pdf_path}")

            # Print ringkasan
            self._print_summary(stats, frame_number)
            return stats

    def _print_summary(self, stats: dict, total_frames: int):
        """Print ringkasan sesi ke konsol."""
        duration = stats.get("session_duration", 0)
        print("\n" + "=" * 60)
        print("  RINGKASAN SESI DETEKSI")
        print("=" * 60)
        print(f"  Durasi        : {int(duration//60):02d}:{int(duration%60):02d}")
        print(f"  Total Frame   : {total_frames:,}")
        print(f"  Total Kejadian: {stats.get('total_events', 0)}")
        print(f"  Skor Kumulatif: {stats.get('total_cheat_score', 0)}")
        print(f"  Tingkat Risiko: {stats.get('risk_level', '-')}")
        print()
        events_by_type = stats.get("events_by_type", {})
        if events_by_type:
            print("  Detail Kejadian:")
            for event_type, count in events_by_type.items():
                print(f"    • {event_type}: {count}x")
        print("=" * 60)
        print(f"  Output tersimpan di: output/")
        print("=" * 60 + "\n")
