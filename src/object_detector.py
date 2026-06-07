"""
Object Detector menggunakan YOLOv8
Mendeteksi: orang, HP, buku, laptop, dan objek mencurigakan lainnya
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2
    center: Tuple[int, int]
    area: int
    is_suspicious: bool = False

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class DetectionResult:
    persons: List[Detection] = field(default_factory=list)
    phones: List[Detection] = field(default_factory=list)
    books: List[Detection] = field(default_factory=list)
    laptops: List[Detection] = field(default_factory=list)
    suspicious_objects: List[Detection] = field(default_factory=list)
    all_detections: List[Detection] = field(default_factory=list)

    @property
    def person_count(self) -> int:
        return len(self.persons)

    @property
    def has_phone(self) -> bool:
        return len(self.phones) > 0

    @property
    def has_multiple_persons(self) -> bool:
        return self.person_count > 1

    @property
    def has_suspicious_object(self) -> bool:
        return len(self.suspicious_objects) > 0


class ObjectDetector:
    """
    Wrapper YOLOv8 untuk deteksi objek dalam konteks ujian.
    Mendukung model pre-trained COCO dan model custom.
    """

    # Kelas COCO yang relevan untuk ujian (index: nama)
    COCO_RELEVANT_CLASSES = {
        0:  "person",
        63: "laptop",
        64: "mouse",
        65: "remote",
        66: "keyboard",
        67: "cell phone",
        73: "book",
        74: "clock",
        76: "scissors",
        77: "teddy bear",
        84: "book",
    }

    # Mapping kelas ke kategori
    SUSPICIOUS_CATEGORIES = {
        "cell phone": "phone",
        "laptop": "laptop",
        "book": "book",
        "remote": "suspicious",
        "scissors": "suspicious",
    }

    # Warna per kategori (BGR)
    COLORS = {
        "person": (0, 255, 0),
        "phone": (0, 0, 255),
        "laptop": (255, 0, 0),
        "book": (0, 165, 255),
        "suspicious": (0, 0, 200),
        "default": (128, 128, 128),
    }

    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.45,  # ↓ TURUN dari 0.5 untuk detect lebih banyak
                 iou_threshold: float = 0.45,
                 device: str = "auto"):

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = None
        self.model_path = model_path
        self.device = self._resolve_device(device)
        self._load_model(model_path)

    def _resolve_device(self, device: str) -> str:
        """Pilih device terbaik yang tersedia."""
        if device != "auto":
            return device
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("Menggunakan GPU (CUDA)")
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                logger.info("Menggunakan GPU (Apple MPS)")
                return "mps"
        except ImportError:
            pass
        logger.info("Menggunakan CPU")
        return "cpu"

    def _load_model(self, model_path: str):
        """Load model YOLOv8."""
        try:
            from ultralytics import YOLO
            logger.info(f"Loading model: {model_path}")
            self.model = YOLO(model_path)
            self.model.to(self.device)

            # Get class names
            self.class_names = self.model.names
            logger.info(f"Model loaded. Classes: {len(self.class_names)}")
        except Exception as e:
            logger.error(f"Gagal load model: {e}")
            raise

    def _classify_detection(self, class_name: str, bbox: Tuple) -> Tuple[str, bool]:
        """Klasifikasikan deteksi ke kategori dan tentukan apakah mencurigakan."""
        class_lower = class_name.lower()

        if class_lower == "person":
            return "person", False
        elif class_lower in ["cell phone", "phone"]:
            return "phone", True
        elif class_lower in ["laptop", "computer"]:
            return "laptop", True
        elif class_lower in ["book", "notebook"]:
            return "book", True
        elif class_lower in ["remote", "scissors"]:
            return "suspicious", True
        else:
            return "default", False

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """
        Deteksi objek dalam frame.
        
        Args:
            frame: Frame BGR dari OpenCV
            
        Returns:
            DetectionResult dengan semua deteksi
        """
        if self.model is None:
            return DetectionResult()

        result_obj = DetectionResult()

        try:
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                device=self.device,
                verbose=False,
                stream=False
            )

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()

                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    area = (x2 - x1) * (y2 - y1)

                    # Nama kelas
                    class_name = self.class_names.get(class_id, str(class_id))

                    # Filter hanya kelas yang relevan
                    if class_name.lower() not in [c.lower() for c in self.COCO_RELEVANT_CLASSES.values()]:
                        if class_name.lower() != "person":
                            continue

                    category, is_suspicious = self._classify_detection(class_name, (x1, y1, x2, y2))

                    detection = Detection(
                        class_id=class_id,
                        class_name=class_name,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        center=center,
                        area=area,
                        is_suspicious=is_suspicious
                    )

                    result_obj.all_detections.append(detection)

                    if category == "person":
                        result_obj.persons.append(detection)
                    elif category == "phone":
                        result_obj.phones.append(detection)
                    elif category == "laptop":
                        result_obj.laptops.append(detection)
                    elif category == "book":
                        result_obj.books.append(detection)
                    elif is_suspicious:
                        result_obj.suspicious_objects.append(detection)

        except Exception as e:
            logger.error(f"Error saat deteksi: {e}")

        return result_obj

    def draw_detections(self, frame: np.ndarray,
                        detection_result: DetectionResult,
                        show_confidence: bool = True) -> np.ndarray:
        """Gambar bounding box dan label deteksi pada frame."""
        for detection in detection_result.all_detections:
            x1, y1, x2, y2 = detection.bbox
            category, _ = self._classify_detection(detection.class_name, detection.bbox)
            color = self.COLORS.get(category, self.COLORS["default"])

            # Gambar bbox
            thickness = 3 if detection.is_suspicious else 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Label
            if show_confidence:
                label = f"{detection.class_name}: {detection.confidence:.2f}"
            else:
                label = detection.class_name

            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            label_y = max(y1 - 10, label_size[1] + 10)
            cv2.rectangle(frame,
                          (x1, label_y - label_size[1] - 5),
                          (x1 + label_size[0] + 5, label_y + 3),
                          color, -1)
            cv2.putText(frame, label,
                        (x1 + 2, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 255), 1)

            # Tanda peringatan untuk objek mencurigakan
            if detection.is_suspicious:
                cx, cy = detection.center
                cv2.putText(frame, "!",
                            (cx - 5, cy + 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 255), 3)

        return frame

    def switch_model(self, model_path: str):
        """Ganti model ke path baru."""
        if Path(model_path).exists():
            self._load_model(model_path)
            logger.info(f"Model diganti ke: {model_path}")
        else:
            logger.warning(f"Model tidak ditemukan: {model_path}")
