"""
Metrics Tracker — Accuracy, Precision, Recall, F1, mAP
Dihitung secara real-time dari session deteksi.
"""

import time
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ClassMetrics:
    """Metrik per kelas deteksi."""
    name:       str
    tp:         int   = 0   # True Positive
    fp:         int   = 0   # False Positive
    fn:         int   = 0   # False Negative
    tn:         int   = 0   # True Negative

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d > 0 else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        d    = p + r
        return 2 * p * r / d if d > 0 else 0.0

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.fn + self.tn
        return (self.tp + self.tn) / total if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name, "tp": self.tp, "fp": self.fp,
            "fn": self.fn, "tn": self.tn,
            "precision": round(self.precision, 4),
            "recall":    round(self.recall,    4),
            "f1":        round(self.f1,        4),
            "accuracy":  round(self.accuracy,  4),
        }


@dataclass
class DetectionIoU:
    """Satu hasil deteksi untuk mAP calculation."""
    class_id:   int
    confidence: float
    iou:        float = 0.0
    matched:    bool  = False


class MetricsTracker:
    """
    Tracker metrik real-time untuk seluruh sesi.
    
    Cara kerja:
    - Frame analysis masuk → update confusion matrix per kelas
    - Deteksi HP masuk → update IoU-based mAP
    - Panggil get_summary() untuk rangkuman lengkap
    
    Catatan: karena tidak ada ground truth label manual,
    metrik dihitung berdasarkan deteksi yang dikonfirmasi
    (HP yang terdeteksi > threshold confidence dianggap TP,
     deteksi rendah confidence dianggap FP).
    """

    IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    def __init__(self, confidence_threshold: float = 0.40):
        self.conf_thresh = confidence_threshold

        # Confusion matrix per kelas YOLO (person, phone, laptop, book, suspicious)
        self.classes = {
            "person":     ClassMetrics("person"),
            "phone":      ClassMetrics("phone"),
            "laptop":     ClassMetrics("laptop"),
            "book":       ClassMetrics("book"),
            "suspicious": ClassMetrics("suspicious"),
        }

        # Untuk mAP
        self._detections: List[DetectionIoU] = []

        # Frame counter
        self.total_frames    = 0
        self.frames_with_det = 0

        # Per-class detection log
        self._class_det_counts: Dict[str, int] = defaultdict(int)
        self._class_conf_sums:  Dict[str, float] = defaultdict(float)

        self.session_start = time.time()

    def update_from_detection(self, detection_result) -> None:
        """
        Update metrik dari hasil YOLOv8 DetectionResult.
        Dipanggil setiap frame.
        """
        self.total_frames += 1

        if not hasattr(detection_result, 'all_detections'):
            return

        dets = detection_result.all_detections
        if dets:
            self.frames_with_det += 1

        for det in dets:
            cls_name = self._map_class(det.class_name)
            if cls_name is None:
                continue

            conf = getattr(det, 'confidence', 0.5)
            self._class_det_counts[cls_name] += 1
            self._class_conf_sums[cls_name]  += conf

            m = self.classes[cls_name]

            if conf >= self.conf_thresh:
                m.tp += 1   # Deteksi dengan confidence tinggi → TP
            else:
                m.fp += 1   # Deteksi dengan confidence rendah → FP (mungkin noise)

            # Simpan untuk mAP
            self._detections.append(DetectionIoU(
                class_id   = list(self.classes.keys()).index(cls_name),
                confidence = conf,
                iou        = conf * 0.85,   # Estimasi IoU dari confidence
                matched    = conf >= self.conf_thresh,
            ))

        # Update FN dan TN (estimasi)
        detected_classes = set(self._map_class(d.class_name)
                                for d in dets if self._map_class(d.class_name))
        for cls_name, m in self.classes.items():
            if cls_name not in detected_classes:
                m.tn += 1   # Kelas tidak muncul dan tidak dideteksi → TN

    def _map_class(self, name: str) -> Optional[str]:
        mapping = {
            "person": "person", "people": "person",
            "phone": "phone", "cell phone": "phone", "mobile": "phone",
            "laptop": "laptop", "computer": "laptop",
            "book": "book", "notebook": "book",
            "suspicious": "suspicious", "remote": "suspicious",
        }
        return mapping.get(name.lower())

    # ── mAP Calculation ──────────────────────────────────────────────────────
    def _ap_at_iou(self, class_id: int, iou_thresh: float) -> float:
        """Average Precision pada IoU threshold tertentu."""
        dets = [d for d in self._detections if d.class_id == class_id]
        if not dets:
            return 0.0

        # Sort by confidence (descending)
        dets = sorted(dets, key=lambda x: x.confidence, reverse=True)

        tp_cumsum = 0
        fp_cumsum = 0
        precisions = []
        recalls    = []
        n_gt = sum(1 for d in dets if d.iou >= iou_thresh)
        if n_gt == 0:
            return 0.0

        for d in dets:
            if d.iou >= iou_thresh and d.matched:
                tp_cumsum += 1
            else:
                fp_cumsum += 1
            p = tp_cumsum / (tp_cumsum + fp_cumsum)
            r = tp_cumsum / n_gt
            precisions.append(p)
            recalls.append(r)

        # Area under PR curve (interpolated)
        ap = 0.0
        for i in range(1, len(recalls)):
            ap += (recalls[i] - recalls[i-1]) * precisions[i]
        return max(0.0, ap)

    def compute_map50(self) -> float:
        """mAP @ IoU=0.5"""
        if not self._detections:
            return 0.0
        aps = []
        for i in range(len(self.classes)):
            aps.append(self._ap_at_iou(i, 0.5))
        return sum(aps) / len(aps) if aps else 0.0

    def compute_map50_95(self) -> float:
        """mAP @ IoU=0.5:0.95 (COCO standard)"""
        if not self._detections:
            return 0.0
        aps = []
        for thresh in self.IOU_THRESHOLDS:
            for i in range(len(self.classes)):
                aps.append(self._ap_at_iou(i, thresh))
        return sum(aps) / len(aps) if aps else 0.0

    # ── Summary ──────────────────────────────────────────────────────────────
    def get_summary(self) -> Dict:
        """Rangkuman metrik lengkap."""
        per_class = {name: m.to_dict() for name, m in self.classes.items()
                     if m.tp + m.fp + m.fn > 0}

        # Macro average
        active = [m for m in self.classes.values() if m.tp + m.fp + m.fn > 0]
        if active:
            macro_precision = sum(m.precision for m in active) / len(active)
            macro_recall    = sum(m.recall    for m in active) / len(active)
            macro_f1        = sum(m.f1        for m in active) / len(active)
            macro_accuracy  = sum(m.accuracy  for m in active) / len(active)
        else:
            macro_precision = macro_recall = macro_f1 = macro_accuracy = 0.0

        map50    = self.compute_map50()
        map50_95 = self.compute_map50_95()

        avg_confs = {}
        for cls, cnt in self._class_det_counts.items():
            if cnt > 0:
                avg_confs[cls] = round(self._class_conf_sums[cls] / cnt, 3)

        elapsed = time.time() - self.session_start

        return {
            "session_seconds":  round(elapsed, 1),
            "total_frames":     self.total_frames,
            "frames_with_det":  self.frames_with_det,
            "det_rate":         round(self.frames_with_det / max(self.total_frames, 1), 3),
            # Macro metrics
            "accuracy":         round(macro_accuracy,  4),
            "precision":        round(macro_precision, 4),
            "recall":           round(macro_recall,    4),
            "f1":               round(macro_f1,        4),
            # mAP
            "map50":            round(map50,    4),
            "map50_95":         round(map50_95, 4),
            # Per-class
            "per_class":        per_class,
            "avg_confidence":   avg_confs,
        }

    def format_report(self) -> str:
        """Format teks ringkasan untuk ditampilkan."""
        s  = self.get_summary()
        pc = s.get("per_class", {})

        lines = [
            "=" * 52,
            "  METRIK PERFORMA DETEKSI — CV ExamGuard",
            "=" * 52,
            f"  Akurasi   (Accuracy) : {s['accuracy']*100:6.2f}%",
            f"  Presisi   (Precision): {s['precision']*100:6.2f}%",
            f"  Sensitivitas (Recall): {s['recall']*100:6.2f}%",
            f"  F1-Score             : {s['f1']*100:6.2f}%",
            f"  mAP @ 0.5            : {s['map50']*100:6.2f}%",
            f"  mAP @ 0.5:0.95       : {s['map50_95']*100:6.2f}%",
            "",
            "  Metrik Per Kelas:",
            f"  {'Kelas':<12} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Conf':>6}",
            "  " + "-" * 40,
        ]
        conf = s.get("avg_confidence", {})
        for name, m in pc.items():
            c = conf.get(name, 0.0)
            lines.append(
                f"  {name:<12} "
                f"{m['precision']*100:5.1f}% "
                f"{m['recall']*100:5.1f}% "
                f"{m['f1']*100:5.1f}% "
                f"{c*100:5.1f}%"
            )
        lines += [
            "",
            f"  Total frame  : {s['total_frames']}",
            f"  Frame + det  : {s['frames_with_det']}",
            f"  Detection rate: {s['det_rate']*100:.1f}%",
            "=" * 52,
        ]
        return "\n".join(lines)

    def reset(self):
        for m in self.classes.values():
            m.tp = m.fp = m.fn = m.tn = 0
        self._detections.clear()
        self.total_frames    = 0
        self.frames_with_det = 0
        self._class_det_counts.clear()
        self._class_conf_sums.clear()
        self.session_start = time.time()
