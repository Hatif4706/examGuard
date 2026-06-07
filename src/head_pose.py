"""
Head Pose Estimator - FIXED VERSION
Perbaikan:
- Angle calculation lebih akurat dan stabil
- EMA smoothing untuk reduce jitter
- min_detection_confidence lebih rendah (0.4)
- Sign convention yang konsisten
"""

import cv2
import numpy as np
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from collections import deque
import logging

logger = logging.getLogger(__name__)

try:
    import mediapipe as mp
    mp_face_mesh     = mp.solutions.face_mesh
    mp_drawing       = mp.solutions.drawing_utils
    mp_drawing_styles= mp.solutions.drawing_styles
except Exception as e:
    raise ImportError(f"MediaPipe gagal diimport: {e}\nJalankan: pip install mediapipe==0.10.5")


@dataclass
class HeadPose:
    face_id:        int
    yaw:            float      # + = kiri, - = kanan
    pitch:          float      # + = bawah, - = atas
    roll:           float
    looking_left:   bool = False
    looking_right:  bool = False
    looking_down:   bool = False
    looking_up:     bool = False
    looking_straight: bool = True
    confidence:     float = 1.0
    landmarks_2d:   Optional[np.ndarray] = None
    nose_tip:       Optional[Tuple[int,int]] = None
    face_bbox:      Optional[Tuple[int,int,int,int]] = None

    def get_direction_label(self) -> str:
        if self.looking_straight:
            return "Melihat ke Depan"
        parts = []
        if self.looking_left:  parts.append("Kiri")
        if self.looking_right: parts.append("Kanan")
        if self.looking_down:  parts.append("Bawah")
        if self.looking_up:    parts.append("Atas")
        return "Melihat ke " + " & ".join(parts) if parts else "Tidak Diketahui"

    def is_suspicious(self, yaw_thresh=20.0, pitch_thresh=15.0) -> bool:
        return (abs(self.yaw) > yaw_thresh or
                self.pitch > pitch_thresh or
                not self.looking_straight)


class AngleSmootherEMA:
    """Exponential Moving Average untuk stabilkan angle agar tidak jitter."""
    def __init__(self, alpha: float = 0.35):
        self.alpha = alpha      # 0.2=very smooth, 0.5=responsive
        self._values: Dict[int, Dict[str, float]] = {}

    def smooth(self, face_id: int, yaw: float, pitch: float, roll: float):
        if face_id not in self._values:
            self._values[face_id] = {"yaw": yaw, "pitch": pitch, "roll": roll}
        else:
            a = self.alpha
            self._values[face_id]["yaw"]   = a*yaw   + (1-a)*self._values[face_id]["yaw"]
            self._values[face_id]["pitch"] = a*pitch + (1-a)*self._values[face_id]["pitch"]
            self._values[face_id]["roll"]  = a*roll  + (1-a)*self._values[face_id]["roll"]
        return (self._values[face_id]["yaw"],
                self._values[face_id]["pitch"],
                self._values[face_id]["roll"])

    def remove(self, face_id: int):
        self._values.pop(face_id, None)

    def clear(self):
        self._values.clear()


class HeadPoseEstimator:
    """
    Estimasi pose kepala menggunakan MediaPipe Face Mesh + PnP Solver.
    Versi FIXED: angle calculation lebih stabil, smoothing, thresholds lebih sensitif.
    """

    # 6 titik referensi 3D wajah (standar dari literatur)
    FACE_3D = np.array([
        [ 0.0,     0.0,    0.0   ],  # 1:   hidung
        [ 0.0,  -330.0,  -65.0  ],  # 152: dagu
        [-225.0,  170.0, -135.0 ],  # 33:  sudut mata kiri
        [ 225.0,  170.0, -135.0 ],  # 263: sudut mata kanan
        [-150.0, -150.0, -125.0 ],  # 78:  mulut kiri
        [ 150.0, -150.0, -125.0 ],  # 308: mulut kanan
    ], dtype=np.float64)

    LANDMARK_INDICES = [1, 152, 33, 263, 78, 308]

    FACE_OVAL_INDICES = [
        10,338,297,332,284,251,389,356,454,
        323,361,288,397,365,379,378,400,377,
        152,148,176,149,150,136,172, 58,132,
         93,234,127,162, 21, 54,103, 67,109
    ]

    def __init__(self,
                 yaw_threshold:   float = 20.0,
                 pitch_threshold: float = 15.0,
                 roll_threshold:  float = 20.0,
                 max_faces:       int   = 4,
                 min_detection_confidence: float = 0.4,
                 min_tracking_confidence:  float = 0.4):

        self.yaw_threshold   = yaw_threshold
        self.pitch_threshold = pitch_threshold
        self.roll_threshold  = roll_threshold

        self.smoother = AngleSmootherEMA(alpha=0.35)

        try:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=max_faces,
                refine_landmarks=True,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence
            )
            logger.info(
                f"HeadPoseEstimator siap | yaw±{yaw_threshold}° pitch±{pitch_threshold}°"
            )
        except Exception as e:
            logger.error(f"Gagal inisialisasi FaceMesh: {e}")
            raise

    # ── Camera Matrix ────────────────────────────────────────────────────────
    def _camera_matrix(self, shape: Tuple):
        h, w = shape[:2]
        fl = float(w)       # focal length estimasi = lebar frame
        cx, cy = w / 2.0, h / 2.0
        K = np.array([[fl, 0, cx],
                      [ 0,fl, cy],
                      [ 0, 0,  1]], dtype=np.float64)
        return K, np.zeros((4,1), dtype=np.float64)

    # ── PnP → Euler angles ───────────────────────────────────────────────────
    def _solve_angles(self, pts2d: np.ndarray, K: np.ndarray, D: np.ndarray
                      ) -> Optional[Tuple[float,float,float]]:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                self.FACE_3D, pts2d, K, D,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not ok:
                return None

            R, _ = cv2.Rodrigues(rvec)

            # ── Ekstrak Euler (ZYX convention, degrees) ──────────────
            # pitch (up/down) = rotation around X
            # yaw   (left/right) = rotation around Y
            # roll  (tilt) = rotation around Z
            pitch_rad = math.atan2(-R[2,0], math.sqrt(R[2,1]**2 + R[2,2]**2))
            yaw_rad   = math.atan2( R[1,0], R[0,0])
            roll_rad  = math.atan2( R[2,1], R[2,2])

            pitch_deg = math.degrees(pitch_rad)
            yaw_deg   = math.degrees(yaw_rad)
            roll_deg  = math.degrees(roll_rad)

            # ── Koreksi sign convention ──────────────────────────────
            # Dari percobaan: yaw > 0 = hidung ke kanan frame = orang melihat KIRI
            #                 yaw < 0 = hidung ke kiri frame  = orang melihat KANAN
            # pitch > 0 = muka ke bawah
            # Ini sudah benar, tapi tergantung koordinat MediaPipe
            # Flip yaw agar positif = kiri (sesuai konvensi kita)
            yaw_deg   = -yaw_deg    # flip: + = kiri, - = kanan

            return float(yaw_deg), float(pitch_deg), float(roll_deg)

        except Exception as e:
            logger.debug(f"solvePnP error: {e}")
            return None

    # ── Face BBox ────────────────────────────────────────────────────────────
    def _face_bbox(self, lms, shape: Tuple) -> Tuple[int,int,int,int]:
        h, w = shape[:2]
        xs = [lms[i].x*w for i in self.FACE_OVAL_INDICES if i < len(lms)]
        ys = [lms[i].y*h for i in self.FACE_OVAL_INDICES if i < len(lms)]
        if not xs:
            return (0, 0, w, h)
        return (max(0,int(min(xs))-8), max(0,int(min(ys))-8),
                min(w,int(max(xs))+8), min(h,int(max(ys))+8))

    # ── Estimate ─────────────────────────────────────────────────────────────
    def estimate(self, frame: np.ndarray) -> List[HeadPose]:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            res = self.face_mesh.process(rgb)
            rgb.flags.writeable = True

            if not res.multi_face_landmarks:
                return []

            h, w = frame.shape[:2]
            K, D = self._camera_matrix(frame.shape)
            poses = []

            for face_id, fl in enumerate(res.multi_face_landmarks):
                lms = fl.landmark

                pts2d = np.array([
                    [lms[i].x*w, lms[i].y*h]
                    for i in self.LANDMARK_INDICES
                ], dtype=np.float64)

                angles = self._solve_angles(pts2d, K, D)
                if angles is None:
                    continue

                raw_yaw, raw_pitch, raw_roll = angles

                # ── Smooth ─────────────────────────────────────────────
                yaw, pitch, roll = self.smoother.smooth(
                    face_id, raw_yaw, raw_pitch, raw_roll)

                # ── Klasifikasi arah ────────────────────────────────────
                looking_left   = yaw   >  self.yaw_threshold
                looking_right  = yaw   < -self.yaw_threshold
                looking_down   = pitch >  self.pitch_threshold
                looking_up     = pitch < -self.pitch_threshold
                looking_straight = not any([looking_left, looking_right,
                                            looking_down, looking_up])

                # Nose tip
                ni = self.LANDMARK_INDICES[0]
                nose = (int(lms[ni].x*w), int(lms[ni].y*h))

                # Landmark 2D
                lm2d = np.array([[int(l.x*w), int(l.y*h)]
                                  for l in lms], dtype=np.int32)

                poses.append(HeadPose(
                    face_id=face_id,
                    yaw=yaw, pitch=pitch, roll=roll,
                    looking_left=looking_left,
                    looking_right=looking_right,
                    looking_down=looking_down,
                    looking_up=looking_up,
                    looking_straight=looking_straight,
                    landmarks_2d=lm2d,
                    nose_tip=nose,
                    face_bbox=self._face_bbox(lms, frame.shape)
                ))

            return poses

        except Exception as e:
            logger.error(f"estimate error: {e}")
            return []

    # ── Draw ─────────────────────────────────────────────────────────────────
    def draw_head_direction(self, frame: np.ndarray, hp: HeadPose,
                            draw_landmarks: bool = False) -> np.ndarray:
        try:
            if hp.nose_tip is None:
                return frame

            nx, ny = hp.nose_tip

            # Warna per status
            if hp.looking_straight:
                color  = (40, 210, 40)       # Hijau terang
                status = "DEPAN - AMAN"
            elif hp.looking_left:
                color  = (40, 40, 230)       # Merah
                status = "MENOLEH KIRI ←"
            elif hp.looking_right:
                color  = (40, 40, 230)       # Merah
                status = "MENOLEH KANAN →"
            elif hp.looking_down:
                color  = (30, 130, 255)      # Orange
                status = "MENUNDUK ↓"
            else:
                color  = (60, 190, 60)
                status = "MELIHAT ATAS ↑"

            # Panah arah pandangan
            arrow_len = 90
            dx = int(-hp.yaw   * arrow_len / 50)
            dy = int( hp.pitch * arrow_len / 50)
            dx = max(-arrow_len, min(arrow_len, dx))
            dy = max(-arrow_len, min(arrow_len, dy))

            cv2.arrowedLine(frame, (nx, ny), (nx+dx, ny+dy),
                            color, 3, tipLength=0.35)
            cv2.circle(frame, (nx, ny), 6, color, -1)
            cv2.circle(frame, (nx, ny), 6, (255,255,255), 1)

            # BBox wajah
            if hp.face_bbox:
                x1, y1, x2, y2 = hp.face_bbox
                thickness = 3 if not hp.looking_straight else 2
                cv2.rectangle(frame, (x1,y1), (x2,y2), color, thickness)

                # Label background
                lbl_h = 44
                ly = max(y1 - lbl_h, 0)
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, ly), (x2, y1), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

                # Teks status
                cv2.putText(frame, status, (x1+4, ly+14),
                            cv2.FONT_HERSHEY_DUPLEX, 0.45, color, 1, cv2.LINE_AA)
                # Sudut
                angle_txt = f"Yaw:{hp.yaw:+.0f}  Pitch:{hp.pitch:+.0f}"
                cv2.putText(frame, angle_txt, (x1+4, ly+30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180,180,180), 1, cv2.LINE_AA)

            # Landmark opsional
            if draw_landmarks and hp.landmarks_2d is not None:
                for i in self.LANDMARK_INDICES:
                    if i < len(hp.landmarks_2d):
                        cv2.circle(frame, tuple(hp.landmarks_2d[i]),
                                   3, (0,220,220), -1)

            return frame
        except Exception as e:
            logger.error(f"draw error: {e}")
            return frame

    def close(self):
        try:
            self.face_mesh.close()
            self.smoother.clear()
        except:
            pass
