"""
Behavior Analyzer v2.3
Perbaikan:
- Kecurangan menoleh: 3 detik (bukan 5)
- Session-based tracking untuk menoleh kiri/kanan/menunduk (sama seperti HP)
- Pitch offset +8° agar tidak false positive saat duduk tegak
"""

import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum

from .head_pose import HeadPose
from .object_detector import DetectionResult

logger = logging.getLogger(__name__)


# ─────────────── ENUM ───────────────────────────────────────────────────────
class CheatType(Enum):
    NORMAL             = "Normal"
    LOOKING_LEFT       = "Menoleh ke Kiri"
    LOOKING_RIGHT      = "Menoleh ke Kanan"
    HEAD_DOWN          = "Kepala Menunduk"
    PHONE_DETECTED     = "Menggunakan HP"
    MULTIPLE_PERSONS   = "Banyak Orang"
    SUSPICIOUS_OBJECT  = "Benda Mencurigakan"
    EXTENDED_GAZE_AWAY = "Pandangan Menjauh > 3 Detik"
    NO_PERSON          = "Peserta Tidak Terdeteksi"
    HANDS_SUSPICIOUS   = "Gerakan Tangan"


class AlertLevel(Enum):
    NONE     = 0
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4

    def label(self) -> str:
        return {0:"AMAN", 1:"WASPADA", 2:"MENCURIGAKAN",
                3:"PELANGGARAN", 4:"KECURANGAN!"}.get(self.value, "?")

    def color_bgr(self) -> Tuple[int,int,int]:
        return {0:(40,210,40), 1:(40,215,215), 2:(40,140,255),
                3:(40,40,220), 4:(20,20,150)}.get(self.value, (128,128,128))


# ─────────────── DATACLASS ──────────────────────────────────────────────────
@dataclass
class BehaviorEvent:
    timestamp:   float
    cheat_type:  CheatType
    alert_level: AlertLevel
    description: str
    cheat_score: int
    face_id:     Optional[int] = None
    extra_data:  Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp":   self.timestamp,
            "time_str":    time.strftime("%H:%M:%S", time.localtime(self.timestamp)),
            "type":        self.cheat_type.value,
            "level":       self.alert_level.label(),
            "description": self.description,
            "score":       self.cheat_score,
        }


@dataclass
class FrameAnalysis:
    timestamp:        float
    alert_level:      AlertLevel
    cheat_score:      int
    detected_cheats:  List[CheatType]
    events:           List[BehaviorEvent]
    person_count:     int
    has_phone:        bool
    head_poses:       List[HeadPose]
    detection_result: DetectionResult
    frame_number:     int   = 0
    face_count:       int   = 0

    # HP session
    phone_session_count:    int   = 0
    phone_session_duration: float = 0.0
    phone_session_active:   bool  = False

    # Gaze kiri session
    left_session_count:    int   = 0
    left_session_duration: float = 0.0
    left_session_active:   bool  = False

    # Gaze kanan session
    right_session_count:    int   = 0
    right_session_duration: float = 0.0
    right_session_active:   bool  = False

    # Gaze menunduk session
    down_session_count:    int   = 0
    down_session_duration: float = 0.0
    down_session_active:   bool  = False

    @property
    def is_cheating(self) -> bool:
        return self.alert_level.value >= AlertLevel.MEDIUM.value

    @property
    def primary_cheat(self) -> Optional[CheatType]:
        priority = [
            CheatType.PHONE_DETECTED, CheatType.MULTIPLE_PERSONS,
            CheatType.EXTENDED_GAZE_AWAY, CheatType.LOOKING_LEFT,
            CheatType.LOOKING_RIGHT, CheatType.HEAD_DOWN,
            CheatType.SUSPICIOUS_OBJECT, CheatType.NO_PERSON,
        ]
        for c in priority:
            if c in self.detected_cheats:
                return c
        return self.detected_cheats[0] if self.detected_cheats else None


# ─────────────── SESSION TRACKER (GENERIK) ──────────────────────────────────
class SessionTracker:
    """
    Session-based tracker — sama seperti HP:
    - Satu kali aktif berurutan = 1 sesi = hitung 1x
    - Jika hilang lalu muncul lagi setelah grace period = sesi baru
    - Tampilkan durasi per sesi dan total semua sesi
    """

    def __init__(self, name: str, grace_period: float = 1.0):
        self.name           = name
        self.grace_period   = grace_period   # detik toleransi oklusi/kembali normal

        self.session_count:    int   = 0
        self.session_active:   bool  = False
        self.session_start:    float = 0.0
        self.session_duration: float = 0.0
        self.total_duration:   float = 0.0
        self._last_seen:       float = 0.0
        self._new_event:       bool  = False

    def update(self, is_active: bool, t: float) -> Dict:
        self._new_event = False
        event_type: Optional[str] = None

        if is_active:
            self._last_seen = t
            if not self.session_active:
                # Sesi BARU dimulai
                self.session_count  += 1
                self.session_active  = True
                self.session_start   = t
                self.session_duration = 0.0
                self._new_event = True
                event_type = "start"
                logger.debug(f"[{self.name}] session #{self.session_count} started")
            else:
                event_type = "ongoing"
            self.session_duration = t - self.session_start

        else:
            if self.session_active:
                gap = t - self._last_seen
                if gap > self.grace_period:
                    # Sesi BERAKHIR
                    self.session_duration  = max(0.0, self._last_seen - self.session_start)
                    self.total_duration   += self.session_duration
                    self.session_active    = False
                    self._new_event = True
                    event_type = "end"
                    logger.debug(
                        f"[{self.name}] session #{self.session_count} "
                        f"ended ({self.session_duration:.1f}s)"
                    )
                # Masih dalam grace period → sesi dianggap masih aktif

        return {
            "session_count":    self.session_count,
            "session_active":   self.session_active,
            "session_duration": self.session_duration,
            "total_duration":   self.total_duration,
            "new_event":        self._new_event,
            "event_type":       event_type,
        }

    def reset(self):
        self.session_count    = 0
        self.session_active   = False
        self.session_start    = 0.0
        self.session_duration = 0.0
        self.total_duration   = 0.0
        self._last_seen       = 0.0
        self._new_event       = False


# ─────────────── BEHAVIOR ANALYZER ──────────────────────────────────────────
class BehaviorAnalyzer:
    """
    Analisis perilaku kecurangan ujian — v2.3
    Session-based: menoleh kiri/kanan/menunduk/HP semuanya dihitung per sesi.
    """

    CHEAT_WEIGHTS = {
        CheatType.PHONE_DETECTED:     8,
        CheatType.EXTENDED_GAZE_AWAY: 7,
        CheatType.MULTIPLE_PERSONS:   5,
        CheatType.HEAD_DOWN:          4,
        CheatType.LOOKING_LEFT:       3,
        CheatType.LOOKING_RIGHT:      3,
        CheatType.SUSPICIOUS_OBJECT:  3,
        CheatType.NO_PERSON:          3,
        CheatType.HANDS_SUSPICIOUS:   2,
    }

    ALERT_THRESHOLDS = {
        AlertLevel.LOW:      3,
        AlertLevel.MEDIUM:   6,
        AlertLevel.HIGH:     10,
        AlertLevel.CRITICAL: 15,
    }

    # Durasi menoleh sebelum dianggap kecurangan (detik)
    GAZE_CHEAT_SECONDS = 3.0

    def __init__(self,
                 gaze_away_duration:       float = 3.0,
                 phone_duration:           float = 3.0,
                 multiple_person_duration: float = 2.0,
                 head_down_duration:       float = 3.0,
                 yaw_threshold:            float = 20.0,
                 pitch_threshold:          float = 22.0,
                 pitch_offset:             float = 8.0):
        """
        pitch_offset: offset kalibrasi pitch (tambahkan agar tidak false positive
                      saat duduk tegak di depan monitor). Default 8°.
        """

        self.gaze_away_duration       = gaze_away_duration
        self.multiple_person_duration = multiple_person_duration
        self.head_down_duration       = head_down_duration
        self.yaw_threshold            = yaw_threshold
        self.pitch_threshold          = pitch_threshold
        self.pitch_offset             = pitch_offset  # kalibrasi posisi duduk

        # Session trackers — grace period 1 detik untuk semua arah
        self.phone_tracker = SessionTracker("phone", grace_period=2.0)
        self.left_tracker  = SessionTracker("look_left",  grace_period=1.0)
        self.right_tracker = SessionTracker("look_right", grace_period=1.0)
        self.down_tracker  = SessionTracker("head_down",  grace_period=1.0)

        # Multi-person (pakai simple tracker)
        self._multi_start:    Optional[float] = None
        self._multi_duration: float = 0.0

        # Face absence
        self._last_face_time: float = time.time()

        # Session stats
        self.event_history:     List[BehaviorEvent] = []
        self.total_cheat_score: int   = 0
        self.frame_count:       int   = 0
        self.session_start:     float = time.time()
        self.cheat_counts:      Dict[CheatType, int] = {c: 0 for c in CheatType}

        # Audio trigger (diambil oleh alert_manager)
        self.trigger_audio_for: Optional[CheatType] = None

        logger.info(
            f"BehaviorAnalyzer v2.3 | yaw±{yaw_threshold}° | "
            f"pitch>{pitch_threshold}°+{pitch_offset}° | "
            f"gaze_cheat={self.GAZE_CHEAT_SECONDS}s"
        )

    def _score_to_level(self, score: int) -> AlertLevel:
        if score >= self.ALERT_THRESHOLDS[AlertLevel.CRITICAL]: return AlertLevel.CRITICAL
        if score >= self.ALERT_THRESHOLDS[AlertLevel.HIGH]:     return AlertLevel.HIGH
        if score >= self.ALERT_THRESHOLDS[AlertLevel.MEDIUM]:   return AlertLevel.MEDIUM
        if score >= self.ALERT_THRESHOLDS[AlertLevel.LOW]:      return AlertLevel.LOW
        return AlertLevel.NONE

    # ── Bantu buat event untuk sesi gaze ─────────────────────────────────────
    def _gaze_event(self, t: float, cheat: CheatType, sess: Dict,
                    angle_val: float, angle_name: str,
                    face_id: Optional[int]) -> BehaviorEvent:
        dur = sess["session_duration"]
        cnt = sess["session_count"]
        evt = sess["event_type"]

        if evt == "start":
            desc = (f"{cheat.value} dimulai (ke-{cnt}x)  "
                    f"{angle_name}:{angle_val:+.0f}°")
            lvl = AlertLevel.LOW
        elif evt == "end":
            desc = (f"{cheat.value} selesai (ke-{cnt}x) — "
                    f"durasi {dur:.1f}s | total {sess['total_duration']:.1f}s")
            lvl = AlertLevel.HIGH if dur >= self.GAZE_CHEAT_SECONDS else AlertLevel.LOW
        else:  # milestone tiap 3 detik setelah threshold
            desc = (f"{cheat.value} ke-{cnt}x — {dur:.0f}s berlangsung "
                    f"{angle_name}:{angle_val:+.0f}°")
            lvl = AlertLevel.HIGH

        return BehaviorEvent(
            timestamp=t, cheat_type=cheat,
            alert_level=lvl, description=desc,
            cheat_score=self.CHEAT_WEIGHTS.get(cheat, 3),
            face_id=face_id,
            extra_data={"session": cnt, "duration": dur, "type": evt}
        )

    # ── HEAD POSE ANALYSIS ────────────────────────────────────────────────────
    def _analyze_head(self, poses: List[HeadPose], t: float
                      ) -> Tuple[List[CheatType], List[BehaviorEvent], int]:
        cheats, events = [], []
        score = 0

        if poses:
            self._last_face_time = t

        # Wajah tidak terlihat ≥ 5 detik
        absent = t - self._last_face_time
        if absent >= 5.0:
            cheats.append(CheatType.NO_PERSON)
            score += self.CHEAT_WEIGHTS[CheatType.NO_PERSON]
            if int(absent) % 5 == 0 and absent % 1.0 < 0.35:
                events.append(BehaviorEvent(
                    timestamp=t, cheat_type=CheatType.NO_PERSON,
                    alert_level=AlertLevel.HIGH,
                    description=f"Peserta tidak terlihat {absent:.0f}s",
                    cheat_score=self.CHEAT_WEIGHTS[CheatType.NO_PERSON],
                    extra_data={"duration": absent}
                ))
            # Reset semua gaze session saat wajah hilang
            self.left_tracker.update(False, t)
            self.right_tracker.update(False, t)
            self.down_tracker.update(False, t)
            return cheats, events, score

        if not poses:
            return cheats, events, score

        p = poses[0]

        # ── Koreksi pitch dengan offset kalibrasi ──────────────────────
        # pitch_corr: pitch sesungguhnya setelah dikurangi offset posisi duduk
        pitch_corr = p.pitch - self.pitch_offset

        looking_down_corrected = pitch_corr > self.pitch_threshold

        # ── MELIHAT LURUS → reset semua, alert hilang ──────────────────
        if p.looking_straight and not looking_down_corrected:
            # Trigger session "end" jika sebelumnya aktif
            ls = self.left_tracker.update(False, t)
            rs = self.right_tracker.update(False, t)
            ds = self.down_tracker.update(False, t)

            # Log penutup sesi jika ada
            for sess, cheat, av, an in [
                (ls, CheatType.LOOKING_LEFT,  p.yaw,   "Yaw"),
                (rs, CheatType.LOOKING_RIGHT, p.yaw,   "Yaw"),
                (ds, CheatType.HEAD_DOWN,     p.pitch, "Pitch"),
            ]:
                if sess["new_event"] and sess["event_type"] == "end":
                    events.append(
                        self._gaze_event(t, cheat, sess, av, an, p.face_id)
                    )
            return cheats, events, score

        # ── MENOLEH KIRI ──────────────────────────────────────────────
        if p.looking_left:
            self.right_tracker.update(False, t)
            self.down_tracker.update(False, t)
            ls = self.left_tracker.update(True, t)
            dur = ls["session_duration"]

            cheats.append(CheatType.LOOKING_LEFT)
            score += self.CHEAT_WEIGHTS[CheatType.LOOKING_LEFT]

            if dur >= self.GAZE_CHEAT_SECONDS:
                cheats.append(CheatType.EXTENDED_GAZE_AWAY)
                score += self.CHEAT_WEIGHTS[CheatType.EXTENDED_GAZE_AWAY]
                self.trigger_audio_for = CheatType.EXTENDED_GAZE_AWAY

            # Log: saat mulai + tiap 3 detik setelah threshold
            if ls["new_event"] or (dur >= self.GAZE_CHEAT_SECONDS and
                                    (dur - self.GAZE_CHEAT_SECONDS) % 3.0 < 0.35):
                events.append(
                    self._gaze_event(t, CheatType.LOOKING_LEFT, ls, p.yaw, "Yaw", p.face_id)
                )

        # ── MENOLEH KANAN ─────────────────────────────────────────────
        elif p.looking_right:
            self.left_tracker.update(False, t)
            self.down_tracker.update(False, t)
            rs = self.right_tracker.update(True, t)
            dur = rs["session_duration"]

            cheats.append(CheatType.LOOKING_RIGHT)
            score += self.CHEAT_WEIGHTS[CheatType.LOOKING_RIGHT]

            if dur >= self.GAZE_CHEAT_SECONDS:
                cheats.append(CheatType.EXTENDED_GAZE_AWAY)
                score += self.CHEAT_WEIGHTS[CheatType.EXTENDED_GAZE_AWAY]
                self.trigger_audio_for = CheatType.EXTENDED_GAZE_AWAY

            if rs["new_event"] or (dur >= self.GAZE_CHEAT_SECONDS and
                                    (dur - self.GAZE_CHEAT_SECONDS) % 3.0 < 0.35):
                events.append(
                    self._gaze_event(t, CheatType.LOOKING_RIGHT, rs, p.yaw, "Yaw", p.face_id)
                )

        else:
            self.left_tracker.update(False, t)
            self.right_tracker.update(False, t)

        # ── KEPALA MENUNDUK (dengan pitch offset) ──────────────────────
        ds = self.down_tracker.update(looking_down_corrected, t)
        if looking_down_corrected:
            dur = ds["session_duration"]
            if dur >= 0.5:   # grace 0.5 detik sebelum dianggap menunduk
                cheats.append(CheatType.HEAD_DOWN)
                score += self.CHEAT_WEIGHTS[CheatType.HEAD_DOWN]

                if dur >= self.GAZE_CHEAT_SECONDS:
                    self.trigger_audio_for = CheatType.HEAD_DOWN

                if ds["new_event"] or (dur >= self.GAZE_CHEAT_SECONDS and
                                        (dur - self.GAZE_CHEAT_SECONDS) % 3.0 < 0.35):
                    events.append(
                        self._gaze_event(t, CheatType.HEAD_DOWN, ds,
                                         pitch_corr, "Pitch", p.face_id)
                    )
        else:
            if ds["new_event"] and ds["event_type"] == "end":
                events.append(
                    self._gaze_event(t, CheatType.HEAD_DOWN, ds, p.pitch, "Pitch", p.face_id)
                )

        return cheats, events, score

    # ── OBJECT ANALYSIS ───────────────────────────────────────────────────────
    def _analyze_objects(self, det: DetectionResult, t: float
                         ) -> Tuple[List[CheatType], List[BehaviorEvent], int]:
        cheats, events = [], []
        score = 0

        # ── HP — session-based ────────────────────────────────────────
        ph = self.phone_tracker.update(det.has_phone, t)

        if ph["session_active"]:
            cheats.append(CheatType.PHONE_DETECTED)
            score += self.CHEAT_WEIGHTS[CheatType.PHONE_DETECTED]

        if ph["new_event"]:
            cnt = ph["session_count"]
            if ph["event_type"] == "start":
                self.trigger_audio_for = CheatType.PHONE_DETECTED
                events.append(BehaviorEvent(
                    timestamp=t, cheat_type=CheatType.PHONE_DETECTED,
                    alert_level=AlertLevel.CRITICAL,
                    description=f"HP TERDETEKSI! (ke-{cnt}x) — memantau durasi...",
                    cheat_score=self.CHEAT_WEIGHTS[CheatType.PHONE_DETECTED],
                    extra_data={"session": cnt, "type": "start"}
                ))
            elif ph["event_type"] == "end":
                events.append(BehaviorEvent(
                    timestamp=t, cheat_type=CheatType.PHONE_DETECTED,
                    alert_level=AlertLevel.CRITICAL,
                    description=(
                        f"HP sesi ke-{cnt}x selesai — "
                        f"durasi {ph['session_duration']:.1f}s | "
                        f"total {ph['total_duration']:.1f}s"
                    ),
                    cheat_score=self.CHEAT_WEIGHTS[CheatType.PHONE_DETECTED],
                    extra_data={"session": cnt,
                                "duration": ph["session_duration"],
                                "total": ph["total_duration"], "type": "end"}
                ))

        # ── Banyak orang ──────────────────────────────────────────────
        if det.has_multiple_persons:
            if self._multi_start is None:
                self._multi_start = t
            self._multi_duration = t - self._multi_start
            if self._multi_duration >= self.multiple_person_duration:
                cheats.append(CheatType.MULTIPLE_PERSONS)
                score += self.CHEAT_WEIGHTS[CheatType.MULTIPLE_PERSONS]
                if self._multi_duration % 5.0 < 0.35:
                    events.append(BehaviorEvent(
                        timestamp=t, cheat_type=CheatType.MULTIPLE_PERSONS,
                        alert_level=AlertLevel.HIGH,
                        description=f"Terdeteksi {det.person_count} orang dalam frame!",
                        cheat_score=self.CHEAT_WEIGHTS[CheatType.MULTIPLE_PERSONS],
                        extra_data={"count": det.person_count}
                    ))
        else:
            self._multi_start = None
            self._multi_duration = 0.0

        # ── Benda mencurigakan ────────────────────────────────────────
        if det.has_suspicious_object:
            cheats.append(CheatType.SUSPICIOUS_OBJECT)
            score += self.CHEAT_WEIGHTS[CheatType.SUSPICIOUS_OBJECT]

        return cheats, events, score

    # ── MAIN ANALYZE ─────────────────────────────────────────────────────────
    def analyze(self,
                head_poses:       List[HeadPose],
                detection_result: DetectionResult,
                frame_number:     int = 0) -> FrameAnalysis:

        t = time.time()
        self.frame_count += 1
        self.trigger_audio_for = None

        h_cheats, h_events, h_score = self._analyze_head(head_poses, t)
        o_cheats, o_events, o_score = self._analyze_objects(detection_result, t)

        all_cheats = list(set(h_cheats + o_cheats))
        all_events = h_events + o_events
        total      = h_score + o_score

        alert_level = self._score_to_level(total)
        self.total_cheat_score += total
        for c in all_cheats:
            self.cheat_counts[c] = self.cheat_counts.get(c, 0) + 1
        self.event_history.extend(all_events)

        ph = self.phone_tracker
        lf = self.left_tracker
        rt = self.right_tracker
        dn = self.down_tracker

        return FrameAnalysis(
            timestamp=t,
            alert_level=alert_level,
            cheat_score=total,
            detected_cheats=all_cheats,
            events=all_events,
            person_count=detection_result.person_count,
            has_phone=detection_result.has_phone,
            head_poses=head_poses,
            detection_result=detection_result,
            frame_number=frame_number,
            face_count=len(head_poses),
            # HP
            phone_session_count=ph.session_count,
            phone_session_duration=ph.session_duration,
            phone_session_active=ph.session_active,
            # Kiri
            left_session_count=lf.session_count,
            left_session_duration=lf.session_duration,
            left_session_active=lf.session_active,
            # Kanan
            right_session_count=rt.session_count,
            right_session_duration=rt.session_duration,
            right_session_active=rt.session_active,
            # Menunduk
            down_session_count=dn.session_count,
            down_session_duration=dn.session_duration,
            down_session_active=dn.session_active,
        )

    def get_session_stats(self) -> Dict:
        elapsed = time.time() - self.session_start
        by_type: Dict[str, int] = {}
        for ev in self.event_history:
            by_type[ev.cheat_type.value] = by_type.get(ev.cheat_type.value, 0) + 1
        return {
            "session_duration":     elapsed,
            "total_frames":         self.frame_count,
            "total_events":         len(self.event_history),
            "total_cheat_score":    self.total_cheat_score,
            "events_by_type":       by_type,
            "phone_sessions":       self.phone_tracker.session_count,
            "phone_total_duration": self.phone_tracker.total_duration,
            "left_sessions":        self.left_tracker.session_count,
            "right_sessions":       self.right_tracker.session_count,
            "down_sessions":        self.down_tracker.session_count,
            "risk_level":           self._risk_level(),
        }

    def _risk_level(self) -> str:
        if self.frame_count == 0: return "Belum Ada Data"
        avg = self.total_cheat_score / self.frame_count
        ph  = self.phone_tracker.session_count
        if ph > 3 or avg > 6: return "SANGAT TINGGI"
        if ph > 1 or avg > 4: return "TINGGI"
        if avg > 2:            return "SEDANG"
        if avg > 0.5:          return "RENDAH"
        return "NORMAL"

    def reset(self):
        self.phone_tracker.reset()
        self.left_tracker.reset()
        self.right_tracker.reset()
        self.down_tracker.reset()
        self._multi_start    = None
        self._multi_duration = 0.0
        self.event_history.clear()
        self.total_cheat_score = 0
        self.frame_count       = 0
        self.session_start     = time.time()
        self.cheat_counts      = {c: 0 for c in CheatType}
        self._last_face_time   = time.time()
        self.trigger_audio_for = None
        logger.info("BehaviorAnalyzer di-reset")
