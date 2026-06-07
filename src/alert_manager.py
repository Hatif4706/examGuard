"""
Alert Manager v2.3 — FIXED
- Audio benar-benar berbunyi (winsound primary, pygame fallback)
- Tidak ada duplikasi kode
- Cooldown per cheat type
"""

import cv2
import os
import time
import logging
import threading
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field
from datetime import datetime

from .behavior_analyzer import AlertLevel, BehaviorEvent, FrameAnalysis, CheatType

logger = logging.getLogger(__name__)


@dataclass
class AlertConfig:
    visual:              bool  = True
    audio:               bool  = True
    log_to_file:         bool  = True
    screenshot_on_alert: bool  = True
    audio_volume:        float = 0.8
    cooldowns: Dict[str, float] = field(default_factory=lambda: {
        "none": 999.0, "low": 8.0, "medium": 5.0, "high": 3.0, "critical": 2.0
    })
    output_dir: str = "output"
    log_dir:    str = "logs"


# ─────────────── AUDIO ───────────────────────────────────────────────────────
class AudioAlert:
    """
    Audio alert — Windows winsound (prioritas) atau pygame.
    Pola beep berbeda per jenis kecurangan.
    """

    # Pola (frekuensi_hz, durasi_ms) per cheat type
    # 0 = jeda senyap
    PATTERNS: Dict[CheatType, list] = {
        CheatType.PHONE_DETECTED:     [(1800,150),(0,60),(1800,150),(0,60),(1800,150)],
        CheatType.EXTENDED_GAZE_AWAY: [(1200,250),(0,80),(1200,250),(0,80),(1500,300)],
        CheatType.LOOKING_LEFT:       [(900,200),(0,60),(1100,200)],
        CheatType.LOOKING_RIGHT:      [(900,200),(0,60),(1100,200)],
        CheatType.HEAD_DOWN:          [(750,350),(0,80),(750,350)],
        CheatType.MULTIPLE_PERSONS:   [(1000,180),(0,60),(1300,180),(0,60),(1600,180)],
        CheatType.NO_PERSON:          [(600,400),(0,100),(600,400)],
    }

    def __init__(self, volume: float = 0.8):
        self.volume    = volume
        self.enabled   = False
        self._backend  = None   # "winsound" | "pygame" | None
        self._pygame   = None
        self._winsound = None
        self._lock     = threading.Lock()
        self._init()

    def _init(self):
        # Coba winsound dulu (Windows, tidak perlu library tambahan)
        try:
            import winsound
            self._winsound = winsound
            self._backend  = "winsound"
            self.enabled   = True
            logger.info("Audio: winsound OK")
            return
        except ImportError:
            pass

        # Fallback: pygame
        try:
            import pygame
            import numpy as np
            pygame.mixer.pre_init(44100, -16, 1, 512)
            pygame.mixer.init()
            self._pygame  = pygame
            self._numpy   = np
            self._backend = "pygame"
            self.enabled  = True
            logger.info("Audio: pygame OK")
        except Exception as e:
            logger.warning(f"Audio tidak tersedia: {e}")

    # ── Buat gelombang sinus dengan pygame ───────────────────────────────────
    def _make_wave(self, freq: int, dur_ms: int):
        if self._pygame is None:
            return None
        try:
            np = self._numpy
            sr = 44100
            n  = int(sr * dur_ms / 1000)
            t  = np.linspace(0, dur_ms/1000, n, False)
            # Fade in/out 10ms untuk mengurangi klik
            fade = min(int(0.01 * sr), n // 4)
            env  = np.ones(n)
            env[:fade]  = np.linspace(0, 1, fade)
            env[-fade:] = np.linspace(1, 0, fade)
            wave = (np.sin(2 * np.pi * freq * t) * env * 32767 * self.volume).astype(np.int16)
            return self._pygame.sndarray.make_sound(wave)
        except Exception as e:
            logger.debug(f"wave error: {e}")
            return None

    # ── Play satu urutan beep (thread) ───────────────────────────────────────
    def _play_pattern(self, pattern: list):
        with self._lock:
            try:
                for freq, dur_ms in pattern:
                    if freq == 0:
                        time.sleep(dur_ms / 1000.0)
                        continue

                    if self._backend == "winsound":
                        self._winsound.Beep(
                            max(37, min(32767, freq)), dur_ms)

                    elif self._backend == "pygame":
                        snd = self._make_wave(freq, dur_ms)
                        if snd:
                            snd.play()
                            time.sleep(dur_ms / 1000.0 + 0.02)
            except Exception as e:
                logger.debug(f"play_pattern error: {e}")

    # ── Public API ────────────────────────────────────────────────────────────
    def play_cheat_alert(self, cheat_type: CheatType):
        """Bunyikan alarm sesuai jenis kecurangan."""
        if not self.enabled:
            return
        pattern = self.PATTERNS.get(cheat_type, [(1000, 300)])
        threading.Thread(
            target=self._play_pattern,
            args=(pattern,),
            daemon=True
        ).start()

    def test_beep(self):
        """Test satu beep pendek untuk verifikasi audio."""
        if not self.enabled:
            logger.warning("Audio tidak tersedia")
            return
        self._play_pattern([(1000, 300)])
        logger.info("Test beep berhasil")


# ─────────────── VISUAL ──────────────────────────────────────────────────────
class VisualAlert:
    """Overlay banner + border berkedip di atas video frame."""

    COLORS = {
        AlertLevel.NONE:     (40, 210,  40),
        AlertLevel.LOW:      (40, 215, 215),
        AlertLevel.MEDIUM:   (40, 140, 255),
        AlertLevel.HIGH:     (40,  40, 220),
        AlertLevel.CRITICAL: (20,  20, 150),
    }
    LABELS = {
        AlertLevel.NONE:     "AMAN",
        AlertLevel.LOW:      "WASPADA",
        AlertLevel.MEDIUM:   "MENCURIGAKAN",
        AlertLevel.HIGH:     "PELANGGARAN!",
        AlertLevel.CRITICAL: "KECURANGAN!",
    }

    def __init__(self):
        self._blink      = False
        self._last_blink = 0.0
        self._interval   = 0.3

    def _tick(self):
        now = time.time()
        if now - self._last_blink >= self._interval:
            self._blink      = not self._blink
            self._last_blink = now

    def draw_alert_banner(self, frame, analysis: FrameAnalysis):
        if analysis.alert_level == AlertLevel.NONE:
            return
        self._tick()
        color = self.COLORS[analysis.alert_level]
        h, w  = frame.shape[:2]

        overlay   = frame.copy()
        banner_h  = 56
        show_fill = True
        if analysis.alert_level.value >= AlertLevel.HIGH.value:
            show_fill = self._blink

        if show_fill:
            cv2.rectangle(overlay, (0, 0), (w, banner_h), color, -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        label     = self.LABELS.get(analysis.alert_level, "ALERT")
        font      = cv2.FONT_HERSHEY_DUPLEX
        (tw, th), _ = cv2.getTextSize(label, font, 1.1, 2)
        cv2.putText(frame, label, ((w - tw)//2, 38),
                    font, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

        if analysis.primary_cheat and analysis.primary_cheat != CheatType.NORMAL:
            detail = analysis.primary_cheat.value
            font2  = cv2.FONT_HERSHEY_SIMPLEX
            (dw, _), _ = cv2.getTextSize(detail, font2, 0.5, 1)
            cv2.putText(frame, detail, ((w - dw)//2, banner_h - 6),
                        font2, 0.5, (255, 255, 200), 1, cv2.LINE_AA)

    def draw_border_flash(self, frame, analysis: FrameAnalysis):
        if analysis.alert_level.value < AlertLevel.HIGH.value:
            return
        self._tick()
        if self._blink:
            color = self.COLORS[analysis.alert_level]
            h, w  = frame.shape[:2]
            cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 8)


# ─────────────── ALERT MANAGER ───────────────────────────────────────────────
class AlertManager:
    """Koordinasi audio + visual + log + screenshot."""

    # Cooldown antar bunyi per cheat type (detik)
    AUDIO_COOLDOWNS = {
        CheatType.PHONE_DETECTED:     3.0,
        CheatType.EXTENDED_GAZE_AWAY: 4.0,
        CheatType.LOOKING_LEFT:       4.0,
        CheatType.LOOKING_RIGHT:      4.0,
        CheatType.HEAD_DOWN:          5.0,
        CheatType.MULTIPLE_PERSONS:   5.0,
        CheatType.NO_PERSON:          6.0,
    }

    def __init__(self, config: Optional[AlertConfig] = None):
        self.config = config or AlertConfig()
        self.audio  = AudioAlert(self.config.audio_volume) if self.config.audio else None
        self.visual = VisualAlert() if self.config.visual else None

        self._audio_last:   Dict[str, float] = {}  # cooldown audio per cheat
        self._shot_last:    float = 0.0
        self._shot_cooldown = 3.0

        self.output_dir = Path(self.config.output_dir)
        self.log_dir    = Path(self.config.log_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = None
        if self.config.log_to_file:
            self._open_log()

        logger.info(f"AlertManager ready | audio={self.config.audio} "
                    f"backend={getattr(self.audio,'_backend','none')}")

    def _open_log(self):
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.log_dir / f"session_{ts}.log"
        self.log_file = open(path, 'w', encoding='utf-8')
        self.log_file.write(
            f"=== CV ExamGuard — Log Deteksi Kecurangan ===\n"
            f"Mulai : {datetime.now():%Y-%m-%d %H:%M:%S}\n"
            + "="*45 + "\n\n"
        )

    # ── Frame processing ─────────────────────────────────────────────────────
    def process_frame(self, frame, analysis: FrameAnalysis,
                      session_name: str = "") -> None:
        # 1. Visual overlay (selalu)
        if self.visual:
            self.visual.draw_alert_banner(frame, analysis)
            self.visual.draw_border_flash(frame, analysis)

        # 2. Log events (selalu jika ada)
        if self.log_file and analysis.events:
            for ev in analysis.events:
                self._log_event(ev)

        # 3. Audio — hanya jika ada cheat aktif
        if self.audio and analysis.detected_cheats:
            self._trigger_audio(analysis)

        # 4. Screenshot saat level tinggi
        if (self.config.screenshot_on_alert
                and analysis.alert_level.value >= AlertLevel.HIGH.value):
            now = time.time()
            if now - self._shot_last >= self._shot_cooldown:
                self._shot_last = now
                self._screenshot(frame, analysis, session_name)

    # ── Audio trigger ─────────────────────────────────────────────────────────
    def _trigger_audio(self, analysis: FrameAnalysis):
        priority = [
            CheatType.PHONE_DETECTED,
            CheatType.EXTENDED_GAZE_AWAY,
            CheatType.MULTIPLE_PERSONS,
            CheatType.HEAD_DOWN,
            CheatType.LOOKING_LEFT,
            CheatType.LOOKING_RIGHT,
            CheatType.NO_PERSON,
        ]
        for ct in priority:
            if ct not in analysis.detected_cheats:
                continue
            cooldown = self.AUDIO_COOLDOWNS.get(ct, 5.0)
            key      = ct.name
            last     = self._audio_last.get(key, 0.0)
            if time.time() - last >= cooldown:
                self._audio_last[key] = time.time()
                self.audio.play_cheat_alert(ct)
                logger.debug(f"Audio triggered: {ct.name}")
            break   # hanya play 1 suara per frame

    # ── Log ───────────────────────────────────────────────────────────────────
    def _log_event(self, ev: BehaviorEvent):
        if not self.log_file:
            return
        try:
            ts   = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
            line = (f"[{ts}] [{ev.alert_level.label():13}] "
                    f"{ev.cheat_type.value}: {ev.description}\n")
            self.log_file.write(line)
            self.log_file.flush()
        except Exception as e:
            logger.error(f"Log error: {e}")

    # ── Screenshot ────────────────────────────────────────────────────────────
    def _screenshot(self, frame, analysis: FrameAnalysis, session_name: str):
        try:
            ts    = datetime.now().strftime("%H%M%S")
            name  = analysis.primary_cheat.name if analysis.primary_cheat else "ALERT"
            path  = self.output_dir / f"shot_{session_name}_{ts}_{name}.jpg"
            ann   = frame.copy()
            h, w  = ann.shape[:2]
            mark  = f"{name} | {datetime.now():%Y-%m-%d %H:%M:%S}"
            cv2.putText(ann, mark, (10, h-12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,255), 1)
            cv2.imwrite(str(path), ann, [cv2.IMWRITE_JPEG_QUALITY, 92])
        except Exception as e:
            logger.error(f"Screenshot error: {e}")

    def close(self):
        if self.log_file:
            self.log_file.write(
                f"\n=== Selesai: {datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
            self.log_file.close()
            self.log_file = None
