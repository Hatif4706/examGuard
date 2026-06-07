"""
Dashboard UI - Redesigned dengan tampilan profesional modern
Clean dark theme, jelas dan informatif
"""

import cv2
import numpy as np
import time
from typing import List, Optional
from collections import deque

from .behavior_analyzer import FrameAnalysis, AlertLevel, CheatType
from typing import Optional as _Opt


# ─────────────────── WARNA PALLETE ───────────────────
C_BG         = (18, 18, 24)       # Background utama
C_PANEL      = (28, 30, 40)       # Panel card
C_BORDER     = (45, 50, 70)       # Border halus
C_TEXT       = (220, 225, 235)    # Teks utama
C_SUBTEXT    = (110, 120, 145)    # Teks sekunder
C_ACCENT     = (80, 160, 255)     # Biru accent
C_GREEN      = (50, 200, 100)     # Hijau sukses
C_YELLOW     = (50, 200, 220)     # Kuning peringatan (BGR)
C_ORANGE     = (50, 140, 255)     # Orange (BGR)
C_RED        = (70, 70, 230)      # Merah bahaya
C_RED_DARK   = (45, 45, 150)      # Merah gelap kritis
C_WHITE      = (255, 255, 255)
C_BLACK      = (0, 0, 0)

# Alert warna sesuai level
ALERT_COLORS = {
    AlertLevel.NONE:     C_GREEN,
    AlertLevel.LOW:      C_YELLOW,
    AlertLevel.MEDIUM:   C_ORANGE,
    AlertLevel.HIGH:     C_RED,
    AlertLevel.CRITICAL: C_RED_DARK,
}

ALERT_LABELS = {
    AlertLevel.NONE:     "AMAN",
    AlertLevel.LOW:      "WASPADA",
    AlertLevel.MEDIUM:   "MENCURIGAKAN",
    AlertLevel.HIGH:     "PERINGATAN",
    AlertLevel.CRITICAL: "KECURANGAN!",
}

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _rect(img, x1, y1, x2, y2, color, alpha=1.0):
    """Fill rectangle, optional semi-transparent."""
    if alpha >= 1.0:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        ov = img.copy()
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(ov, alpha, img, 1 - alpha, 0, img)


def _border(img, x1, y1, x2, y2, color, thick=1):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)


def _text(img, txt, x, y, font=FONT, scale=0.45, color=C_TEXT, thick=1):
    cv2.putText(img, str(txt), (x, y), font, scale, color, thick, cv2.LINE_AA)


def _hline(img, y, x1, x2, color=C_BORDER, thick=1):
    cv2.line(img, (x1, y), (x2, y), color, thick)


def _pill(img, x1, y1, x2, y2, color, radius=5):
    """Draw filled rounded rectangle (pill shape)."""
    h = y2 - y1
    r = min(radius, h // 2)
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
    for cx, cy in [(x1+r, y1+r), (x2-r, y1+r), (x1+r, y2-r), (x2-r, y2-r)]:
        cv2.circle(img, (cx, cy), r, color, -1)


class DashboardRenderer:
    """
    Dashboard kanan frame - tampilan profesional & bersih.
    Lebar panel: 340px. Semua elemen pada grid yang rapi.
    """

    W = 340          # Lebar panel
    PAD = 12         # Padding horizontal
    CARD_GAP = 6     # Jarak antar card

    def __init__(self, dashboard_width: int = 340):
        self.W = dashboard_width
        self.score_history = deque(maxlen=90)
        self._frame_times: deque = deque(maxlen=30)
        self._fps = 0.0
        self.session_start = time.time()
        self.total_events = 0
        self.recent_events: deque = deque(maxlen=7)
        self._metrics_ref = None   # set dari luar: dashboard._metrics_ref = metrics

    # ─── FPS ───────────────────────────────────────────────────
    def _tick_fps(self):
        now = time.time()
        self._frame_times.append(now)
        if len(self._frame_times) >= 2:
            span = self._frame_times[-1] - self._frame_times[0]
            self._fps = (len(self._frame_times) - 1) / max(span, 1e-5)

    # ─── UPDATE STATE ──────────────────────────────────────────
    def _update(self, analysis: FrameAnalysis):
        self.score_history.append(min(analysis.cheat_score, 25))
        for ev in analysis.events:
            self.total_events += 1
            ts = time.strftime("%H:%M:%S", time.localtime(ev.timestamp))
            self.recent_events.appendleft({
                "time": ts,
                "label": ev.cheat_type.value,
                "level": ev.alert_level,
            })

    # ─── CHART ─────────────────────────────────────────────────
    def _draw_chart(self, panel, x, y, w, h):
        """Mini area chart warna sesuai skor."""
        _rect(panel, x, y, x + w, y + h, (22, 24, 34))
        _border(panel, x, y, x + w, y + h, C_BORDER)

        data = list(self.score_history)
        if len(data) < 2:
            _text(panel, "Menunggu data...", x + 6, y + h // 2 + 5,
                  scale=0.35, color=C_SUBTEXT)
            return

        mx = max(max(data), 1)
        pts = []
        for i, v in enumerate(data):
            px = x + int(i * w / max(len(data) - 1, 1))
            py = y + h - int(v * (h - 6) / mx) - 3
            pts.append((px, py))

        # Area fill
        fill = [(x, y + h)] + pts + [(pts[-1][0], y + h)]
        ov = panel.copy()
        cv2.fillPoly(ov, [np.array(fill, np.int32)], C_ACCENT)
        cv2.addWeighted(ov, 0.2, panel, 0.8, 0, panel)

        # Line
        for i in range(1, len(pts)):
            cv2.line(panel, pts[i-1], pts[i], C_ACCENT, 2, cv2.LINE_AA)

        # Grid lines
        for pct in [0.33, 0.66]:
            gy = y + int(h * pct)
            cv2.line(panel, (x, gy), (x + w, gy), C_BORDER, 1)

    # ─── CARD HEADER ───────────────────────────────────────────
    def _section(self, panel, y, title):
        """Draw section label."""
        _text(panel, title, self.PAD, y, scale=0.32,
              color=C_SUBTEXT, thick=1)
        return y + 14

    # ─── MAIN RENDER ───────────────────────────────────────────
    def render(self, frame: np.ndarray, analysis: FrameAnalysis) -> np.ndarray:
        self._tick_fps()
        self._update(analysis)

        fh, fw = frame.shape[:2]
        W, PAD = self.W, self.PAD
        panel = np.full((fh, W, 3), C_BG, dtype=np.uint8)

        # Garis pembatas kiri panel
        cv2.line(panel, (0, 0), (0, fh), C_BORDER, 2)

        y = 10

        # ══════════════ HEADER ════════════════════════════════
        _rect(panel, 0, y - 2, W, y + 34, C_PANEL)
        _hline(panel, y + 34, 0, W, C_BORDER)

        _text(panel, "ExamGuard Monitor", PAD, y + 15,
              font=FONT_BOLD, scale=0.55, color=C_ACCENT, thick=1)

        # Jam
        now_str = time.strftime("%H:%M:%S")
        tw = cv2.getTextSize(now_str, FONT, 0.38, 1)[0][0]
        _text(panel, now_str, W - tw - PAD, y + 15,
              scale=0.38, color=C_SUBTEXT)

        y += 42

        # ══════════════ ALERT BANNER ══════════════════════════
        al = analysis.alert_level
        a_color = ALERT_COLORS[al]
        a_label = ALERT_LABELS[al]

        # Background banner dengan gradient efek
        _rect(panel, PAD, y, W - PAD, y + 46, C_PANEL)
        _rect(panel, PAD, y, W - PAD, y + 46, a_color, alpha=0.18)
        _border(panel, PAD, y, W - PAD, y + 46, a_color, thick=2)

        # Status icon (filled circle)
        cv2.circle(panel, (PAD + 18, y + 23), 10, a_color, -1)
        cv2.circle(panel, (PAD + 18, y + 23), 10, C_BG, 2)

        # Teks status
        _text(panel, a_label, PAD + 36, y + 17,
              font=FONT_BOLD, scale=0.58, color=a_color, thick=1)

        # Skor
        score_str = f"Skor: {analysis.cheat_score}"
        sw = cv2.getTextSize(score_str, FONT, 0.38, 1)[0][0]
        _text(panel, score_str, W - sw - PAD - 4, y + 38,
              scale=0.38, color=C_SUBTEXT)

        # Progress bar skor
        bar_x = PAD + 36
        bar_w = W - PAD - bar_x - 4
        bar_y = y + 32
        _rect(panel, bar_x, bar_y, bar_x + bar_w, bar_y + 5, C_BORDER)
        fill_w = int(min(analysis.cheat_score / 15, 1.0) * bar_w)
        if fill_w > 0:
            _rect(panel, bar_x, bar_y, bar_x + fill_w, bar_y + 5, a_color)

        y += 54

        # ══════════════ STATS GRID (2×2) ══════════════════════
        elapsed = time.time() - self.session_start
        face_count = len(analysis.head_poses)

        stats = [
            ("Durasi",        f"{int(elapsed//60):02d}:{int(elapsed%60):02d}"),
            ("FPS",           f"{self._fps:.1f}"),
            ("Wajah/Orang",   f"{face_count} / {analysis.person_count}"),
            ("Total Event",   str(self.total_events)),
        ]

        card_w = (W - 2 * PAD - 6) // 2
        card_h = 44

        for i, (lbl, val) in enumerate(stats):
            col = i % 2
            row = i // 2
            cx = PAD + col * (card_w + 6)
            cy = y + row * (card_h + 5)

            _pill(panel, cx, cy, cx + card_w, cy + card_h, C_PANEL, radius=6)

            _text(panel, lbl, cx + 8, cy + 14,
                  scale=0.32, color=C_SUBTEXT)
            _text(panel, val, cx + 8, cy + 33,
                  font=FONT_BOLD, scale=0.52, color=C_TEXT, thick=1)

        y += 2 * (card_h + 5) + 4

        # ══════════════ STATUS PANDANGAN ══════════════════════
        _hline(panel, y, PAD, W - PAD, C_BORDER)
        y += 4
        y = self._section(panel, y, "STATUS PANDANGAN")

        if analysis.head_poses:
            pose     = analysis.head_poses[0]
            left_dur = analysis.left_session_duration
            right_dur= analysis.right_session_duration
            down_dur = analysis.down_session_duration
            THRESH   = 3.0   # detik kecurangan

            if pose.looking_left:
                pct   = min(left_dur / THRESH, 1.0)
                if left_dur >= THRESH:
                    look_label = f"← KIRI! {left_dur:.0f}s (ke-{analysis.left_session_count}x)"
                    look_color = C_RED
                else:
                    look_label = f"← Menoleh kiri {left_dur:.1f}s/{THRESH:.0f}s"
                    look_color = C_ORANGE
            elif pose.looking_right:
                pct   = min(right_dur / THRESH, 1.0)
                if right_dur >= THRESH:
                    look_label = f"KANAN! → {right_dur:.0f}s (ke-{analysis.right_session_count}x)"
                    look_color = C_RED
                else:
                    look_label = f"Menoleh kanan → {right_dur:.1f}s/{THRESH:.0f}s"
                    look_color = C_ORANGE
            elif pose.looking_down:
                pct   = min(down_dur / THRESH, 1.0)
                if down_dur >= THRESH:
                    look_label = f"↓ MENUNDUK! {down_dur:.0f}s (ke-{analysis.down_session_count}x)"
                    look_color = C_RED
                else:
                    look_label = f"↓ Menunduk {down_dur:.1f}s/{THRESH:.0f}s"
                    look_color = C_ORANGE
            elif pose.looking_up:
                look_label = "↑ MELIHAT ATAS"
                look_color = C_YELLOW
                pct        = 0.0
            else:
                look_label = "✓ MELIHAT DEPAN"
                look_color = C_GREEN
                pct        = 0.0

            # Card pandangan
            _pill(panel, PAD, y, W - PAD, y + 52, C_PANEL, radius=6)
            _rect(panel, PAD, y, PAD + 4, y + 52, look_color, alpha=1.0)

            _text(panel, look_label, PAD + 12, y + 17,
                  font=FONT_BOLD, scale=0.48, color=look_color, thick=1)

            angle_str = f"Yaw {pose.yaw:+.0f}°   Pitch {pose.pitch:+.0f}°"
            _text(panel, angle_str, PAD + 12, y + 31,
                  scale=0.34, color=C_SUBTEXT)

            # Progress bar menuju threshold 3 detik
            bar_x = PAD + 12
            bar_w = W - PAD - bar_x - 10
            bar_y = y + 41
            _rect(panel, bar_x, bar_y, bar_x + bar_w, bar_y + 5, C_BORDER)
            if pct > 0:
                fill_color = C_RED if pct >= 1.0 else C_ORANGE
                _rect(panel, bar_x, bar_y, bar_x + int(bar_w * pct), bar_y + 5, fill_color)

            y += 60

        else:
            _pill(panel, PAD, y, W - PAD, y + 32, C_PANEL, radius=6)
            _rect(panel, PAD, y, PAD + 4, y + 32, C_RED)
            _text(panel, "Wajah Tidak Terdeteksi", PAD + 12, y + 20,
                  font=FONT_BOLD, scale=0.45, color=C_RED, thick=1)
            y += 39

        # ══════════════ STATUS OBJEK ══════════════════════════
        y += 2
        _hline(panel, y, PAD, W - PAD, C_BORDER)
        y += 4
        y = self._section(panel, y, "STATUS DETEKSI OBJEK")

        # ── HP Status (session-based) ──────────────────────────────────
        ph_active   = analysis.phone_session_active
        ph_count    = analysis.phone_session_count
        ph_dur      = analysis.phone_session_duration
        ph_detected = analysis.has_phone

        if ph_detected or ph_active:
            dot_color = C_RED
            if ph_count > 0:
                ph_label = f"HP ke-{ph_count}x  ({ph_dur:.0f}s)"
            else:
                ph_label = "HP TERDETEKSI!"
        else:
            dot_color = C_GREEN
            if ph_count > 0:
                ph_label = f"Aman (total {ph_count}x terdeteksi)"
            else:
                ph_label = "HP Tidak Terdeteksi"

        _pill(panel, PAD, y, W - PAD, y + 26, C_PANEL, radius=5)
        cv2.circle(panel, (PAD + 11, y + 13), 6, dot_color, -1)
        _text(panel, "HP / Gadget", PAD + 24, y + 17, scale=0.40, color=C_TEXT)
        st_w = cv2.getTextSize(ph_label, FONT, 0.37, 1)[0][0]
        _text(panel, ph_label, W - PAD - st_w - 6, y + 17, scale=0.37, color=dot_color)
        y += 30

        # ── Banyak Orang ──────────────────────────────────────────────
        multi = analysis.person_count > 1
        objects_rest = [
            ("Banyak Orang",      multi,  multi),
            ("Benda Mencurigakan", analysis.detection_result.has_suspicious_object,
             analysis.detection_result.has_suspicious_object),
        ]
        for obj_lbl, triggered, _ in objects_rest:
            dot_color = C_RED if triggered else C_GREEN
            _pill(panel, PAD, y, W - PAD, y + 26, C_PANEL, radius=5)
            cv2.circle(panel, (PAD + 11, y + 13), 6, dot_color, -1)
            status_str = "TERDETEKSI!" if triggered else "Aman"
            _text(panel, obj_lbl, PAD + 24, y + 17, scale=0.40, color=C_TEXT)
            st_w = cv2.getTextSize(status_str, FONT, 0.40, 1)[0][0]
            _text(panel, status_str, W - PAD - st_w - 6, y + 17,
                  scale=0.40, color=dot_color)
            y += 30

        # ══════════════ GRAFIK SKOR ════════════════════════════
        y += 2
        _hline(panel, y, PAD, W - PAD, C_BORDER)
        y += 4
        y = self._section(panel, y, "GRAFIK RISIKO")

        chart_h = 52
        if y + chart_h + 10 < fh - 90:
            self._draw_chart(panel, PAD, y, W - 2 * PAD, chart_h)
            y += chart_h + 6

        # ══════════════ LOG KEJADIAN ══════════════════════════
        _hline(panel, y, PAD, W - PAD, C_BORDER)
        y += 4
        y = self._section(panel, y, "LOG KEJADIAN TERAKHIR")

        if self.recent_events:
            for ev in list(self.recent_events):
                if y + 22 > fh - 28:
                    break
                ev_color = ALERT_COLORS.get(ev["level"], C_SUBTEXT)

                _rect(panel, PAD, y, W - PAD, y + 20, (24, 26, 38))

                # Time badge
                _pill(panel, PAD, y + 3, PAD + 54, y + 17,
                      (35, 38, 55), radius=4)
                _text(panel, ev["time"], PAD + 2, y + 14,
                      scale=0.30, color=C_SUBTEXT)

                # Event label (truncate)
                lbl = ev["label"][:26]
                _text(panel, lbl, PAD + 60, y + 14,
                      scale=0.36, color=ev_color)

                y += 21
        else:
            _text(panel, "Belum ada kejadian mencurigakan",
                  PAD, y + 12, scale=0.36, color=C_SUBTEXT)
            y += 20

        # ══════════════ METRIK PERFORMA ═══════════════════════
        if self._metrics_ref is not None and y + 80 < fh - 28:
            _hline(panel, y, PAD, W - PAD, C_BORDER)
            y += 4
            y = self._section(panel, y, "METRIK DETEKSI (SESI INI)")
            try:
                s = self._metrics_ref.get_summary()
                m_items = [
                    ("Accuracy",  s["accuracy"]),
                    ("Precision", s["precision"]),
                    ("Recall",    s["recall"]),
                    ("F1-Score",  s["f1"]),
                    ("mAP@0.5",   s["map50"]),
                ]
                col_w2 = (W - 2*PAD - 4) // 2
                for i, (lbl, val) in enumerate(m_items):
                    col = i % 2
                    row = i // 2
                    bx  = PAD + col * (col_w2 + 4)
                    by  = y + row * 28
                    if by + 26 >= fh - 28:
                        break
                    _pill(panel, bx, by, bx + col_w2, by + 26, C_PANEL, radius=4)
                    _text(panel, lbl, bx + 5, by + 11, scale=0.30, color=C_SUBTEXT)
                    pct_str = f"{val*100:.1f}%"
                    bar_color = C_GREEN if val >= 0.7 else (C_YELLOW if val >= 0.5 else C_RED)
                    _text(panel, pct_str, bx + 5, by + 22, scale=0.40,
                          color=bar_color, font=FONT_BOLD)
                y += 3 * 28 + 4
            except Exception:
                pass

        # ══════════════ FOOTER ════════════════════════════════
        fy = fh - 22
        _hline(panel, fy - 6, 0, W, C_BORDER)
        _text(panel, "CV ExamGuard v2.3  |  YOLOv8 + MediaPipe",
              PAD, fy + 10, scale=0.28, color=C_BORDER)

        # ─── Gabung frame + panel ──────────────────────────────
        return np.hstack([frame, panel])
