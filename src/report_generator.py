"""
Report Generator - Membuat laporan HTML/PDF sesi deteksi kecurangan
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from .behavior_analyzer import BehaviorEvent, CheatType

logger = logging.getLogger(__name__)


class ReportGenerator:
    """Generates HTML and PDF reports for exam monitoring sessions."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_html_report(self, session_stats: Dict,
                               events: List[BehaviorEvent],
                               session_name: str) -> str:
        """Generate laporan HTML lengkap."""

        duration = session_stats.get("session_duration", 0)
        duration_str = f"{int(duration // 60):02d}:{int(duration % 60):02d}"
        risk_level = session_stats.get("risk_level", "Tidak Diketahui")
        total_events = session_stats.get("total_events", 0)
        events_by_type = session_stats.get("events_by_type", {})

        # Hitung persentase kecurangan
        total_frames = max(session_stats.get("total_frames", 1), 1)
        cheat_frames = session_stats.get("cheat_frame_counts", {})
        phone_pct = round(cheat_frames.get("Menggunakan HP", 0) / total_frames * 100, 1)
        gaze_pct = round(
            (cheat_frames.get("Melihat ke Kiri", 0) +
             cheat_frames.get("Melihat ke Kanan", 0)) / total_frames * 100, 1)
        multi_pct = round(cheat_frames.get("Banyak Orang Terdeteksi", 0) / total_frames * 100, 1)

        # Warna level risiko
        risk_colors = {
            "NORMAL": "#10b981",
            "RENDAH": "#f59e0b",
            "SEDANG": "#f97316",
            "TINGGI": "#ef4444",
            "SANGAT TINGGI": "#dc2626"
        }
        risk_color = risk_colors.get(risk_level, "#6b7280")

        # Generate tabel events
        events_rows = ""
        for ev in sorted(events[-100:], key=lambda x: x.timestamp, reverse=True):
            time_str = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
            level_class = ev.alert_level.name.lower()
            events_rows += f"""
            <tr class="event-row {level_class}">
                <td>{time_str}</td>
                <td><span class="badge {level_class}">{ev.alert_level.label()}</span></td>
                <td>{ev.cheat_type.value}</td>
                <td>{ev.description}</td>
                <td>{ev.cheat_score}</td>
            </tr>"""

        # Generate chart data (untuk chart.js)
        events_timeline = {}
        for ev in events:
            minute = int((ev.timestamp - (events[0].timestamp if events else ev.timestamp)) // 60)
            events_timeline[minute] = events_timeline.get(minute, 0) + 1

        timeline_labels = list(events_timeline.keys())
        timeline_data = list(events_timeline.values())

        cheat_type_labels = list(events_by_type.keys())
        cheat_type_data = list(events_by_type.values())

        html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Laporan Pengawasan Ujian - {session_name}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0f172a; color: #e2e8f0; }}
  .container {{ max-width: 1200px; margin: 0 auto; padding: 24px; }}

  /* Header */
  .header {{ background: linear-gradient(135deg, #1e3a5f, #1a2744); border-radius: 16px;
             padding: 32px; margin-bottom: 24px; border: 1px solid #334155; }}
  .header h1 {{ font-size: 2rem; color: #60a5fa; margin-bottom: 8px; }}
  .header .subtitle {{ color: #94a3b8; font-size: 0.9rem; }}
  .header .meta {{ display: flex; gap: 24px; margin-top: 16px; flex-wrap: wrap; }}
  .header .meta-item {{ background: rgba(255,255,255,0.05); padding: 8px 16px;
                        border-radius: 8px; font-size: 0.85rem; }}
  .header .meta-item strong {{ color: #60a5fa; }}

  /* Risk badge besar */
  .risk-banner {{ text-align: center; padding: 24px; background: rgba(255,255,255,0.05);
                  border-radius: 12px; margin-bottom: 24px; border: 2px solid {risk_color}; }}
  .risk-banner .risk-label {{ font-size: 1rem; color: #94a3b8; margin-bottom: 8px; }}
  .risk-banner .risk-value {{ font-size: 3rem; font-weight: bold; color: {risk_color}; }}

  /* Cards grid */
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px; margin-bottom: 24px; }}
  .card {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
  .card .card-title {{ font-size: 0.8rem; color: #64748b; text-transform: uppercase;
                       letter-spacing: 0.05em; margin-bottom: 8px; }}
  .card .card-value {{ font-size: 2rem; font-weight: bold; color: #f1f5f9; }}
  .card .card-sub {{ font-size: 0.75rem; color: #64748b; margin-top: 4px; }}

  /* Charts */
  .charts {{ display: grid; grid-template-columns: 1fr 1fr; gap: 24px; margin-bottom: 24px; }}
  .chart-box {{ background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }}
  .chart-box h3 {{ margin-bottom: 16px; color: #94a3b8; font-size: 0.9rem; }}

  /* Events table */
  .events-section {{ background: #1e293b; border-radius: 12px; padding: 24px;
                     border: 1px solid #334155; }}
  .events-section h2 {{ margin-bottom: 16px; color: #94a3b8; }}
  .events-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  .events-table th {{ text-align: left; padding: 10px 12px; background: #0f172a;
                      color: #64748b; border-bottom: 2px solid #334155; }}
  .events-table td {{ padding: 10px 12px; border-bottom: 1px solid #1e293b; }}
  .events-table tr:hover {{ background: rgba(255,255,255,0.03); }}

  /* Badges */
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.75rem; font-weight: 600; }}
  .badge.low {{ background: #fef3c7; color: #92400e; }}
  .badge.medium {{ background: #fed7aa; color: #9a3412; }}
  .badge.high {{ background: #fecaca; color: #991b1b; }}
  .badge.critical {{ background: #dc2626; color: white; }}

  /* Event row colors */
  .event-row.high td:first-child {{ border-left: 3px solid #ef4444; }}
  .event-row.critical td:first-child {{ border-left: 3px solid #dc2626; background: rgba(220,38,38,0.05); }}
  .event-row.medium td:first-child {{ border-left: 3px solid #f97316; }}

  /* Stats bar */
  .stat-bar {{ background: #0f172a; border-radius: 8px; height: 8px; margin-top: 8px; overflow: hidden; }}
  .stat-bar-fill {{ height: 100%; border-radius: 8px; transition: width 1s ease; }}

  /* Footer */
  .footer {{ text-align: center; color: #475569; font-size: 0.8rem; margin-top: 24px; padding: 16px; }}

  @media (max-width: 768px) {{
    .charts {{ grid-template-columns: 1fr; }}
    .header h1 {{ font-size: 1.5rem; }}
  }}
</style>
</head>
<body>
<div class="container">

  <!-- Header -->
  <div class="header">
    <h1>🎓 Laporan Pengawasan Ujian</h1>
    <div class="subtitle">Sistem Deteksi Kecurangan Berbasis Computer Vision</div>
    <div class="meta">
      <div class="meta-item"><strong>Sesi:</strong> {session_name}</div>
      <div class="meta-item"><strong>Tanggal:</strong> {datetime.now().strftime('%d %B %Y')}</div>
      <div class="meta-item"><strong>Durasi:</strong> {duration_str}</div>
      <div class="meta-item"><strong>Total Frame:</strong> {total_frames:,}</div>
    </div>
  </div>

  <!-- Risk Banner -->
  <div class="risk-banner">
    <div class="risk-label">TINGKAT RISIKO KESELURUHAN</div>
    <div class="risk-value">{risk_level}</div>
  </div>

  <!-- Cards -->
  <div class="cards">
    <div class="card">
      <div class="card-title">Total Kejadian</div>
      <div class="card-value">{total_events}</div>
      <div class="card-sub">alert terdeteksi</div>
    </div>
    <div class="card">
      <div class="card-title">Deteksi HP</div>
      <div class="card-value" style="color:#ef4444">{phone_pct}%</div>
      <div class="card-sub">dari total waktu ujian</div>
      <div class="stat-bar"><div class="stat-bar-fill" style="width:{min(phone_pct,100)}%;background:#ef4444"></div></div>
    </div>
    <div class="card">
      <div class="card-title">Pandangan Mencurigakan</div>
      <div class="card-value" style="color:#f97316">{gaze_pct}%</div>
      <div class="card-sub">melihat ke samping</div>
      <div class="stat-bar"><div class="stat-bar-fill" style="width:{min(gaze_pct,100)}%;background:#f97316"></div></div>
    </div>
    <div class="card">
      <div class="card-title">Banyak Orang</div>
      <div class="card-value" style="color:#a855f7">{multi_pct}%</div>
      <div class="card-sub">lebih dari 1 peserta</div>
      <div class="stat-bar"><div class="stat-bar-fill" style="width:{min(multi_pct,100)}%;background:#a855f7"></div></div>
    </div>
    <div class="card">
      <div class="card-title">Skor Risiko Total</div>
      <div class="card-value">{session_stats.get('total_cheat_score', 0)}</div>
      <div class="card-sub">akumulasi skor</div>
    </div>
  </div>

  <!-- Charts -->
  <div class="charts">
    <div class="chart-box">
      <h3>📊 Timeline Kejadian</h3>
      <canvas id="timelineChart" height="200"></canvas>
    </div>
    <div class="chart-box">
      <h3>🍩 Distribusi Jenis Kecurangan</h3>
      <canvas id="typeChart" height="200"></canvas>
    </div>
  </div>

  <!-- Events Table -->
  <div class="events-section">
    <h2>📋 Log Kejadian (100 Terakhir)</h2>
    <table class="events-table">
      <thead>
        <tr>
          <th>Waktu</th>
          <th>Level</th>
          <th>Jenis</th>
          <th>Deskripsi</th>
          <th>Skor</th>
        </tr>
      </thead>
      <tbody>
        {events_rows if events_rows else '<tr><td colspan="5" style="text-align:center;color:#64748b;padding:24px">Tidak ada kejadian mencurigakan</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="footer">
    Dihasilkan oleh Sistem Deteksi Kecurangan Ujian | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
  </div>
</div>

<script>
// Timeline chart
const tlCtx = document.getElementById('timelineChart').getContext('2d');
new Chart(tlCtx, {{
  type: 'line',
  data: {{
    labels: {json.dumps([f'Mnt {l}' for l in timeline_labels])},
    datasets: [{{
      label: 'Kejadian per Menit',
      data: {json.dumps(timeline_data)},
      borderColor: '#ef4444',
      backgroundColor: 'rgba(239,68,68,0.15)',
      fill: true,
      tension: 0.4
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8' }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#64748b' }}, grid: {{ color: '#1e293b' }}, beginAtZero: true }}
    }}
  }}
}});

// Type chart
const typeCtx = document.getElementById('typeChart').getContext('2d');
new Chart(typeCtx, {{
  type: 'doughnut',
  data: {{
    labels: {json.dumps(cheat_type_labels if cheat_type_labels else ['Tidak Ada Data'])},
    datasets: [{{
      data: {json.dumps(cheat_type_data if cheat_type_data else [1])},
      backgroundColor: ['#ef4444','#f97316','#f59e0b','#10b981','#3b82f6','#8b5cf6','#ec4899','#14b8a6'],
      borderWidth: 0
    }}]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }}, position: 'bottom' }}
    }}
  }}
}});
</script>
</body>
</html>"""

        # # Simpan file
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"report_{session_name}_{timestamp}.html"
        # filepath = self.output_dir / filename
        # with open(filepath, 'w', encoding='utf-8') as f:
        #     f.write(html)

        # logger.info(f"Laporan HTML disimpan: {filepath}")
        # return str(filepath)

    def generate_pdf_report(self, session_stats: Dict,
                             events: List[BehaviorEvent],
                             session_name: str) -> Optional[str]:
        """Generate laporan PDF."""
        try:
            from fpdf import FPDF

            class ExamPDF(FPDF):
                def header(self):
                    self.set_font('Arial', 'B', 14)
                    self.set_text_color(30, 80, 150)
                    self.cell(0, 10, 'LAPORAN PENGAWASAN UJIAN', 0, 1, 'C')
                    self.set_font('Arial', '', 9)
                    self.set_text_color(100, 100, 100)
                    self.cell(0, 6, 'Sistem Deteksi Kecurangan Berbasis Computer Vision', 0, 1, 'C')
                    self.ln(5)

                def footer(self):
                    self.set_y(-15)
                    self.set_font('Arial', 'I', 8)
                    self.set_text_color(150, 150, 150)
                    self.cell(0, 10, f'Halaman {self.page_no()}', 0, 0, 'C')

            pdf = ExamPDF()
            pdf.add_page()
            pdf.set_auto_page_break(auto=True, margin=15)

            # Info sesi
            pdf.set_font('Arial', 'B', 11)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 8, f'Sesi: {session_name}', 0, 1)
            pdf.set_font('Arial', '', 10)
            pdf.cell(0, 6, f'Tanggal: {datetime.now().strftime("%d %B %Y %H:%M")}', 0, 1)

            duration = session_stats.get("session_duration", 0)
            pdf.cell(0, 6, f'Durasi: {int(duration//60):02d}:{int(duration%60):02d}', 0, 1)
            pdf.ln(5)

            # Tingkat risiko
            risk_level = session_stats.get("risk_level", "Tidak Diketahui")
            pdf.set_font('Arial', 'B', 14)
            risk_colors = {
                "NORMAL": (16, 185, 129),
                "RENDAH": (245, 158, 11),
                "SEDANG": (249, 115, 22),
                "TINGGI": (239, 68, 68),
                "SANGAT TINGGI": (185, 28, 28)
            }
            r, g, b = risk_colors.get(risk_level, (100, 100, 100))
            pdf.set_text_color(r, g, b)
            pdf.cell(0, 10, f'Tingkat Risiko: {risk_level}', 0, 1, 'C')
            pdf.ln(5)

            # Statistik
            pdf.set_font('Arial', 'B', 11)
            pdf.set_text_color(50, 50, 50)
            pdf.cell(0, 8, 'STATISTIK SESI:', 0, 1)
            pdf.set_font('Arial', '', 10)

            stats_items = [
                ("Total Kejadian", session_stats.get("total_events", 0)),
                ("Total Frame Diproses", session_stats.get("total_frames", 0)),
                ("Skor Risiko Kumulatif", session_stats.get("total_cheat_score", 0)),
            ]

            for label, value in stats_items:
                pdf.cell(80, 7, label + ":", 1, 0, 'L')
                pdf.cell(40, 7, str(value), 1, 1, 'C')
            pdf.ln(5)

            # Tabel events
            if events:
                pdf.set_font('Arial', 'B', 11)
                pdf.cell(0, 8, f'LOG KEJADIAN (100 terakhir):', 0, 1)
                pdf.set_font('Arial', 'B', 9)

                col_widths = [25, 25, 50, 80, 15]
                headers = ['Waktu', 'Level', 'Jenis', 'Deskripsi', 'Skor']

                for i, header in enumerate(headers):
                    pdf.cell(col_widths[i], 7, header, 1, 0, 'C')
                pdf.ln()

                pdf.set_font('Arial', '', 8)
                for ev in sorted(events[-100:], key=lambda x: x.timestamp, reverse=True):
                    time_str = datetime.fromtimestamp(ev.timestamp).strftime("%H:%M:%S")
                    row = [time_str, ev.alert_level.label(), ev.cheat_type.value,
                           ev.description[:45], str(ev.cheat_score)]
                    for i, cell in enumerate(row):
                        pdf.cell(col_widths[i], 6, str(cell), 1, 0, 'L')
                    pdf.ln()

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"report_{session_name}_{timestamp}.pdf"
            filepath = self.output_dir / filename
            pdf.output(str(filepath))
            logger.info(f"Laporan PDF disimpan: {filepath}")
            return str(filepath)

        except ImportError:
            logger.warning("fpdf2 tidak tersedia, skip PDF report")
            return None
        except Exception as e:
            logger.error(f"Error generating PDF: {e}")
            return None
