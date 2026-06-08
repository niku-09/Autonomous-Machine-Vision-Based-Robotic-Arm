"""
============================================================
 dashboard_server.py — Live Inventory Dashboard Backend
============================================================

 Runs a Flask web server that:
   - Streams live camera feed (MJPEG)
   - Runs box detection + classification continuously
   - Tracks inventory across sorting sessions
   - Serves real-time data to the dashboard UI

 HOW TO RUN
 ----------
   pip install flask
   python dashboard_server.py

   Then open browser: http://localhost:5000

 INTEGRATES WITH
 ---------------
   detect_boxes.py   — box detection
   classify_box.py   — sticker classification
   calibration.npz   — camera calibration

 NOTE: Dashboard is view-only (no robot control).
       Run main.py separately for robot control.
       Both can run simultaneously — dashboard reads
       from camera while main.py controls the robot.
============================================================
"""

import sys
import time
import threading
import json
import math
from datetime import datetime
from collections import defaultdict
from flask import Flask, Response, jsonify, render_template_string

sys.path.append(".")

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
CAMERA_INDEX    = 0
SCAN_INTERVAL   = 2.0      # seconds between background scans
DASHBOARD_PORT  = 5000
USE_DETECTION   = True     # set False to run dashboard without detection

# ─────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────────────
state = {
    "session_start":     datetime.now().isoformat(),
    "active_boxes":      [],        # currently on board
    "sorted_history":    [],        # all sorted boxes this session
    "inventory": {
        "FRAGILE":   0,
        "ZONE_A":    0,
        "ZONE_B":    0,
        "IMMEDIATE": 0,
        "DAMAGED":   0,
        "UNKNOWN":   0,
    },
    "sorted_counts": {
        "FRAGILE":   0,
        "ZONE_A":    0,
        "ZONE_B":    0,
        "IMMEDIATE": 0,
        "DAMAGED":   0,
    },
    "total_detected":    0,
    "total_sorted":      0,
    "session_trend":     [],        # [{time, active, total}]
    "last_scan":         None,
    "fps":               0.0,
    "feed_status":       "ONLINE",
    "detection_active":  False,
    "packages":          {},        # pkg_id → {label, first_seen, coords, status}
    "pkg_counter":       0,
}

state_lock = threading.Lock()
latest_frame = None
frame_lock   = threading.Lock()

LABEL_DISPLAY = {
    "FRAGILE":   "Fragile",
    "ZONE_A":    "Zone A",
    "ZONE_B":    "Zone B",
    "IMMEDIATE": "Immediate",
    "DAMAGED":   "Damaged",
    "UNKNOWN":   "Unknown",
}

LABEL_COLORS = {
    "FRAGILE":   "#ef4444",
    "ZONE_A":    "#3b82f6",
    "ZONE_B":    "#eab308",
    "IMMEDIATE": "#22c55e",
    "DAMAGED":   "#f97316",
    "UNKNOWN":   "#6b7280",
}

# ─────────────────────────────────────────────────────────────
#  DETECTION THREAD
# ─────────────────────────────────────────────────────────────
def detection_thread():
    global latest_frame
    import cv2
    import numpy as np

    try:
        from detect_boxes  import BoxDetector
        from classify_box  import classify_all_boxes
    except Exception as e:
        print(f"[WARN] Could not load detection modules: {e}")
        print("       Running in camera-only mode.")
        _camera_only_thread()
        return

    try:
        detector = BoxDetector()
    except Exception as e:
        print(f"[ERR] BoxDetector failed: {e}")
        state["feed_status"] = "OFFLINE"
        return

    # Capture background
    print("[BG] Capturing background (board must be clear)...")
    time.sleep(2)
    try:
        detector.capture_background()
        print("[BG] Background captured.")
    except Exception as e:
        print(f"[WARN] Background capture failed: {e}")

    fps_t   = time.time()
    fps_cnt = 0
    last_scan_t = 0

    with state_lock:
        state["detection_active"] = True
        state["feed_status"]      = "ONLINE"

    while True:
        try:
            ret, raw = detector.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = detector.undistort(raw)

            # Annotate frame with last known boxes
            disp = frame.copy()
            with state_lock:
                boxes = state["active_boxes"]
            for b in boxes:
                px, py = int(b.get("px", 0)), int(b.get("py", 0))
                label  = b.get("label", "UNKNOWN")
                col    = _hex_to_bgr(LABEL_COLORS.get(label, "#6b7280"))
                cv2.circle(disp, (px, py), 8, col, -1)
                cv2.rectangle(disp,
                              (b.get("bx",0), b.get("by",0)),
                              (b.get("bx",0)+b.get("bw",0),
                               b.get("by",0)+b.get("bh",0)),
                              col, 2)
                cv2.putText(disp, label,
                            (b.get("bx",0), b.get("by",0)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            col, 2)

            # FPS overlay
            fps_cnt += 1
            if fps_cnt >= 10:
                elapsed = time.time() - fps_t
                fps     = fps_cnt / elapsed
                with state_lock:
                    state["fps"] = round(fps, 1)
                fps_cnt = 0
                fps_t   = time.time()

            with state_lock:
                fps_val = state["fps"]
            cv2.putText(disp, f"FPS: {fps_val:.1f}",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0,220,80), 2)

            # Encode frame as JPEG
            _, buf = cv2.imencode('.jpg', disp,
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
            with frame_lock:
                latest_frame = buf.tobytes()

            # Periodic detection scan
            now = time.time()
            if now - last_scan_t >= SCAN_INTERVAL:
                last_scan_t = now
                _run_scan(detector, frame, classify_all_boxes)

        except Exception as e:
            print(f"[ERR] Detection loop: {e}")
            time.sleep(0.5)


def _camera_only_thread():
    """Fallback: just stream camera without detection."""
    import cv2
    global latest_frame
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        state["feed_status"] = "OFFLINE"
        return
    state["feed_status"] = "ONLINE"
    while True:
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode('.jpg', frame,
                                  [cv2.IMWRITE_JPEG_QUALITY, 80])
            with frame_lock:
                latest_frame = buf.tobytes()
        time.sleep(0.033)


def _run_scan(detector, frame, classify_all_boxes):
    """Run one detection+classification scan and update state."""
    try:
        boxes, _ = detector.detect(visualise=False)
        if not boxes:
            with state_lock:
                state["active_boxes"] = []
                state["last_scan"]    = datetime.now().isoformat()
                _update_trend()
            return

        results = classify_all_boxes(frame, detector.background, boxes)
        active  = []
        counts  = defaultdict(int)

        with state_lock:
            for r in results:
                x, y, z = r["coords"]
                label    = r["label"]
                px, py   = int(r["pixel"][0]), int(r["pixel"][1])
                bx, by, bw, bh = r["rect"]
                counts[label] += 1

                # Package tracking
                pkg_id = _find_or_create_package(x, y, label)

                active.append({
                    "pkg_id":     pkg_id,
                    "label":      label,
                    "confidence": round(r["confidence"]*100),
                    "x":          round(x, 1),
                    "y":          round(y, 1),
                    "px": px, "py": py,
                    "bx": bx, "by": by, "bw": bw, "bh": bh,
                    "first_seen": state["packages"].get(pkg_id, {}).get("first_seen", datetime.now().isoformat()),
                })

            state["active_boxes"] = active
            state["total_detected"] = max(state["total_detected"], len(active))
            for label, cnt in counts.items():
                if label in state["inventory"]:
                    state["inventory"][label] = cnt
            # Zero out labels not seen
            for label in state["inventory"]:
                if label not in counts:
                    state["inventory"][label] = 0
            state["last_scan"] = datetime.now().isoformat()
            _update_trend()

    except Exception as e:
        print(f"[ERR] Scan: {e}")


def _find_or_create_package(x, y, label):
    """Match to existing package or create new one."""
    for pkg_id, pkg in state["packages"].items():
        if pkg["status"] == "active":
            dx = abs(pkg["x"] - x)
            dy = abs(pkg["y"] - y)
            if dx < 5.0 and dy < 5.0:
                pkg["label"] = label
                pkg["x"]     = x
                pkg["y"]     = y
                return pkg_id
    # New package
    state["pkg_counter"] += 1
    pkg_id = f"PKG-{state['pkg_counter']:04d}"
    state["packages"][pkg_id] = {
        "label":      label,
        "x": x, "y": y,
        "first_seen": datetime.now().isoformat(),
        "status":     "active",
    }
    return pkg_id


def _update_trend():
    """Append a trend data point (called inside state_lock)."""
    total  = sum(state["sorted_counts"].values())
    active = len(state["active_boxes"])
    state["session_trend"].append({
        "t":      datetime.now().strftime("%H:%M:%S"),
        "active": active,
        "total":  total,
    })
    # Keep last 60 points
    if len(state["session_trend"]) > 60:
        state["session_trend"] = state["session_trend"][-60:]


def _hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (b, g, r)


# ─────────────────────────────────────────────────────────────
#  PUBLIC API: mark a box as sorted (called from main.py)
# ─────────────────────────────────────────────────────────────
def record_sorted(label, x, y):
    """Call this from main.py after a successful pick and place."""
    with state_lock:
        if label in state["sorted_counts"]:
            state["sorted_counts"][label] += 1
        state["total_sorted"] += 1
        state["sorted_history"].append({
            "label":  label,
            "x":      round(x,1),
            "y":      round(y,1),
            "time":   datetime.now().isoformat(),
        })
        # Mark package as sorted
        for pkg_id, pkg in state["packages"].items():
            if pkg["status"] == "active":
                if abs(pkg["x"]-x)<5 and abs(pkg["y"]-y)<5:
                    pkg["status"] = "sorted"
                    break


# ─────────────────────────────────────────────────────────────
#  FLASK APP
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/video_feed")
def video_feed():
    return Response(_gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


def _gen_frames():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n\r\n"
                   + frame + b"\r\n")
        time.sleep(0.033)


@app.route("/api/state")
def api_state():
    with state_lock:
        data = {
            "session_start":  state["session_start"],
            "active_count":   len(state["active_boxes"]),
            "total_detected": state["total_detected"],
            "total_sorted":   state["total_sorted"],
            "inventory":      state["inventory"],
            "sorted_counts":  state["sorted_counts"],
            "active_boxes":   state["active_boxes"],
            "sorted_history": state["sorted_history"][-20:],
            "session_trend":  state["session_trend"][-30:],
            "last_scan":      state["last_scan"],
            "fps":            state["fps"],
            "feed_status":    state["feed_status"],
            "packages":       {
                k: v for k, v in list(state["packages"].items())[-10:]
            },
            "fragile_share": (
                round(state["sorted_counts"]["FRAGILE"] /
                      max(1, state["total_sorted"]) * 100)
                if state["total_sorted"] > 0 else 0
            ),
        }
    return jsonify(data)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with state_lock:
        state["sorted_counts"] = {k: 0 for k in state["sorted_counts"]}
        state["total_sorted"]  = 0
        state["total_detected"] = 0
        state["sorted_history"] = []
        state["session_trend"]  = []
        state["packages"]       = {}
        state["pkg_counter"]    = 0
        state["session_start"]  = datetime.now().isoformat()
    return jsonify({"ok": True})


# ─────────────────────────────────────────────────────────────
#  DASHBOARD HTML
# ─────────────────────────────────────────────────────────────
DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SortOS — Package Intelligence Dashboard</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
  :root {
    --bg:       #080c10;
    --surface:  #0d1117;
    --border:   #1e2836;
    --text:     #c9d1d9;
    --muted:    #4a5568;
    --accent:   #00d4aa;
    --red:      #ef4444;
    --blue:     #3b82f6;
    --yellow:   #eab308;
    --green:    #22c55e;
    --orange:   #f97316;
    --grey:     #6b7280;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    height: 100vh;
    display: grid;
    grid-template-rows: 56px 1fr;
    overflow: hidden;
  }

  /* ── HEADER ── */
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 24px;
    border-bottom: 1px solid var(--border);
    background: var(--surface);
    z-index: 10;
  }
  .logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 18px;
    font-weight: 800;
    letter-spacing: 0.08em;
    color: #fff;
  }
  .logo-dot {
    width: 10px; height: 10px;
    background: var(--accent);
    border-radius: 50%;
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%,100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(0.8); }
  }
  .header-meta {
    display: flex;
    gap: 24px;
    align-items: center;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--muted);
  }
  .status-pill {
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
  }
  .status-online { background: rgba(34,197,94,0.15); color: var(--green); border: 1px solid rgba(34,197,94,0.3); }
  .status-offline { background: rgba(239,68,68,0.15); color: var(--red); border: 1px solid rgba(239,68,68,0.3); }
  #clock { color: var(--accent); font-size: 12px; }
  .reset-btn {
    padding: 5px 14px;
    background: transparent;
    border: 1px solid var(--border);
    color: var(--muted);
    border-radius: 6px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    cursor: pointer;
    letter-spacing: 0.08em;
    transition: all 0.2s;
  }
  .reset-btn:hover { border-color: var(--accent); color: var(--accent); }

  /* ── MAIN GRID ── */
  .main {
    display: grid;
    grid-template-columns: 340px 1fr 280px;
    grid-template-rows: 1fr;
    gap: 0;
    overflow: hidden;
  }

  /* ── PANELS ── */
  .panel {
    border-right: 1px solid var(--border);
    overflow-y: auto;
    scrollbar-width: none;
  }
  .panel::-webkit-scrollbar { display: none; }
  .panel-section {
    padding: 16px;
    border-bottom: 1px solid var(--border);
  }
  .section-label {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
  }

  /* ── STAT CARDS ── */
  .stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 8px;
  }
  .stat-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px;
  }
  .stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #fff;
    line-height: 1;
  }
  .stat-label {
    font-size: 10px;
    color: var(--muted);
    margin-top: 4px;
    letter-spacing: 0.05em;
  }
  .stat-accent { color: var(--accent); }

  /* ── CATEGORY BARS ── */
  .category-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 10px;
  }
  .category-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
  }
  .category-name {
    font-size: 11px;
    font-weight: 600;
    flex: 1;
    letter-spacing: 0.04em;
  }
  .category-bar-wrap {
    flex: 2;
    background: var(--border);
    border-radius: 4px;
    height: 4px;
    overflow: hidden;
  }
  .category-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
  }
  .category-count {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    color: var(--text);
    width: 24px;
    text-align: right;
  }

  /* ── LIVE FEED CENTER ── */
  .feed-panel {
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }
  .feed-header {
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
    flex-shrink: 0;
  }
  .feed-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  .feed-meta {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    display: flex;
    gap: 16px;
  }
  .feed-wrap {
    flex: 1;
    overflow: hidden;
    position: relative;
    background: #000;
  }
  .feed-wrap img {
    width: 100%;
    height: 100%;
    object-fit: contain;
  }
  .feed-overlay {
    position: absolute;
    top: 10px;
    left: 10px;
    display: flex;
    flex-direction: column;
    gap: 6px;
    pointer-events: none;
  }
  .feed-tag {
    background: rgba(8,12,16,0.85);
    border: 1px solid var(--border);
    padding: 3px 8px;
    border-radius: 4px;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: var(--accent);
  }

  /* ── TREND CHART ── */
  .chart-wrap {
    height: 90px;
    position: relative;
    margin-top: 4px;
  }
  canvas { display: block; }

  /* ── RIGHT PANEL — PACKAGES ── */
  .pkg-list { display: flex; flex-direction: column; gap: 8px; }
  .pkg-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    border-left: 3px solid var(--muted);
    transition: border-color 0.3s;
  }
  .pkg-id {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    color: var(--muted);
    margin-bottom: 3px;
  }
  .pkg-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.04em;
  }
  .pkg-meta {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: var(--muted);
    margin-top: 3px;
  }
  .pkg-status {
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    float: right;
    margin-top: -22px;
  }
  .active-tag { color: var(--accent); }
  .sorted-tag { color: var(--muted); }

  /* ── ACTIVE BOXES ── */
  .active-box-list { display: flex; flex-direction: column; gap: 6px; }
  .active-box {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 10px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-size: 11px;
  }
  .active-box-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
    animation: blink 1.5s infinite;
  }
  @keyframes blink {
    0%,100% { opacity: 1; }
    50% { opacity: 0.3; }
  }
  .active-box-info { flex: 1; }
  .active-box-coords {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    color: var(--muted);
  }
  .conf-badge {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    padding: 2px 6px;
    border-radius: 10px;
    background: rgba(255,255,255,0.05);
  }

  /* ── FRAGILE SHARE ── */
  .big-metric {
    text-align: center;
    padding: 8px 0;
  }
  .big-number {
    font-family: 'Space Mono', monospace;
    font-size: 42px;
    font-weight: 700;
    color: #fff;
    line-height: 1;
  }
  .big-unit {
    font-size: 18px;
    color: var(--muted);
  }
  .big-sub {
    font-size: 10px;
    color: var(--muted);
    margin-top: 4px;
    letter-spacing: 0.06em;
  }

  /* scrollbar for panels */
  .panel { padding-bottom: 24px; }
</style>
</head>
<body>

<!-- HEADER -->
<header>
  <div class="logo">
    <div class="logo-dot"></div>
    SORT<span style="color:var(--accent)">OS</span>
    <span style="font-size:11px;font-weight:400;color:var(--muted);margin-left:4px">Warehouse Intelligence</span>
  </div>
  <div class="header-meta">
    <span id="clock">--:--:--</span>
    <span id="fps-display">-- FPS</span>
    <span id="feed-status" class="status-pill status-online">ONLINE</span>
    <span id="session-time" style="color:var(--muted)">Session: 0m</span>
    <button class="reset-btn" onclick="resetSession()">RESET SESSION</button>
  </div>
</header>

<!-- MAIN -->
<div class="main">

  <!-- LEFT PANEL -->
  <div class="panel">

    <!-- Stats -->
    <div class="panel-section">
      <div class="section-label">Session Overview</div>
      <div class="stat-grid">
        <div class="stat-card">
          <div class="stat-value stat-accent" id="active-count">0</div>
          <div class="stat-label">Active on board</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="total-sorted">0</div>
          <div class="stat-label">Sorted this session</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="total-detected">0</div>
          <div class="stat-label">Total detected</div>
        </div>
        <div class="stat-card">
          <div class="stat-value" id="fragile-share" style="color:var(--red)">0%</div>
          <div class="stat-label">Fragile share</div>
        </div>
      </div>
    </div>

    <!-- Category inventory -->
    <div class="panel-section">
      <div class="section-label">Current Board Inventory</div>
      <div id="category-bars"></div>
    </div>

    <!-- Sorted counts -->
    <div class="panel-section">
      <div class="section-label">Sorted This Session</div>
      <div id="sorted-bars"></div>
    </div>

    <!-- Trend -->
    <div class="panel-section">
      <div class="section-label">Session Trend</div>
      <div class="chart-wrap">
        <canvas id="trendChart" width="308" height="90"></canvas>
      </div>
      <div style="display:flex;gap:16px;margin-top:8px;font-size:10px;color:var(--muted)">
        <span style="display:flex;align-items:center;gap:4px">
          <span style="width:16px;height:2px;background:var(--accent);display:inline-block"></span> Active
        </span>
        <span style="display:flex;align-items:center;gap:4px">
          <span style="width:16px;height:2px;background:var(--blue);display:inline-block"></span> Cumulative
        </span>
      </div>
    </div>

  </div>

  <!-- CENTER — LIVE FEED -->
  <div class="feed-panel">
    <div class="feed-header">
      <span class="feed-title">Live Feed — Workspace Camera</span>
      <div class="feed-meta">
        <span id="scan-time">Last scan: —</span>
        <span id="active-boxes-count">0 boxes in frame</span>
      </div>
    </div>
    <div class="feed-wrap">
      <img src="/video_feed" alt="Live Feed">
      <div class="feed-overlay">
        <div class="feed-tag" id="feed-fps">-- FPS</div>
        <div class="feed-tag" id="feed-boxes">0 detected</div>
      </div>
    </div>

    <!-- Active boxes below feed -->
    <div style="padding:12px 16px;border-top:1px solid var(--border);background:var(--surface);flex-shrink:0">
      <div class="section-label" style="margin-bottom:8px">Detected in Frame</div>
      <div class="active-box-list" id="active-box-list">
        <div style="color:var(--muted);font-size:11px;font-family:'Space Mono',monospace">No boxes detected</div>
      </div>
    </div>
  </div>

  <!-- RIGHT PANEL — PACKAGES -->
  <div class="panel" style="border-right:none">

    <!-- Fragile share big -->
    <div class="panel-section">
      <div class="section-label">Fragile Health</div>
      <div class="big-metric">
        <div class="big-number">
          <span id="fragile-big">0</span><span class="big-unit">%</span>
        </div>
        <div class="big-sub">Fragile packages sorted</div>
      </div>
    </div>

    <!-- Recent packages -->
    <div class="panel-section">
      <div class="section-label">Recent Detections</div>
      <div class="pkg-list" id="pkg-list">
        <div style="color:var(--muted);font-size:11px;font-family:'Space Mono',monospace">Waiting for packages...</div>
      </div>
    </div>

    <!-- Last sorted -->
    <div class="panel-section">
      <div class="section-label">Sorted Log</div>
      <div id="sorted-log" style="display:flex;flex-direction:column;gap:6px">
        <div style="color:var(--muted);font-size:11px;font-family:'Space Mono',monospace">No items sorted yet</div>
      </div>
    </div>

  </div>
</div>

<script>
const COLORS = {
  FRAGILE:   '#ef4444',
  ZONE_A:    '#3b82f6',
  ZONE_B:    '#eab308',
  IMMEDIATE: '#22c55e',
  DAMAGED:   '#f97316',
  UNKNOWN:   '#6b7280',
};
const LABELS = {
  FRAGILE:'Fragile', ZONE_A:'Zone A', ZONE_B:'Zone B',
  IMMEDIATE:'Immediate', DAMAGED:'Damaged', UNKNOWN:'Unknown'
};
const CATEGORIES = ['FRAGILE','ZONE_A','ZONE_B','IMMEDIATE','DAMAGED'];

let trendData = { labels:[], active:[], total:[] };
let sessionStart = Date.now();

// Clock
function updateClock() {
  const now = new Date();
  document.getElementById('clock').textContent =
    now.toTimeString().slice(0,8);
  const mins = Math.floor((Date.now()-sessionStart)/60000);
  document.getElementById('session-time').textContent =
    `Session: ${mins}m`;
}
setInterval(updateClock, 1000);
updateClock();

// Category bars
function renderBars(containerId, data, maxVal) {
  const el = document.getElementById(containerId);
  const max = maxVal || Math.max(1, ...Object.values(data));
  el.innerHTML = CATEGORIES.map(key => {
    const val = data[key] || 0;
    const pct = Math.min(100, (val / max) * 100);
    return `
      <div class="category-row">
        <div class="category-dot" style="background:${COLORS[key]}"></div>
        <div class="category-name">${LABELS[key]}</div>
        <div class="category-bar-wrap">
          <div class="category-bar-fill"
               style="width:${pct}%;background:${COLORS[key]}"></div>
        </div>
        <div class="category-count">${val}</div>
      </div>`;
  }).join('');
}

// Trend chart
const canvas  = document.getElementById('trendChart');
const ctx     = canvas.getContext('2d');
function drawTrend(trend) {
  const W = canvas.width, H = canvas.height;
  ctx.clearRect(0, 0, W, H);
  if (!trend || trend.length < 2) return;

  const actMax   = Math.max(1, ...trend.map(t=>t.active));
  const totMax   = Math.max(1, ...trend.map(t=>t.total));
  const n        = trend.length;

  function drawLine(vals, maxV, color) {
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth   = 1.5;
    ctx.shadowColor = color;
    ctx.shadowBlur  = 4;
    vals.forEach((v,i) => {
      const x = (i/(n-1)) * W;
      const y = H - (v/maxV) * (H-8) - 4;
      i===0 ? ctx.moveTo(x,y) : ctx.lineTo(x,y);
    });
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // Grid lines
  ctx.strokeStyle = '#1e2836';
  ctx.lineWidth   = 0.5;
  [0.25,0.5,0.75].forEach(f => {
    ctx.beginPath();
    ctx.moveTo(0, H*f); ctx.lineTo(W, H*f);
    ctx.stroke();
  });

  drawLine(trend.map(t=>t.active), actMax, '#00d4aa');
  drawLine(trend.map(t=>t.total),  totMax, '#3b82f6');
}

// Active boxes in feed
function renderActiveBoxes(boxes) {
  const el = document.getElementById('active-box-list');
  if (!boxes || boxes.length === 0) {
    el.innerHTML = '<div style="color:var(--muted);font-size:11px;font-family:\'Space Mono\',monospace">No boxes detected</div>';
    return;
  }
  el.innerHTML = boxes.map(b => `
    <div class="active-box">
      <div class="active-box-dot" style="background:${COLORS[b.label]||'#6b7280'}"></div>
      <div class="active-box-info">
        <div style="font-weight:600;font-size:11px;color:${COLORS[b.label]||'#6b7280'}">${LABELS[b.label]||b.label}</div>
        <div class="active-box-coords">(${b.x}, ${b.y}) cm · ${b.pkg_id}</div>
      </div>
      <div class="conf-badge">${b.confidence}%</div>
    </div>`).join('');
}

// Package list
function renderPackages(packages) {
  const el = document.getElementById('pkg-list');
  const entries = Object.entries(packages).reverse().slice(0,8);
  if (entries.length === 0) {
    el.innerHTML = '<div style="color:var(--muted);font-size:11px;font-family:\'Space Mono\',monospace">Waiting for packages...</div>';
    return;
  }
  el.innerHTML = entries.map(([id, pkg]) => `
    <div class="pkg-card" style="border-left-color:${COLORS[pkg.label]||'#6b7280'}">
      <div class="pkg-id">${id}
        <span class="pkg-status ${pkg.status==='sorted'?'sorted-tag':'active-tag'}">
          ${pkg.status==='sorted'?'SORTED':'ACTIVE'}
        </span>
      </div>
      <div class="pkg-label" style="color:${COLORS[pkg.label]||'#fff'}">${LABELS[pkg.label]||pkg.label}</div>
      <div class="pkg-meta">(${pkg.x?.toFixed(1)}, ${pkg.y?.toFixed(1)}) cm</div>
    </div>`).join('');
}

// Sorted log
function renderSortedLog(history) {
  const el = document.getElementById('sorted-log');
  if (!history || history.length === 0) {
    el.innerHTML = '<div style="color:var(--muted);font-size:11px;font-family:\'Space Mono\',monospace">No items sorted yet</div>';
    return;
  }
  el.innerHTML = [...history].reverse().slice(0,8).map(h => {
    const t = new Date(h.time).toTimeString().slice(0,8);
    return `
      <div style="display:flex;align-items:center;gap:8px;padding:6px 8px;
                  background:var(--bg);border:1px solid var(--border);border-radius:6px;
                  border-left:3px solid ${COLORS[h.label]||'#6b7280'}">
        <div style="flex:1">
          <div style="font-size:11px;font-weight:600;color:${COLORS[h.label]||'#fff'}">${LABELS[h.label]||h.label}</div>
          <div style="font-family:'Space Mono',monospace;font-size:9px;color:var(--muted)">${t} · (${h.x}, ${h.y})</div>
        </div>
        <div style="font-size:9px;color:var(--green);font-family:'Space Mono',monospace">✓ SORTED</div>
      </div>`;
  }).join('');
}

// Main state poll
async function fetchState() {
  try {
    const r    = await fetch('/api/state');
    const data = await r.json();

    // Header
    document.getElementById('fps-display').textContent  = `${data.fps} FPS`;
    document.getElementById('feed-fps').textContent     = `${data.fps} FPS`;
    const online = data.feed_status === 'ONLINE';
    const fsEl   = document.getElementById('feed-status');
    fsEl.textContent  = data.feed_status;
    fsEl.className    = 'status-pill ' + (online ? 'status-online' : 'status-offline');

    // Stats
    document.getElementById('active-count').textContent   = data.active_count;
    document.getElementById('total-sorted').textContent   = data.total_sorted;
    document.getElementById('total-detected').textContent = data.total_detected;
    document.getElementById('fragile-share').textContent  = data.fragile_share + '%';
    document.getElementById('fragile-big').textContent    = data.fragile_share;
    document.getElementById('feed-boxes').textContent     = `${data.active_count} detected`;
    document.getElementById('active-boxes-count').textContent =
      `${data.active_count} box${data.active_count!==1?'es':''} in frame`;

    if (data.last_scan) {
      const t = new Date(data.last_scan).toTimeString().slice(0,8);
      document.getElementById('scan-time').textContent = `Last scan: ${t}`;
    }

    // Bars
    const invMax = Math.max(1, ...Object.values(data.inventory));
    const srtMax = Math.max(1, ...Object.values(data.sorted_counts));
    renderBars('category-bars', data.inventory, invMax);
    renderBars('sorted-bars',   data.sorted_counts, srtMax);

    // Trend
    if (data.session_trend && data.session_trend.length > 1)
      drawTrend(data.session_trend);

    // Active boxes
    renderActiveBoxes(data.active_boxes);

    // Packages
    renderPackages(data.packages);

    // Sorted log
    renderSortedLog(data.sorted_history);

    // Session start
    if (data.session_start)
      sessionStart = new Date(data.session_start).getTime();

  } catch(e) {
    console.warn('State fetch failed:', e);
  }
}

async function resetSession() {
  if (!confirm('Reset session? All sorted counts will be cleared.')) return;
  await fetch('/api/reset', {method:'POST'});
  sessionStart = Date.now();
}

// Poll every 1.5 seconds
fetchState();
setInterval(fetchState, 1500);
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   SortOS — Package Intelligence Dashboard                ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║   Open browser:  http://localhost:{DASHBOARD_PORT}               ║")
    print("║   Dashboard auto-updates every 1.5 seconds               ║")
    print("║   Run main.py separately to control the robot            ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Start detection thread
    t = threading.Thread(target=detection_thread, daemon=True)
    t.start()

    # Start Flask
    app.run(host="0.0.0.0", port=DASHBOARD_PORT,
            debug=False, threaded=True)
