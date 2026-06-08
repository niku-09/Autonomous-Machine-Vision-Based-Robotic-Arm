"""
============================================================
 sortos.py — Autonomous Package Sorting Robot
 All-in-one: Web UI + Robot Control + Live Feed + Inventory
============================================================

 HOW TO RUN
 ----------
   python sortos.py

   Browser opens automatically at http://localhost:5000
   Everything is controlled from the browser.

 FEATURES
 --------
   - Live camera feed in browser
   - Type sorting commands in browser
   - Click SPACE to capture background
   - Real-time inventory tracking
   - Package log with timestamps
   - Session trend chart
   - Robot control — all from one tab

 REQUIRES
 --------
   pip install flask
   All files in same folder:
     detect_boxes.py, classify_box.py,
     robot_arm_IK.py, calibration.npz
   Arduino on COM4 with robot_arm_pick_place.ino
============================================================
"""

import sys, time, threading, json, math, webbrowser
from datetime import datetime
from collections import defaultdict
from flask import Flask, Response, jsonify, render_template_string, request

sys.path.append(".")

# ─────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────
SERIAL_PORT     = "COM4"
BAUD_RATE       = 9600
CAMERA_INDEX    = 0
PORT            = 5000
APPROACH_Z      = 13.0
GRASP_Z         = 1.5
GRIPPER_OPEN    = 90.0
GRIPPER_CLOSE   = 70.0
HOME_T1         = -45.0
SCAN_INTERVAL   = 2.5

DROP_ZONES = {
    "FRAGILE":   (25, -20),
    "ZONE_A":    (20, -22),
    "ZONE_B":    (10, -22),
    "IMMEDIATE": (25,  22),
    "DAMAGED":   (15,  22),
}

LABEL_COLORS_HEX = {
    "FRAGILE":   "#ef4444",
    "ZONE_A":    "#3b82f6",
    "ZONE_B":    "#eab308",
    "IMMEDIATE": "#22c55e",
    "DAMAGED":   "#f97316",
    "UNKNOWN":   "#6b7280",
}
LABEL_DISPLAY = {
    "FRAGILE":"Fragile","ZONE_A":"Zone A","ZONE_B":"Zone B",
    "IMMEDIATE":"Immediate","DAMAGED":"Damaged","UNKNOWN":"Unknown"
}
PROMPT_MAP = {
    "fragile":"FRAGILE","zone a":"ZONE_A","zone b":"ZONE_B",
    "immediate":"IMMEDIATE","damaged":"DAMAGED","all":"ALL"
}

# ─────────────────────────────────────────────────────────────
#  SHARED STATE
# ─────────────────────────────────────────────────────────────
state = {
    "session_start":      datetime.now().isoformat(),
    "robot_status":       "IDLE",
    "robot_msg":          "Waiting for command",
    "bg_captured":        False,
    "active_boxes":       [],
    "inventory":          {k:0 for k in LABEL_COLORS_HEX},
    "sorted_counts":      {k:0 for k in DROP_ZONES},
    "total_sorted":       0,
    "total_detected":     0,
    "sorted_history":     [],
    "session_trend":      [],
    "packages":           {},
    "pkg_counter":        0,
    "fps":                0.0,
    "feed_ok":            True,
    "last_scan":          None,
    "log":                [],
    # Feature 1: throughput
    "sort_timestamps":    [],   # timestamps of each sort for boxes/hr calc
    "throughput_bph":     0.0,  # boxes per hour
    # Feature 2: toasts (browser notifications)
    "toasts":             [],   # [{id, msg, type, t}]
    "toast_counter":      0,
    # Feature 3: confidence threshold + uncertain queue
    "conf_threshold":     60,   # % — below this goes to uncertain queue
    "uncertain_queue":    [],   # boxes needing manual review
}
state_lock   = threading.Lock()
latest_frame = None
frame_lock   = threading.Lock()
detector_ref = None  # global reference to BoxDetector

# Command queue from browser → robot thread
cmd_queue = []
cmd_lock  = threading.Lock()

def push_toast(msg, kind="info"):
    """Push a browser notification toast."""
    with state_lock:
        state["toast_counter"] += 1
        state["toasts"].insert(0, {
            "id":   state["toast_counter"],
            "msg":  msg,
            "type": kind,
            "t":    datetime.now().strftime("%H:%M:%S"),
        })
        if len(state["toasts"]) > 20:
            state["toasts"] = state["toasts"][:20]

def update_throughput():
    """Recalculate boxes per hour from recent sort timestamps."""
    now = time.time()
    # Keep only timestamps from last 60 minutes
    with state_lock:
        state["sort_timestamps"] = [t for t in state["sort_timestamps"] if now - t < 3600]
        n = len(state["sort_timestamps"])
        if n == 0:
            state["throughput_bph"] = 0.0
        elif n == 1:
            state["throughput_bph"] = round(n * 3600 / max(1, now - state["sort_timestamps"][0] + 1), 1)
        else:
            span = now - state["sort_timestamps"][-1]
            state["throughput_bph"] = round(n / max(1, span) * 3600, 1)

# ─────────────────────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────────────────────
def log(msg, level="info"):
    entry = {"t": datetime.now().strftime("%H:%M:%S"), "msg": msg, "level": level}
    with state_lock:
        state["log"].insert(0, entry)
        if len(state["log"]) > 100:
            state["log"] = state["log"][:100]
    print(f"[{entry['t']}] {msg}")

# ─────────────────────────────────────────────────────────────
#  SERIAL
# ─────────────────────────────────────────────────────────────
ser = None

def connect_serial():
    global ser
    try:
        import serial, serial.tools.list_ports
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=30)
        time.sleep(2)
        while ser.in_waiting:
            ser.readline()
        log(f"Arduino connected on {SERIAL_PORT}", "success")
        return True
    except Exception as e:
        log(f"Serial failed: {e} — simulation mode", "warn")
        return False

def send_cmd(t1, t2, t3, gripper, slow=False):
    s = 1 if slow else 0
    cmd = f"{t1:.2f},{t2:.2f},{t3:.2f},{gripper:.1f},{s}\n"
    if ser is None:
        return True
    try:
        ser.write(cmd.encode())
        ser.flush()
        resp = ser.readline().decode('utf-8', errors='replace').strip()
        return resp.startswith("OK:")
    except:
        return False

def move_ik(x, y, z, gripper, slow=False):
    from robot_arm_IK import inverse_kinematics
    result, err = inverse_kinematics(x, y, z)
    if result is None:
        log(f"IK fail: {err}", "error")
        return False
    return send_cmd(result[0], result[1], result[2], gripper, slow)

def move_home():
    log("→ Home: lift arm")
    send_cmd(0.0, 0.0, 90.0, GRIPPER_OPEN, slow=True)
    time.sleep(1.0)
    log("→ Home: twist out of view")
    send_cmd(HOME_T1, 0.0, 90.0, GRIPPER_OPEN, slow=False)
    time.sleep(0.5)

# ─────────────────────────────────────────────────────────────
#  PICK AND PLACE
# ─────────────────────────────────────────────────────────────
def pick_and_place_box(x, y, label):
    drop_x, drop_y = DROP_ZONES[label]
    log(f"Picking {label} at ({x:.1f},{y:.1f})")

    with state_lock:
        state["robot_status"] = "PICKING"
        state["robot_msg"]    = f"Picking {LABEL_DISPLAY[label]} at ({x:.1f},{y:.1f})"

    # Approach
    log("[1] Approach above box")
    if not move_ik(x, y, APPROACH_Z, GRIPPER_OPEN, slow=False): return False

    # Staged descent
    log("[2] Descend to grasp")
    if not move_ik(x, y, GRASP_Z, GRIPPER_OPEN, slow=True): return False

    # Close gripper
    log("[3] Close gripper")
    from robot_arm_IK import inverse_kinematics
    r, _ = inverse_kinematics(x, y, GRASP_Z)
    if r: send_cmd(r[0], r[1], r[2], GRIPPER_CLOSE, slow=True)
    time.sleep(0.8)

    # Staged lift
    log("[4] Staged lift")
    with state_lock:
        state["robot_msg"] = "Lifting box..."
    for zz in [4.0, 7.0, 10.0, APPROACH_Z]:
        if not move_ik(x, y, zz, GRIPPER_CLOSE, slow=True): return False
        time.sleep(0.1)

    # Move to drop zone
    log(f"[5] Move to {label} drop zone ({drop_x},{drop_y})")
    with state_lock:
        state["robot_msg"] = f"Moving to {LABEL_DISPLAY[label]} zone"
    if not move_ik(drop_x, drop_y, APPROACH_Z, GRIPPER_CLOSE, slow=False): return False

    # Descend at drop
    log("[6] Place descend")
    if not move_ik(drop_x, drop_y, GRASP_Z, GRIPPER_CLOSE, slow=True): return False

    # Open gripper
    log("[7] Release")
    r2, _ = inverse_kinematics(drop_x, drop_y, GRASP_Z)
    if r2: send_cmd(r2[0], r2[1], r2[2], GRIPPER_OPEN, slow=True)
    time.sleep(0.8)

    # Lift from drop
    log("[8] Lift from drop zone")
    move_ik(drop_x, drop_y, APPROACH_Z, GRIPPER_OPEN, slow=True)

    # Return home
    log("[9] Return home")
    with state_lock:
        state["robot_msg"] = "Returning home..."
    move_home()

    # Record + throughput + toast
    with state_lock:
        state["sorted_counts"][label] = state["sorted_counts"].get(label,0) + 1
        state["total_sorted"] += 1
        state["sort_timestamps"].append(time.time())
        state["sorted_history"].insert(0, {
            "label": label, "x": round(x,1), "y": round(y,1),
            "time": datetime.now().isoformat()
        })
    update_throughput()
    push_toast(f"{LABEL_DISPLAY[label]} box sorted to {LABEL_DISPLAY[label]} zone", "success")
    log(f"✓ {label} box placed in {LABEL_DISPLAY[label]} zone", "success")
    return True

# ─────────────────────────────────────────────────────────────
#  SORTING COMMAND
# ─────────────────────────────────────────────────────────────
def run_sort_command(target):
    from classify_box import classify_all_boxes
    log(f"Command received: sort {target}", "info")

    with state_lock:
        state["robot_status"] = "SCANNING"
        state["robot_msg"]    = "Moving to home, scanning..."
        bg = state["bg_captured"]
    if not bg:
        log("Background not captured! Press CAPTURE BACKGROUND first.", "error")
        with state_lock:
            state["robot_status"] = "IDLE"
            state["robot_msg"]    = "Error: capture background first"
        return

    move_home()
    time.sleep(1.0)

    # Grab frame and detect
    with frame_lock:
        frame_bytes = latest_frame

    if frame_bytes is None:
        log("No camera frame available", "error")
        with state_lock:
            state["robot_status"] = "IDLE"
            state["robot_msg"]    = "Error: no camera frame"
        return

    import cv2, numpy as np
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    with state_lock:
        boxes = state["active_boxes_raw"] if "active_boxes_raw" in state else []

    if detector_ref is None or detector_ref.background is None:
        log("Background not captured", "error")
        with state_lock:
            state["robot_status"] = "IDLE"
            state["robot_msg"]    = "Error: capture background first"
        return

    bg_frame = detector_ref.background

    if not boxes:
        log("No boxes detected on board", "warn")
        with state_lock:
            state["robot_status"] = "IDLE"
            state["robot_msg"]    = "No boxes found"
        return

    results = classify_all_boxes(frame, bg_frame, boxes)
    results.sort(key=lambda r: math.hypot(r["coords"][0], r["coords"][1]))

    with state_lock:
        conf_thresh = state["conf_threshold"]

    # Split by confidence threshold
    confident   = [r for r in results if r["confidence"] * 100 >= conf_thresh]
    uncertain   = [r for r in results if r["confidence"] * 100 <  conf_thresh]

    # Add uncertain boxes to manual review queue
    if uncertain:
        with state_lock:
            for r in uncertain:
                x, y, _ = r["coords"]
                state["uncertain_queue"].insert(0, {
                    "label":      r["label"],
                    "confidence": round(r["confidence"]*100),
                    "x": round(x,1), "y": round(y,1),
                    "time": datetime.now().isoformat(),
                })
                if len(state["uncertain_queue"]) > 20:
                    state["uncertain_queue"] = state["uncertain_queue"][:20]
        push_toast(f"{len(uncertain)} box(es) sent to uncertain queue (low confidence)", "warn")
        log(f"{len(uncertain)} box(es) below {conf_thresh}% confidence → manual review", "warn")

    if target == "ALL":
        targets = [r for r in confident if r["label"] != "UNKNOWN"]
    else:
        targets = [r for r in confident if r["label"] == target]

    if not targets:
        found = [r["label"] for r in results]
        log(f"No confident {target} boxes found. Detected: {found}", "warn")
        push_toast(f"No {target} boxes found above {conf_thresh}% confidence", "warn")
        with state_lock:
            state["robot_status"] = "IDLE"
            state["robot_msg"]    = f"No {target} boxes found"
        return

    log(f"Found {len(targets)} {target} box(es) — sorting nearest first", "info")

    picked = 0
    for i, box in enumerate(targets):
        x, y, _ = box["coords"]
        label   = box["label"]
        with state_lock:
            state["robot_msg"] = f"Picking box {i+1}/{len(targets)}"
        ok = pick_and_place_box(x, y, label)
        if ok:
            picked += 1
        else:
            log(f"Pick {i+1} failed — skipping", "error")
            move_home()
        time.sleep(0.3)

    with state_lock:
        state["robot_status"] = "IDLE"
        state["robot_msg"]    = f"Done — sorted {picked}/{len(targets)} boxes"
    log(f"Sorting complete: {picked}/{len(targets)} sorted", "success")

# ─────────────────────────────────────────────────────────────
#  CAMERA + DETECTION THREAD
# ─────────────────────────────────────────────────────────────
def camera_thread():
    global latest_frame, detector_ref
    import cv2, numpy as np
    try:
        from detect_boxes import BoxDetector
        from classify_box import classify_all_boxes
        detector = BoxDetector()
        detector_ref = detector  # expose to Flask routes
    except Exception as e:
        log(f"Detection init failed: {e}", "error")
        with state_lock:
            state["feed_ok"] = False
        return

    fps_t, fps_cnt = time.time(), 0
    last_scan_t    = 0
    log("Camera thread started", "info")

    with state_lock:
        state["feed_ok"] = True

    while True:
        try:
            ret, raw = detector.cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = detector.undistort(raw)

            # Draw active boxes
            disp = frame.copy()
            with state_lock:
                boxes_ui = state["active_boxes"]
            for b in boxes_ui:
                col = _hex_to_bgr(LABEL_COLORS_HEX.get(b.get("label","UNKNOWN"), "#6b7280"))
                bx,by,bw,bh = b.get("bx",0),b.get("by",0),b.get("bw",0),b.get("bh",0)
                cv2.rectangle(disp, (bx,by), (bx+bw,by+bh), col, 2)
                cv2.putText(disp, b.get("label","?"),
                            (bx, by-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
                cv2.circle(disp, (int(b.get("px",0)), int(b.get("py",0))), 6, col, -1)

            # FPS
            fps_cnt += 1
            if fps_cnt >= 15:
                fps = fps_cnt / (time.time() - fps_t)
                with state_lock:
                    state["fps"] = round(fps, 1)
                fps_cnt, fps_t = 0, time.time()

            with state_lock:
                fps_v = state["fps"]
            cv2.putText(disp, f"{fps_v:.0f} FPS",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,220,80), 2)

            # Encode
            _, buf = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 75])
            with frame_lock:
                latest_frame = buf.tobytes()

            # Periodic scan
            now = time.time()
            if now - last_scan_t >= SCAN_INTERVAL:
                last_scan_t = now
                with state_lock:
                    bg_ok = state["bg_captured"]
                if bg_ok:
                    _do_scan(detector, frame, classify_all_boxes)

        except Exception as e:
            log(f"Camera loop error: {e}", "error")
            time.sleep(0.5)


def _do_scan(detector, frame, classify_all_boxes):
    try:
        boxes, _ = detector.detect(visualise=False)
        with state_lock:
            state["active_boxes_raw"] = boxes
            state["last_scan"]        = datetime.now().isoformat()

        if not boxes:
            with state_lock:
                state["active_boxes"] = []
                state["inventory"]    = {k:0 for k in LABEL_COLORS_HEX}
            return

        bg_frame = detector.background
        if bg_frame is None:
            return

        results = classify_all_boxes(frame, bg_frame, boxes)
        active, counts = [], defaultdict(int)

        with state_lock:
            for r in results:
                x,y,z  = r["coords"]
                label   = r["label"]
                px,py   = int(r["pixel"][0]), int(r["pixel"][1])
                bx,by,bw,bh = r["rect"]
                counts[label] += 1
                pkg_id = _find_or_create_pkg(x, y, label)
                active.append({
                    "pkg_id":label,"label":label,
                    "confidence":round(r["confidence"]*100),
                    "x":round(x,1),"y":round(y,1),
                    "px":px,"py":py,"bx":bx,"by":by,"bw":bw,"bh":bh,
                })
            state["active_boxes"]   = active
            state["total_detected"] = max(state["total_detected"], len(active))
            for k in state["inventory"]:
                state["inventory"][k] = counts.get(k, 0)
            _update_trend()
    except Exception as e:
        pass


def _find_or_create_pkg(x, y, label):
    for pid, p in state["packages"].items():
        if p["status"]=="active" and abs(p["x"]-x)<5 and abs(p["y"]-y)<5:
            p["label"] = label; p["x"]=x; p["y"]=y
            return pid
    state["pkg_counter"] += 1
    pid = f"PKG-{state['pkg_counter']:04d}"
    state["packages"][pid] = {"label":label,"x":x,"y":y,
                               "first_seen":datetime.now().isoformat(),"status":"active"}
    return pid


def _update_trend():
    state["session_trend"].append({
        "t":      datetime.now().strftime("%H:%M:%S"),
        "active": len(state["active_boxes"]),
        "total":  state["total_sorted"],
    })
    if len(state["session_trend"]) > 60:
        state["session_trend"] = state["session_trend"][-60:]


def _hex_to_bgr(h):
    h = h.lstrip('#')
    r,g,b = int(h[0:2],16),int(h[2:4],16),int(h[4:6],16)
    return (b,g,r)

# ─────────────────────────────────────────────────────────────
#  ROBOT COMMAND THREAD
# ─────────────────────────────────────────────────────────────
def robot_thread():
    while True:
        with cmd_lock:
            if cmd_queue:
                cmd = cmd_queue.pop(0)
            else:
                cmd = None
        if cmd:
            try:
                run_sort_command(cmd)
            except Exception as e:
                log(f"Robot error: {e}", "error")
                with state_lock:
                    state["robot_status"] = "IDLE"
                    state["robot_msg"]    = f"Error: {e}"
        time.sleep(0.1)

# ─────────────────────────────────────────────────────────────
#  FLASK
# ─────────────────────────────────────────────────────────────
app = Flask(__name__)


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/video_feed")
def video_feed():
    return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


def _gen():
    while True:
        with frame_lock:
            f = latest_frame
        if f:
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
        time.sleep(0.033)


@app.route("/api/state")
def api_state():
    with state_lock:
        d = {
            "robot_status":   state["robot_status"],
            "robot_msg":      state["robot_msg"],
            "bg_captured":    state["bg_captured"],
            "active_count":   len(state["active_boxes"]),
            "active_boxes":   state["active_boxes"],
            "total_sorted":   state["total_sorted"],
            "total_detected": state["total_detected"],
            "inventory":      state["inventory"],
            "sorted_counts":  state["sorted_counts"],
            "sorted_history": state["sorted_history"][:15],
            "session_trend":  state["session_trend"][-30:],
            "fps":            state["fps"],
            "feed_ok":        state["feed_ok"],
            "last_scan":      state["last_scan"],
            "session_start":  state["session_start"],
            "log":            state["log"][:20],
            "fragile_pct":    round(state["sorted_counts"].get("FRAGILE",0)/max(1,state["total_sorted"])*100),
            "packages":       dict(list(state["packages"].items())[-8:]),
            "throughput_bph":  state["throughput_bph"],
            "uncertain_count": len(state["uncertain_queue"]),
            "conf_threshold":  state["conf_threshold"],
            "toasts":          list(state["toasts"]),
        }
    return jsonify(d)


@app.route("/api/capture_bg", methods=["POST"])
def api_capture_bg():
    global detector_ref
    if detector_ref is None:
        return jsonify({"ok":False,"msg":"Camera not ready yet"})
    try:
        detector_ref.capture_background()
        with state_lock:
            state["bg_frame"]    = detector_ref.background.copy()
            state["bg_captured"] = True
        log("Background captured — board cleared", "success")
        return jsonify({"ok":True})
    except Exception as e:
        return jsonify({"ok":False,"msg":str(e)})


@app.route("/api/command", methods=["POST"])
def api_command():
    data   = request.json or {}
    prompt = data.get("prompt","").lower().strip()
    for kw, label in PROMPT_MAP.items():
        if kw in prompt:
            with state_lock:
                busy = state["robot_status"] != "IDLE"
            if busy:
                return jsonify({"ok":False,"msg":"Robot is busy"})
            with cmd_lock:
                cmd_queue.append(label)
            log(f"Queued command: {label}", "info")
            return jsonify({"ok":True,"label":label})
    return jsonify({"ok":False,"msg":f"Unrecognised command: '{prompt}'"})


@app.route("/api/home", methods=["POST"])
def api_home():
    with state_lock:
        busy = state["robot_status"] != "IDLE"
    if busy:
        return jsonify({"ok":False,"msg":"Robot is busy"})
    threading.Thread(target=move_home, daemon=True).start()
    return jsonify({"ok":True})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    with state_lock:
        state["sorted_counts"]   = {k:0 for k in DROP_ZONES}
        state["total_sorted"]    = 0
        state["total_detected"]  = 0
        state["sorted_history"]  = []
        state["session_trend"]   = []
        state["packages"]        = {}
        state["pkg_counter"]     = 0
        state["session_start"]   = datetime.now().isoformat()
        state["log"]             = []
        state["sort_timestamps"] = []
        state["throughput_bph"]  = 0.0
        state["uncertain_queue"] = []
        state["toasts"]          = []
    log("Session reset", "info")
    return jsonify({"ok":True})


@app.route("/api/toasts")
def api_toasts():
    with state_lock:
        t = list(state["toasts"])
    return jsonify(t)


@app.route("/api/set_threshold", methods=["POST"])
def api_set_threshold():
    data = request.json or {}
    val  = int(data.get("threshold", 60))
    val  = max(0, min(100, val))
    with state_lock:
        state["conf_threshold"] = val
    log(f"Confidence threshold set to {val}%", "info")
    return jsonify({"ok":True, "threshold": val})


@app.route("/api/uncertain_queue")
def api_uncertain():
    with state_lock:
        q = list(state["uncertain_queue"])
    return jsonify(q)


@app.route("/api/dismiss_uncertain", methods=["POST"])
def api_dismiss_uncertain():
    with state_lock:
        state["uncertain_queue"] = []
    return jsonify({"ok":True})

# ─────────────────────────────────────────────────────────────
#  HTML
# ─────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Sort OS — Warehouse Intelligence Platform</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
/* ═══════════════════════════════════════════════════════════
   SORT OS — Premium Industrial Warehouse Intelligence UI
   Light industrial palette · Teal accents · Refined depth
   ═══════════════════════════════════════════════════════════ */

/* ─── LIGHT THEME (DEFAULT) ──────────────────────────── */
:root,[data-theme="light"]{
  --bg:        #e9ecf1;
  --bg2:       #e1e5eb;
  --surf:      #f3f5f8;
  --surf2:     #edf0f4;
  --surf3:     #e5e9ef;
  --border:    #d4d9e2;
  --border-h:  #bfc5d0;
  --text:      #1a1f2e;
  --text2:     #4e5668;
  --muted:     #7e8a9e;
  --accent:    #0d9488;
  --accent2:   #0284c7;
  --accent-dim:rgba(13,148,136,.08);
  --accent-glow:rgba(13,148,136,.18);
  --red:       #dc2626; --red-dim:rgba(220,38,38,.07);
  --blue:      #2563eb; --blue-dim:rgba(37,99,235,.07);
  --yellow:    #d97706; --yellow-dim:rgba(217,119,6,.07);
  --green:     #0d9488; --green-dim:rgba(13,148,136,.07);
  --orange:    #ea580c; --orange-dim:rgba(234,88,12,.07);
  --grey:      #6b7280;
  --header-h:  56px;
  --shadow-xs: 0 1px 2px rgba(0,0,0,.05);
  --shadow:    0 1px 3px rgba(0,0,0,.07), 0 1px 2px rgba(0,0,0,.04);
  --shadow-md: 0 4px 12px rgba(0,0,0,.07), 0 1px 3px rgba(0,0,0,.04);
  --shadow-lg: 0 10px 28px rgba(0,0,0,.09), 0 4px 10px rgba(0,0,0,.04);
  --radius:    10px;
  --radius-lg: 14px;
  --font:      'DM Sans', system-ui, -apple-system, sans-serif;
  --mono:      'JetBrains Mono', ui-monospace, monospace;
  --feed-bg:   #0f1419;
}

/* ─── DARK THEME ─────────────────────────────────────── */
[data-theme="dark"]{
  --bg:        #0e1117;
  --bg2:       #151921;
  --surf:      #1a1f2b;
  --surf2:     #212735;
  --surf3:     #282f3e;
  --border:    #2d3548;
  --border-h:  #3d465c;
  --text:      #e4e8f0;
  --text2:     #9aa3b8;
  --muted:     #606c84;
  --accent:    #2dd4bf;
  --accent2:   #38bdf8;
  --accent-dim:rgba(45,212,191,.1);
  --accent-glow:rgba(45,212,191,.22);
  --red:       #f87171; --red-dim:rgba(248,113,113,.1);
  --blue:      #60a5fa; --blue-dim:rgba(96,165,250,.1);
  --yellow:    #fbbf24; --yellow-dim:rgba(251,191,36,.1);
  --green:     #2dd4bf; --green-dim:rgba(45,212,191,.1);
  --orange:    #fb923c; --orange-dim:rgba(251,146,60,.1);
  --grey:      #6b7585;
  --shadow-xs: 0 1px 2px rgba(0,0,0,.2);
  --shadow:    0 2px 8px rgba(0,0,0,.3);
  --shadow-md: 0 4px 16px rgba(0,0,0,.35);
  --shadow-lg: 0 10px 32px rgba(0,0,0,.45);
  --feed-bg:   #060810;
}

/* ─── RESET ──────────────────────────────────────────── */
*{box-sizing:border-box;margin:0;padding:0;}
body{
  background:var(--bg);color:var(--text);
  font-family:var(--font);
  height:100vh;
  display:grid;
  grid-template-rows:var(--header-h) 1fr;
  overflow:hidden;
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
}
::-webkit-scrollbar{width:5px;}
::-webkit-scrollbar-track{background:transparent;}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:var(--border-h);}
::selection{background:var(--accent-dim);color:var(--accent);}

/* ─── HEADER ──────────────────────────────────────────── */
header{
  display:flex;align-items:center;justify-content:space-between;
  padding:0 20px;
  background:var(--surf);
  border-bottom:1px solid var(--border);
  z-index:100;gap:14px;
  box-shadow:var(--shadow-xs);
}
.logo{display:flex;align-items:center;gap:12px;white-space:nowrap;}
.logo-mark{
  width:36px;height:36px;
  background:linear-gradient(140deg,var(--accent),var(--accent2));
  border-radius:9px;display:flex;align-items:center;justify-content:center;
  box-shadow:0 2px 10px var(--accent-glow);
  position:relative;flex-shrink:0;
}
.logo-mark::after{
  content:'';position:absolute;inset:-2px;border-radius:11px;
  background:linear-gradient(140deg,var(--accent),var(--accent2));
  filter:blur(10px);opacity:.3;z-index:-1;
}
.logo-mark svg{width:20px;height:20px;}
.logo-sub{
  font-size:9.5px;font-weight:700;color:var(--text2);
  letter-spacing:.18em;text-transform:uppercase;line-height:1;
  margin-bottom:1px;
}
.logo-main{font-size:19px;font-weight:800;letter-spacing:.06em;line-height:1.1;}
.logo-main span{color:var(--accent);}

.hmeta{display:flex;gap:8px;align-items:center;font-family:var(--mono);font-size:10px;}
.clock{color:var(--text2);font-size:10px;letter-spacing:.03em;font-weight:500;}
.pill{
  padding:3px 10px;border-radius:20px;font-size:8px;font-weight:700;
  letter-spacing:.08em;text-transform:uppercase;border:1px solid;
  transition:all .2s;
}
.pill-ok{background:var(--green-dim);color:var(--green);border-color:rgba(13,148,136,.18);}
.pill-err{background:var(--red-dim);color:var(--red);border-color:rgba(220,38,38,.18);}
.pill-busy{background:var(--yellow-dim);color:var(--yellow);border-color:rgba(217,119,6,.18);}
.hbtn{
  padding:6px 14px;
  background:var(--surf2);
  border:1px solid var(--border);
  color:var(--text2);border-radius:8px;
  font-family:var(--mono);font-size:8px;font-weight:600;
  cursor:pointer;letter-spacing:.07em;transition:all .18s;text-transform:uppercase;
}
.hbtn:hover{background:var(--surf3);color:var(--text);border-color:var(--border-h);box-shadow:var(--shadow-xs);}
.hbtn.danger:hover{background:var(--red-dim);color:var(--red);border-color:rgba(220,38,38,.25);}
.theme-toggle{
  width:32px;height:32px;border-radius:8px;
  background:var(--surf2);border:1px solid var(--border);
  display:flex;align-items:center;justify-content:center;
  cursor:pointer;transition:all .18s;font-size:15px;color:var(--text2);
}
.theme-toggle:hover{background:var(--surf3);color:var(--text);box-shadow:var(--shadow-xs);}

/* ─── MAIN GRID ───────────────────────────────────────── */
.main{
  display:grid;
  grid-template-columns:290px 1fr 280px;
  overflow:hidden;
  background:var(--bg);
  gap:0;
}
.panel{
  overflow-y:auto;
  scrollbar-width:thin;
  background:var(--bg);
  padding:8px;
  display:flex;flex-direction:column;gap:8px;
}
.card{
  background:var(--surf);
  border:1px solid var(--border);
  border-radius:var(--radius-lg);
  overflow:visible;
  transition:border-color .22s, box-shadow .22s;
  box-shadow:var(--shadow-xs);
}
.card:hover{border-color:var(--border-h);box-shadow:var(--shadow);}
.card-head{
  padding:10px 14px;
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:9px;
  background:var(--surf2);
  border-radius:var(--radius-lg) var(--radius-lg) 0 0;
}
.card-title{
  font-size:10px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--muted);font-family:var(--mono);
  flex:1;
}
.card-body{padding:10px 14px;}
.card-icon{width:15px;height:15px;color:var(--accent);opacity:.7;flex-shrink:0;}

/* ─── COMMAND ─────────────────────────────────────────── */
.cmd-wrap{display:flex;gap:6px;margin-bottom:8px;}
.cmd-input{
  flex:1;background:var(--bg);
  border:1px solid var(--border);
  color:var(--text);padding:8px 12px;border-radius:var(--radius);
  font-family:var(--mono);font-size:11px;
  outline:none;transition:all .22s;
}
.cmd-input:focus{
  border-color:var(--accent);
  box-shadow:0 0 0 3px var(--accent-dim);
}
.cmd-input::placeholder{color:var(--muted);}
.cmd-btn{
  padding:8px 16px;
  background:var(--accent);
  color:#fff;border:none;border-radius:var(--radius);
  font-family:var(--font);font-size:11px;font-weight:700;
  cursor:pointer;transition:all .22s;white-space:nowrap;
  box-shadow:0 2px 8px var(--accent-glow);
}
.cmd-btn:hover{filter:brightness(1.12);box-shadow:0 4px 16px var(--accent-glow);transform:translateY(-1px);}
.cmd-btn:active{transform:translateY(0);}
.cmd-btn:disabled{opacity:.35;cursor:not-allowed;box-shadow:none;filter:none;transform:none;}

.cmd-chips{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;}
.chip{
  padding:5px 12px;border-radius:20px;font-size:9px;font-weight:700;
  cursor:pointer;border:1px solid;letter-spacing:.04em;
  transition:all .18s;background:transparent;font-family:var(--mono);
}
.chip:hover{transform:translateY(-1px);box-shadow:var(--shadow);}

/* ─── STATUS BAR ──────────────────────────────────────── */
.status-bar{
  padding:9px 16px;background:var(--surf);
  border-bottom:1px solid var(--border);
  display:flex;align-items:center;gap:9px;font-size:11px;
}
.status-icon{width:7px;height:7px;border-radius:50%;flex-shrink:0;transition:all .3s;}
.status-idle{background:var(--muted);}
.status-working{background:var(--yellow);animation:pulse 1.2s infinite;box-shadow:0 0 10px var(--yellow-dim);}
.status-done{background:var(--green);box-shadow:0 0 10px var(--green-dim);}
@keyframes pulse{0%,100%{opacity:1;transform:scale(1)}50%{opacity:.35;transform:scale(.65)}}

/* ─── STATS ───────────────────────────────────────────── */
.stat-grid{display:grid;grid-template-columns:1fr 1fr;gap:6px;}
.stat{
  background:var(--bg);
  border:1px solid var(--border);
  border-radius:var(--radius);
  padding:10px 12px;
  transition:all .18s;
  position:relative;
}
.stat:hover{border-color:var(--border-h);box-shadow:var(--shadow-xs);}
.stat-v{
  font-family:var(--mono);font-size:24px;
  font-weight:700;color:var(--text);line-height:1;
}
.stat-l{font-size:8px;color:var(--muted);margin-top:4px;
        letter-spacing:.05em;text-transform:uppercase;font-family:var(--mono);font-weight:500;}

/* ─── CATEGORY BARS ───────────────────────────────────── */
.cat-row{display:flex;align-items:center;gap:6px;margin-bottom:5px;}
.cat-dot{width:6px;height:6px;border-radius:3px;flex-shrink:0;}
.cat-name{font-size:10px;font-weight:600;flex:1;color:var(--text2);}
.cat-bar{flex:2;background:var(--bg);border-radius:4px;height:5px;overflow:hidden;}
.cat-fill{height:100%;border-radius:4px;transition:width .6s cubic-bezier(.4,0,.2,1);}
.cat-n{font-family:var(--mono);font-size:10px;
       color:var(--text);width:22px;text-align:right;font-weight:700;}

/* ─── FEED PANEL ──────────────────────────────────────── */
.feed-panel{display:flex;flex-direction:column;overflow:hidden;background:var(--bg);border-left:1px solid var(--border);border-right:1px solid var(--border);}
.feed-hdr{
  padding:12px 16px;border-bottom:1px solid var(--border);background:var(--surf);
  display:flex;align-items:center;justify-content:space-between;flex-shrink:0;
}
.feed-title{
  font-size:11px;font-weight:700;letter-spacing:.1em;
  text-transform:uppercase;color:var(--text);font-family:var(--mono);
  display:flex;align-items:center;gap:8px;
}
.feed-title .live-dot{
  width:8px;height:8px;border-radius:50%;background:var(--green);
  animation:pulse 1.8s infinite;box-shadow:0 0 6px var(--green-dim);
}
.feed-meta{font-family:var(--mono);font-size:9px;color:var(--muted);display:flex;gap:14px;font-weight:500;}
.feed-img{
  flex:1;overflow:hidden;background:var(--feed-bg);position:relative;
}
.feed-img img{width:100%;height:100%;object-fit:contain;}
.feed-tag{
  position:absolute;top:10px;left:10px;
  background:rgba(10,14,20,.8);
  border:1px solid rgba(45,212,191,.15);
  padding:4px 10px;border-radius:6px;
  font-family:var(--mono);font-size:9px;color:#2dd4bf;font-weight:600;
  backdrop-filter:blur(10px);
}
.feed-bottom{
  padding:10px 16px;border-top:1px solid var(--border);
  background:var(--surf);flex-shrink:0;max-height:140px;overflow-y:auto;
}

/* ─── KPI BAR ─────────────────────────────────────────── */
.kpi-bar{
  display:grid;grid-template-columns:repeat(4,1fr);gap:0;
  border-bottom:1px solid var(--border);flex-shrink:0;
}
.kpi{
  text-align:center;padding:12px 6px;
  background:var(--surf);
  transition:all .18s;
  border-right:1px solid var(--border);
  position:relative;
}
.kpi:last-child{border-right:none;}
.kpi:hover{background:var(--surf2);}
.kpi-v{
  font-family:var(--mono);font-size:20px;
  font-weight:700;color:var(--text);line-height:1;
}
.kpi-l{font-size:8px;color:var(--muted);margin-top:4px;
       letter-spacing:.08em;text-transform:uppercase;font-family:var(--mono);font-weight:500;}

/* ─── ACTIVE BOXES ────────────────────────────────────── */
.abox{
  display:flex;align-items:center;gap:8px;
  padding:7px 10px;background:var(--bg);
  border:1px solid var(--border);border-radius:8px;
  margin-bottom:5px;font-size:11px;transition:all .15s;
}
.abox:hover{border-color:var(--border-h);box-shadow:var(--shadow-xs);}
.abox-dot{width:8px;height:8px;border-radius:50%;flex-shrink:0;animation:pulse 2s infinite;}
.abox-info{flex:1;}
.abox-coords{font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:1px;}
.conf{font-family:var(--mono);font-size:9px;
      padding:3px 8px;border-radius:6px;background:var(--surf2);color:var(--muted);font-weight:600;
      border:1px solid var(--border);}

/* ─── TREND CHART ─────────────────────────────────────── */
.chart-wrap{
  height:85px;position:relative;
  background:var(--bg);border-radius:8px;padding:8px;
  border:1px solid var(--border);
}

/* ─── LOG ─────────────────────────────────────────────── */
.log-entry{
  padding:6px 10px;border-radius:7px;font-family:var(--mono);
  font-size:10px;margin-bottom:3px;display:flex;gap:8px;align-items:flex-start;
  background:var(--bg);border:1px solid var(--border);
  transition:all .12s;
}
.log-entry:hover{border-color:var(--border-h);}
.log-t{color:var(--muted);flex-shrink:0;font-weight:500;}
.log-info{color:var(--text2);}
.log-success{color:var(--green);}
.log-warn{color:var(--yellow);}
.log-error{color:var(--red);}

/* ─── SORTED LOG ──────────────────────────────────────── */
.sort-entry{
  display:flex;align-items:center;gap:8px;
  padding:7px 10px;background:var(--bg);
  border:1px solid var(--border);border-radius:8px;
  border-left:3px solid var(--muted);margin-bottom:5px;
  transition:all .18s;
}
.sort-entry:hover{border-color:var(--border-h);box-shadow:var(--shadow-xs);}

/* ─── PACKAGE CARD ────────────────────────────────────── */
.pkg{
  background:var(--bg);border:1px solid var(--border);border-radius:var(--radius);
  padding:10px 12px;margin-bottom:6px;border-left:3px solid var(--muted);
  transition:all .18s;
}
.pkg:hover{border-color:var(--border-h);box-shadow:var(--shadow-xs);}
.pkg-id{font-family:var(--mono);font-size:9px;color:var(--muted);margin-bottom:3px;font-weight:500;}
.pkg-label{font-size:12px;font-weight:700;color:var(--text);}
.pkg-meta{font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:3px;}

/* ─── TOAST ───────────────────────────────────────────── */
.toast-container{position:fixed;top:64px;right:18px;z-index:9999;
                 display:flex;flex-direction:column;gap:8px;pointer-events:none;}
.toast{
  padding:12px 16px;border-radius:var(--radius);font-size:12px;font-weight:600;
  border:1px solid;display:flex;align-items:center;gap:10px;
  animation:toastIn .35s cubic-bezier(.34,1.56,.64,1),toastOut .35s ease 3.5s forwards;
  pointer-events:none;min-width:240px;max-width:340px;
  box-shadow:var(--shadow-lg);backdrop-filter:blur(14px);
}
.toast-success{background:rgba(13,148,136,.08);color:var(--green);border-color:rgba(13,148,136,.18);}
.toast-warn{background:rgba(217,119,6,.08);color:var(--yellow);border-color:rgba(217,119,6,.18);}
.toast-error{background:rgba(220,38,38,.08);color:var(--red);border-color:rgba(220,38,38,.18);}
.toast-info{background:rgba(37,99,235,.08);color:var(--blue);border-color:rgba(37,99,235,.18);}
@keyframes toastIn{from{transform:translateX(120%) scale(.92);opacity:0}to{transform:none;opacity:1}}
@keyframes toastOut{from{opacity:1}to{opacity:0;transform:translateX(120%)}}

/* ─── SLIDER ──────────────────────────────────────────── */
.slider-wrap{
  margin-top:6px;
  background:linear-gradient(135deg, var(--bg), var(--surf2));
  border:1px solid var(--border);border-radius:var(--radius);padding:9px 10px;
}
.slider-label{display:flex;justify-content:space-between;
              font-size:9px;color:var(--text2);margin-bottom:6px;
              font-family:var(--mono);font-weight:600;}
input[type=range]{
  width:100%;-webkit-appearance:none;height:5px;
  border-radius:5px;
  background:linear-gradient(90deg, #bcd4e6 0%, #d1dce6 40%, #c8d0da 100%);
  outline:none;
  transition:all .15s;
}
[data-theme="dark"] input[type=range]{
  background:linear-gradient(90deg, #2d3548 0%, #3d465c 100%);
}
input[type=range]::-webkit-slider-thumb{
  -webkit-appearance:none;width:18px;height:18px;border-radius:50%;
  background:linear-gradient(135deg, var(--accent), var(--accent2));
  cursor:pointer;
  border:2.5px solid #fff;
  box-shadow:0 1px 6px var(--accent-glow), 0 0 0 1px var(--accent-dim);
  transition:all .18s;
}
input[type=range]::-webkit-slider-thumb:hover{
  transform:scale(1.18);
  box-shadow:0 2px 10px var(--accent-glow), 0 0 0 3px var(--accent-dim);
}
input[type=range]::-moz-range-track{
  height:5px;border-radius:5px;
  background:linear-gradient(90deg, #bcd4e6 0%, #d1dce6 40%, #c8d0da 100%);
}
input[type=range]::-moz-range-thumb{
  width:18px;height:18px;border-radius:50%;border:2.5px solid #fff;
  background:linear-gradient(135deg, var(--accent), var(--accent2));
  cursor:pointer;
  box-shadow:0 1px 6px var(--accent-glow);
}

/* ─── UNCERTAIN BADGE ─────────────────────────────────── */
.uncertain-badge{
  background:var(--yellow-dim);color:var(--yellow);
  border:1px solid rgba(217,119,6,.18);border-radius:10px;
  padding:2px 8px;font-size:9px;font-weight:700;margin-left:4px;
}

/* ─── BG CAPTURE ──────────────────────────────────────── */
.bg-btn{
  width:100%;padding:8px;background:var(--bg);
  border:1.5px dashed var(--border);color:var(--muted);
  border-radius:var(--radius);font-family:var(--font);
  font-size:10px;font-weight:600;cursor:pointer;
  transition:all .22s;margin-bottom:8px;
}
.bg-btn:hover{border-color:var(--accent);color:var(--accent);background:var(--accent-dim);}
.bg-btn.captured{border-color:var(--green);color:var(--green);
                 border-style:solid;background:var(--green-dim);}

/* ─── EMPTY STATE ─────────────────────────────────────── */
.empty{color:var(--muted);font-size:10px;font-family:var(--mono);padding:6px 0;font-weight:500;}

/* ─── DISMISS BTN ─────────────────────────────────────── */
.dismiss-btn{
  margin-top:8px;width:100%;padding:7px;background:transparent;
  border:1px solid var(--yellow);color:var(--yellow);border-radius:8px;
  font-size:9px;cursor:pointer;font-family:var(--mono);font-weight:600;
  letter-spacing:.05em;transition:all .18s;text-transform:uppercase;
}
.dismiss-btn:hover{background:var(--yellow-dim);box-shadow:var(--shadow-xs);}

/* ─── RESPONSIVE ──────────────────────────────────────── */
@media(max-width:1200px){
  .main{grid-template-columns:265px 1fr 255px;}
}
@media(max-width:1000px){
  .main{grid-template-columns:1fr;grid-template-rows:auto 1fr auto;}
  .panel{max-height:35vh;}
  .feed-panel{border-left:none;border-right:none;}
}
@media(max-width:700px){
  header{padding:0 10px;gap:8px;}
  .hmeta{gap:5px;flex-wrap:wrap;}
  .logo-main{font-size:16px;}
  .panel{padding:6px;}
}
</style>
</head>
<body>

<!-- TOAST CONTAINER -->
<div class="toast-container" id="toast-container"></div>

<header>
  <div class="logo">
    <div class="logo-mark">
      <!-- Sort OS: Robotic arm reaching down to grasp box on ground -->
      <svg viewBox="-22 -28 64 54" fill="none" xmlns="http://www.w3.org/2000/svg">
        <rect x="-20" y="20" width="40" height="4" rx="1.5" fill="#fff" opacity="0.9"/>
        <path d="M-6 20 L-8 14 L8 14 L6 20Z" fill="#fff" opacity="0.7"/>
        <circle cx="0" cy="11" r="5" fill="none" stroke="#fff" stroke-width="1.8"/>
        <circle cx="0" cy="11" r="2" fill="#fff"/>
        <line x1="0" y1="9" x2="-8" y2="-4" stroke="#fff" stroke-width="4" stroke-linecap="round"/>
        <circle cx="-8" cy="-4" r="3.5" fill="none" stroke="#fff" stroke-width="1.5"/>
        <circle cx="-8" cy="-4" r="1.5" fill="#fff"/>
        <line x1="-8" y1="-4" x2="12" y2="6" stroke="#fff" stroke-width="3.2" stroke-linecap="round"/>
        <circle cx="12" cy="6" r="2.5" fill="none" stroke="#fff" stroke-width="1.2"/>
        <line x1="12" y1="6" x2="9" y2="12" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
        <line x1="9" y1="12" x2="9.5" y2="16" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="12" y1="6" x2="17" y2="12" stroke="#fff" stroke-width="1.8" stroke-linecap="round"/>
        <line x1="17" y1="12" x2="16.5" y2="16" stroke="#fff" stroke-width="1.5" stroke-linecap="round"/>
        <rect x="7" y="16" width="12" height="9" rx="1.2" fill="none" stroke="#fff" stroke-width="1.4"/>
        <line x1="13" y1="16" x2="13" y2="25" stroke="#fff" stroke-width="0.6" opacity="0.45"/>
        <line x1="7" y1="20" x2="19" y2="20" stroke="#fff" stroke-width="0.6" opacity="0.45"/>
      </svg>
    </div>
    <div>
      <div class="logo-sub">Warehouse Management System</div>
      <div class="logo-main">SORT <span>OS</span></div>
    </div>
  </div>
  <div class="hmeta">
    <span id="clock" class="clock">--:--:--</span>
    <span id="fps-hdr" style="color:var(--muted)">-- FPS</span>
    <span id="feed-pill" class="pill pill-ok">FEED OK</span>
    <span id="robot-pill" class="pill pill-ok">IDLE</span>
    <span id="session-t" style="color:var(--muted)">0m</span>
    <button class="hbtn" onclick="goHome()">HOME</button>
    <button class="hbtn danger" onclick="resetSession()">RESET</button>
    <div class="theme-toggle" onclick="toggleTheme()" title="Toggle theme">&#9681;</div>
  </div>
</header>

<div class="main">

<!-- LEFT PANEL -->
<div class="panel">
  <div class="card">
    <div class="card-head">
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="3"/><path d="M12 1v4M12 19v4M4.22 4.22l2.83 2.83M16.95 16.95l2.83 2.83M1 12h4M19 12h4M4.22 19.78l2.83-2.83M16.95 7.05l2.83-2.83"/></svg>
      <span class="card-title">Command Centre</span>
    </div>
    <div class="card-body">
      <button class="bg-btn" id="bg-btn" onclick="captureBg()">
        + Capture Background &mdash; clear board first
      </button>
      <div class="cmd-wrap">
        <input class="cmd-input" id="cmd-in" placeholder="e.g. sort fragile, sort all..."
               onkeydown="if(event.key==='Enter')sendCommand()">
        <button class="cmd-btn" id="cmd-btn" onclick="sendCommand()">Sort &rarr;</button>
      </div>
      <div class="slider-wrap">
        <div class="slider-label">
          <span>Confidence Threshold</span>
          <span id="thresh-val" style="color:var(--accent);font-weight:700">60%</span>
        </div>
        <input type="range" id="thresh-slider" min="0" max="100" value="60"
               oninput="updateThreshold(this.value)">
        <div style="font-size:8px;color:var(--muted);margin-top:3px;font-weight:500">
          Below threshold &rarr; Uncertain queue
        </div>
      </div>
      <div class="cmd-chips" style="margin-top:8px">
        <div class="chip" style="color:var(--red);border-color:var(--red);background:var(--red-dim)"
             onclick="quickSort('sort fragile')">Fragile</div>
        <div class="chip" style="color:var(--blue);border-color:var(--blue);background:var(--blue-dim)"
             onclick="quickSort('sort zone a')">Zone A</div>
        <div class="chip" style="color:var(--yellow);border-color:var(--yellow);background:var(--yellow-dim)"
             onclick="quickSort('sort zone b')">Zone B</div>
        <div class="chip" style="color:var(--green);border-color:var(--green);background:var(--green-dim)"
             onclick="quickSort('sort immediate')">Immediate</div>
        <div class="chip" style="color:var(--orange);border-color:var(--orange);background:var(--orange-dim)"
             onclick="quickSort('sort damaged')">Damaged</div>
        <div class="chip" style="color:var(--accent2);border-color:var(--accent2);background:rgba(2,132,199,.06)"
             onclick="quickSort('sort all')">Sort All</div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></svg>
      <span class="card-title">Session Overview</span>
    </div>
    <div class="card-body">
      <div class="stat-grid">
        <div class="stat"><div class="stat-v" id="s-active" style="color:var(--accent)">0</div><div class="stat-l">On Board</div></div>
        <div class="stat"><div class="stat-v" id="s-sorted">0</div><div class="stat-l">Sorted</div></div>
        <div class="stat"><div class="stat-v" id="s-detected">0</div><div class="stat-l">Detected</div></div>
        <div class="stat"><div class="stat-v" id="s-fragile" style="color:var(--red)">0%</div><div class="stat-l">Fragile %</div></div>
      </div>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"/><path d="M9 12l2 2 4-4"/></svg>
      <span class="card-title">Live Inventory</span>
    </div>
    <div class="card-body" id="inv-bars"></div>
  </div>

  <div class="card">
    <div class="card-head">
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
      <span class="card-title">Session Output</span>
    </div>
    <div class="card-body" id="sorted-bars"></div>
  </div>

  <div class="card">
    <div class="card-head">
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>
      <span class="card-title">Trend</span>
    </div>
    <div class="card-body">
      <div class="chart-wrap">
        <canvas id="tc" width="258" height="72"></canvas>
      </div>
    </div>
  </div>
</div>

<!-- CENTRE — LIVE FEED -->
<div class="feed-panel">
  <div class="feed-hdr">
    <span class="feed-title"><span class="live-dot"></span> Live Camera Feed</span>
    <div class="feed-meta">
      <span id="scan-t">Scan: &mdash;</span>
      <span id="boxes-n">0 boxes</span>
    </div>
  </div>
  <div class="status-bar">
    <div class="status-icon status-idle" id="status-icon"></div>
    <span id="status-msg" style="color:var(--muted);font-size:12px;font-weight:500">Waiting for command</span>
  </div>
  <div class="kpi-bar">
    <div class="kpi">
      <div class="kpi-v" id="kpi-bph" style="color:var(--accent)">0</div>
      <div class="kpi-l">Boxes / Hour</div>
    </div>
    <div class="kpi">
      <div class="kpi-v" id="kpi-sorted">0</div>
      <div class="kpi-l">Total Sorted</div>
    </div>
    <div class="kpi">
      <div class="kpi-v" id="kpi-active" style="color:var(--yellow)">0</div>
      <div class="kpi-l">On Board</div>
    </div>
    <div class="kpi">
      <div class="kpi-v" id="kpi-uncertain" style="color:var(--orange)">0</div>
      <div class="kpi-l">Uncertain</div>
    </div>
  </div>
  <div class="feed-img">
    <img src="/video_feed" alt="Live Feed">
    <div class="feed-tag" id="fps-feed">-- FPS</div>
  </div>
  <div class="feed-bottom">
    <div class="card-title" style="margin-bottom:8px;font-size:9px">Detected in Frame</div>
    <div id="abox-list"><div class="empty">No boxes detected</div></div>
  </div>
</div>

<!-- RIGHT PANEL -->
<div class="panel">
  <div class="card">
    <div class="card-head">
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>
      <span class="card-title">Activity Log</span>
    </div>
    <div class="card-body" style="max-height:200px;overflow-y:auto" id="log-list"></div>
  </div>

  <div class="card">
    <div class="card-head">
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="9 11 12 14 22 4"/><path d="M21 12v7a2 2 0 01-2 2H5a2 2 0 01-2-2V5a2 2 0 012-2h11"/></svg>
      <span class="card-title">Sorted Log</span>
    </div>
    <div class="card-body" style="max-height:180px;overflow-y:auto" id="sorted-log"></div>
  </div>

  <div class="card">
    <div class="card-head">
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="color:var(--yellow)"><path d="M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
      <span class="card-title">Uncertain Queue</span>
      <span class="uncertain-badge" id="uncertain-badge" style="display:none">0</span>
    </div>
    <div class="card-body">
      <div id="uncertain-list"><div class="empty">No uncertain detections</div></div>
      <button onclick="dismissUncertain()" id="dismiss-btn" class="dismiss-btn" style="display:none">
        DISMISS ALL
      </button>
    </div>
  </div>

  <div class="card">
    <div class="card-head">
      <svg class="card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 00-1-1.73l-7-4a2 2 0 00-2 0l-7 4A2 2 0 003 8v8a2 2 0 001 1.73l7 4a2 2 0 002 0l7-4A2 2 0 0021 16z"/></svg>
      <span class="card-title">Package Tracker</span>
    </div>
    <div class="card-body" style="max-height:200px;overflow-y:auto" id="pkg-list"></div>
  </div>
</div>

</div>

<script>
var C={FRAGILE:'#dc2626',ZONE_A:'#2563eb',ZONE_B:'#d97706',IMMEDIATE:'#0d9488',DAMAGED:'#ea580c',UNKNOWN:'#6b7280'};
var L={FRAGILE:'Fragile',ZONE_A:'Zone A',ZONE_B:'Zone B',IMMEDIATE:'Immediate',DAMAGED:'Damaged',UNKNOWN:'Unknown'};
var CATS=['FRAGILE','ZONE_A','ZONE_B','IMMEDIATE','DAMAGED'];
var sessionStart=Date.now();

/* Sync colors with current theme */
function themeColors(){
  var s=getComputedStyle(document.documentElement);
  C.FRAGILE   = s.getPropertyValue('--red').trim();
  C.ZONE_A    = s.getPropertyValue('--blue').trim();
  C.ZONE_B    = s.getPropertyValue('--yellow').trim();
  C.IMMEDIATE = s.getPropertyValue('--green').trim();
  C.DAMAGED   = s.getPropertyValue('--orange').trim();
  C.UNKNOWN   = s.getPropertyValue('--grey').trim();
}
themeColors();

// ── THEME ────────────────────────────────────────────────
function toggleTheme(){
  var t=document.documentElement.getAttribute('data-theme');
  document.documentElement.setAttribute('data-theme',t==='dark'?'light':'dark');
  localStorage.setItem('sortos-theme',t==='dark'?'light':'dark');
  setTimeout(themeColors,50);
}
(function(){var s=localStorage.getItem('sortos-theme');if(s){document.documentElement.setAttribute('data-theme',s);setTimeout(themeColors,50);}})();

// ── CLOCK ────────────────────────────────────────────────
setInterval(function(){
  document.getElementById('clock').textContent=new Date().toTimeString().slice(0,8);
  document.getElementById('session-t').textContent=Math.floor((Date.now()-sessionStart)/60000)+'m';
},1000);

// ── CORE FUNCTIONS ────────────────────────────────────────
async function sendCommand(){
  var input=document.getElementById('cmd-in');
  var prompt=input.value.trim();
  if(!prompt)return;
  try{
    var r=await fetch('/api/command',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({prompt:prompt})
    });
    var d=await r.json();
    if(!d.ok)alert(d.msg);
    else input.value='';
  }catch(e){alert('Network error');}
}

function quickSort(cmd){
  document.getElementById('cmd-in').value=cmd;
  sendCommand();
}

async function goHome(){
  try{await fetch('/api/home',{method:'POST'});}catch(e){alert('Network error');}
}

async function resetSession(){
  if(!confirm('Reset session?'))return;
  try{await fetch('/api/reset',{method:'POST'});}catch(e){alert('Network error');}
}

// ── BARS ─────────────────────────────────────────────────
function renderBars(id,data,max){
  var el=document.getElementById(id);
  var m=max||Math.max(1,...Object.values(data));
  el.innerHTML=CATS.map(function(k){return '<div class="cat-row">'+
    '<div class="cat-dot" style="background:'+C[k]+'"></div>'+
    '<div class="cat-name">'+L[k]+'</div>'+
    '<div class="cat-bar"><div class="cat-fill" style="width:'+Math.min(100,(data[k]||0)/m*100)+'%;background:'+C[k]+'"></div></div>'+
    '<div class="cat-n">'+(data[k]||0)+'</div></div>';
  }).join('');
}

// ── TREND ────────────────────────────────────────────────
var canvas=document.getElementById('tc'),ctx=canvas.getContext('2d');
function drawTrend(trend){
  var W=canvas.width,H=canvas.height;
  ctx.clearRect(0,0,W,H);
  if(!trend||trend.length<2)return;
  var n=trend.length;
  var aMax=Math.max(1,...trend.map(function(t){return t.active;}));
  var tMax=Math.max(1,...trend.map(function(t){return t.total;}));
  ctx.strokeStyle=getComputedStyle(document.documentElement).getPropertyValue('--border').trim();
  ctx.lineWidth=0.5;
  [.25,.5,.75].forEach(function(f){ctx.beginPath();ctx.moveTo(0,H*f);ctx.lineTo(W,H*f);ctx.stroke();});
  var sets=[
    [trend.map(function(t){return t.active;}),aMax,getComputedStyle(document.documentElement).getPropertyValue('--accent').trim()],
    [trend.map(function(t){return t.total;}),tMax,getComputedStyle(document.documentElement).getPropertyValue('--accent2').trim()]
  ];
  sets.forEach(function(s){
    var vals=s[0],mx=s[1],col=s[2];
    ctx.beginPath();ctx.strokeStyle=col;ctx.lineWidth=1.8;ctx.lineJoin='round';ctx.lineCap='round';
    vals.forEach(function(v,i){var x=i/(n-1)*W,y=H-(v/mx)*(H-8)-4;i?ctx.lineTo(x,y):ctx.moveTo(x,y);});
    ctx.stroke();
  });
}

// ── ACTIVE BOXES ─────────────────────────────────────────
function renderABoxes(boxes){
  var el=document.getElementById('abox-list');
  if(!boxes||!boxes.length){el.innerHTML='<div class="empty">No boxes detected</div>';return;}
  el.innerHTML=boxes.map(function(b){return '<div class="abox">'+
    '<div class="abox-dot" style="background:'+(C[b.label]||'#6b7280')+'"></div>'+
    '<div class="abox-info">'+
      '<div style="font-weight:700;color:'+(C[b.label]||'var(--text)')+'">'+
        (L[b.label]||b.label)+'</div>'+
      '<div class="abox-coords">('+b.x+', '+b.y+') cm</div>'+
    '</div>'+
    '<div class="conf">'+b.confidence+'%</div></div>';}).join('');
}

// ── LOG ──────────────────────────────────────────────────
function renderLog(log){
  document.getElementById('log-list').innerHTML=(log||[]).map(function(e){
    return '<div class="log-entry"><span class="log-t">'+e.t+'</span><span class="log-'+e.level+'">'+e.msg+'</span></div>';
  }).join('')||'<div class="empty">No activity yet</div>';
}

// ── SORTED LOG ───────────────────────────────────────────
function renderSortedLog(hist){
  document.getElementById('sorted-log').innerHTML=(hist||[]).map(function(h){
    var t=new Date(h.time).toTimeString().slice(0,8);
    return '<div class="sort-entry" style="border-left-color:'+(C[h.label]||'#6b7280')+'">'+
      '<div style="flex:1">'+
        '<div style="font-size:11px;font-weight:700;color:'+(C[h.label]||'var(--text)')+'">'+
          (L[h.label]||h.label)+'</div>'+
        '<div style="font-family:var(--mono);font-size:9px;color:var(--muted);margin-top:2px">'+
          t+' &middot; ('+h.x+', '+h.y+') cm</div>'+
      '</div>'+
      '<div style="font-size:11px;color:var(--green);font-weight:700">&#10003;</div></div>';
  }).join('')||'<div class="empty">Nothing sorted yet</div>';
}

// ── PACKAGES ─────────────────────────────────────────────
function renderPkgs(pkgs){
  var entries=Object.entries(pkgs||{}).reverse().slice(0,6);
  document.getElementById('pkg-list').innerHTML=entries.map(function(e){
    var id=e[0],p=e[1];
    var statusBg=p.status==='sorted'?'var(--green-dim)':'var(--accent-dim)';
    var statusCol=p.status==='sorted'?'var(--green)':'var(--accent)';
    var statusText=p.status==='sorted'?'SORTED':'ACTIVE';
    return '<div class="pkg" style="border-left-color:'+(C[p.label]||'#6b7280')+'">'+
      '<div class="pkg-id">'+id+
        '<span style="float:right;background:'+statusBg+';color:'+statusCol+
              ';padding:2px 8px;border-radius:10px;font-size:8px;font-weight:700;letter-spacing:.06em">'+
          statusText+'</span></div>'+
      '<div class="pkg-label" style="color:'+(C[p.label]||'var(--text)')+';margin-top:3px">'+
        (L[p.label]||p.label)+'</div>'+
      '<div class="pkg-meta">('+((p.x||0).toFixed(1))+', '+((p.y||0).toFixed(1))+') cm</div></div>';
  }).join('')||'<div class="empty">No packages tracked</div>';
}

// ── STATUS ───────────────────────────────────────────────
function updateStatus(status,msg){
  var icon=document.getElementById('status-icon');
  var msgEl=document.getElementById('status-msg');
  var pill=document.getElementById('robot-pill');
  msgEl.textContent=msg;
  icon.className='status-icon';
  if(status==='IDLE'){icon.classList.add('status-idle');pill.textContent='IDLE';pill.className='pill pill-ok';}
  else if(status==='SCANNING'||status==='PICKING'){icon.classList.add('status-working');pill.textContent='BUSY';pill.className='pill pill-busy';}
  else{icon.classList.add('status-done');pill.textContent='DONE';pill.className='pill pill-ok';}
}

// ── ANIMATED COUNTERS ────────────────────────────────────
function animateCount(el, newVal, prefix, suffix) {
  prefix=prefix||''; suffix=suffix||'';
  var current = parseFloat(el.dataset.val || '0');
  if (current === newVal) return;
  el.dataset.val = newVal;
  var diff = newVal - current;
  var steps = 18;
  var stepVal = diff / steps;
  var cur = current, i = 0;
  var iv = setInterval(function() {
    cur += stepVal; i++;
    el.textContent = prefix + (Number.isInteger(newVal) ? Math.round(cur) : cur.toFixed(1)) + suffix;
    if (i >= steps) { clearInterval(iv); el.textContent = prefix + newVal + suffix; }
  }, 25);
}

// ── TOAST NOTIFICATIONS ──────────────────────────────────
var shownToasts = {};
function renderToasts(toasts) {
  var container = document.getElementById('toast-container');
  (toasts || []).forEach(function(t) {
    if (shownToasts[t.id]) return;
    shownToasts[t.id] = true;
    var el = document.createElement('div');
    el.className = 'toast toast-'+(t.type||'info');
    var icons = {success:'\u2713',warn:'\u26a0',error:'\u2715',info:'\u2139'};
    el.innerHTML = '<span>'+(icons[t.type]||'\u2139')+'</span><span>'+t.msg+'</span>';
    container.appendChild(el);
    setTimeout(function() { el.remove(); }, 4000);
  });
}

// ── KPIs ─────────────────────────────────────────────────
function updateKPIs(d) {
  animateCount(document.getElementById('kpi-bph'), d.throughput_bph || 0);
  animateCount(document.getElementById('kpi-sorted'), d.total_sorted || 0);
  animateCount(document.getElementById('kpi-active'), d.active_count || 0);
  animateCount(document.getElementById('kpi-uncertain'), d.uncertain_count || 0);
}

// ── CONFIDENCE THRESHOLD ─────────────────────────────────
function updateThreshold(val) {
  document.getElementById('thresh-val').textContent = val + '%';
  clearTimeout(window._threshTimer);
  window._threshTimer = setTimeout(async function() {
    await fetch('/api/set_threshold', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({threshold: parseInt(val)})
    });
  }, 400);
}

function renderUncertain(count) {
  var badge   = document.getElementById('uncertain-badge');
  var btn     = document.getElementById('dismiss-btn');
  var listEl  = document.getElementById('uncertain-list');
  if (count > 0) {
    badge.textContent = count;
    badge.style.display = 'inline';
    btn.style.display   = 'block';
  } else {
    badge.style.display = 'none';
    btn.style.display   = 'none';
    listEl.innerHTML    = '<div class="empty">No uncertain detections</div>';
  }
}

async function dismissUncertain() {
  await fetch('/api/dismiss_uncertain', {method:'POST'});
}

function syncThreshold(thresh) {
  var slider = document.getElementById('thresh-slider');
  if (parseInt(slider.value) !== thresh) {
    slider.value = thresh;
    document.getElementById('thresh-val').textContent = thresh + '%';
  }
}

// ── BACKGROUND CAPTURE ───────────────────────────────────
async function captureBg(){
  var btn = document.getElementById('bg-btn');
  btn.textContent = 'Capturing...';
  try {
    var r = await fetch('/api/capture_bg', {method:'POST'});
    var d = await r.json();
    if(d.ok){
      btn.className = 'bg-btn captured';
      btn.textContent = '\u2713  Background Ready';
    } else {
      btn.textContent = '+ Capture Background \u2014 clear board first';
      alert('Error: ' + d.msg);
    }
  } catch(e) {
    btn.textContent = '+ Capture Background \u2014 clear board first';
    alert('Network error');
  }
}

// ── MAIN STATE POLL ──────────────────────────────────────
async function fetchState(){
  try{
    var r=await fetch('/api/state');
    var d=await r.json();
    document.getElementById('fps-hdr').textContent=d.fps+' FPS';
    document.getElementById('fps-feed').textContent=d.fps+' FPS';
    document.getElementById('feed-pill').textContent=d.feed_ok?'FEED OK':'OFFLINE';
    document.getElementById('feed-pill').className='pill '+(d.feed_ok?'pill-ok':'pill-err');

    animateCount(document.getElementById('s-active'), d.active_count);
    animateCount(document.getElementById('s-sorted'), d.total_sorted);
    animateCount(document.getElementById('s-detected'), d.total_detected);
    document.getElementById('s-fragile').textContent = d.fragile_pct+'%';

    document.getElementById('boxes-n').textContent=d.active_count+' box'+(d.active_count!==1?'es':'');
    if(d.last_scan)document.getElementById('scan-t').textContent='Scan: '+new Date(d.last_scan).toTimeString().slice(0,8);

    renderBars('inv-bars',d.inventory);
    renderBars('sorted-bars',d.sorted_counts);
    if(d.session_trend&&d.session_trend.length>1)drawTrend(d.session_trend);
    renderABoxes(d.active_boxes);
    renderLog(d.log);
    renderSortedLog(d.sorted_history);
    renderPkgs(d.packages);
    updateStatus(d.robot_status,d.robot_msg);

    renderToasts(d.toasts);
    updateKPIs(d);
    renderUncertain(d.uncertain_count || 0);
    syncThreshold(d.conf_threshold || 60);

    var busy=d.robot_status!=='IDLE';
    document.getElementById('cmd-btn').disabled=busy;
    var bgBtn=document.getElementById('bg-btn');
    if(d.bg_captured && !bgBtn.classList.contains('captured')){
      bgBtn.className='bg-btn captured';
      bgBtn.textContent='\u2713  Background Ready';
    }
    if(d.session_start)sessionStart=new Date(d.session_start).getTime();
  }catch(e){console.warn(e);}
}

fetchState();
setInterval(fetchState,1500);
</script>
</body>
</html>"""

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   SortOS — Autonomous Package Sorting Robot              ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║   Opening browser at http://localhost:{PORT}               ║")
    print("║   Everything is controlled from the browser              ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Connect serial
    connect_serial()

    # Start camera thread
    threading.Thread(target=camera_thread, daemon=True).start()

    # Start robot command thread
    threading.Thread(target=robot_thread, daemon=True).start()

    # Open browser after short delay
    def open_browser():
        time.sleep(1.5)
        webbrowser.open(f"http://localhost:{PORT}")
    threading.Thread(target=open_browser, daemon=True).start()

    # Run Flask
    log("SortOS starting...", "info")
    app.run(host="0.0.0.0", port=PORT, debug=False, threaded=True)
