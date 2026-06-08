"""
============================================================
 main.py — Autonomous Package Sorting Robot
 Complete Orchestrator
============================================================

 USAGE
 -----
   python main.py

 PROMPT EXAMPLES
 ---------------
   "sort fragile"         → picks all FRAGILE (red) boxes
   "sort zone a"          → picks all ZONE A  (blue) boxes
   "sort zone b"          → picks all ZONE B  (yellow) boxes
   "sort immediate"       → picks all IMMEDIATE (green) boxes
   "sort damaged"         → picks all DAMAGED (orange) boxes
   "sort all"             → sorts everything to respective zones
   "q" or "quit"          → exit

 DROP ZONES (robot frame, cm)
 ----------------------------
   FRAGILE:   (25, -20)
   ZONE_A:    (20, -22)
   ZONE_B:    (10, -22)
   IMMEDIATE: (25, +22)
   DAMAGED:   (15, +22)

 SEQUENCE PER BOX
 ----------------
   1. Home (twisted right, out of camera)
   2. Detect + classify ALL boxes
   3. Filter by prompt, sort nearest first
   4. For each matching box:
        a. Move to (x, y, APPROACH_Z)   — normal speed
        b. Descend to (x, y, GRASP_Z)   — slow
        c. Close gripper                 — slow
        d. Lift to (x, y, APPROACH_Z)   — slow
        e. Move to drop zone approach    — normal speed
        f. Descend to drop height        — slow
        g. Open gripper                  — slow
        h. Lift from drop zone           — slow
        i. Return HOME (twisted right)   — normal speed
        j. Re-detect remaining boxes

 REQUIRES
 --------
   detect_boxes.py, classify_box.py, robot_arm_IK.py
   calibration.npz, robot_arm_pick_place.ino on Arduino
   pip install pyserial opencv-python numpy
============================================================
"""

import math
import time
import sys
import cv2
import numpy as np

sys.path.append(".")
from detect_boxes  import BoxDetector
from classify_box  import classify_all_boxes, LABEL_COLORS
from robot_arm_IK  import inverse_kinematics

# Optional dashboard integration
try:
    from dashboard_server import record_sorted as _dashboard_record
    DASHBOARD_ACTIVE = True
    print("[INFO] Dashboard integration active")
except ImportError:
    DASHBOARD_ACTIVE = False

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed — simulation mode")

# ─────────────────────────────────────────────────────────────
#  CONFIGURE THESE
# ─────────────────────────────────────────────────────────────
SERIAL_PORT   = "COM4"
BAUD_RATE     = 9600

APPROACH_Z    = 13.0    # cm — safe travel height (above shoulder)
GRASP_Z       =  1.5    # cm — grasp height (working fine)
GRIPPER_OPEN  = 90.0    # degrees
GRIPPER_CLOSE = 70.0    # degrees — confirmed for 3cm cardboard box

# Home = arm twisted RIGHT, out of camera field of view
HOME_T1 = -45.0    # degrees — twist right
HOME_T2 =   0.0
HOME_T3 =  90.0

# Drop zones (x, y) in robot frame cm — z fixed at GRASP_Z
DROP_ZONES = {
    "FRAGILE":   (25, -20),
    "ZONE_A":    (20, -22),
    "ZONE_B":    (10, -22),
    "IMMEDIATE": (25,  22),
    "DAMAGED":   (15,  22),
}

# Approach height for drop zones
DROP_APPROACH_Z = 13.0   # cm

# Pause times
PAUSE_GRIP  = 0.8   # seconds after closing/opening gripper
PAUSE_MOVE  = 0.2   # seconds after each movement

# ─────────────────────────────────────────────────────────────
#  PROMPT PARSING
# ─────────────────────────────────────────────────────────────
PROMPT_MAP = {
    "fragile":   "FRAGILE",
    "zone a":    "ZONE_A",
    "zone b":    "ZONE_B",
    "immediate": "IMMEDIATE",
    "damaged":   "DAMAGED",
    "all":       "ALL",
}

def parse_prompt(text):
    text = text.lower().strip()
    for keyword, label in PROMPT_MAP.items():
        if keyword in text:
            return label
    return None

# ─────────────────────────────────────────────────────────────
#  SERIAL
# ─────────────────────────────────────────────────────────────
def connect_serial():
    if not SERIAL_AVAILABLE:
        return None
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=30)
        time.sleep(2)
        while ser.in_waiting:
            msg = ser.readline().decode('utf-8', errors='replace').strip()
            if msg: print(f"  [Arduino] {msg}")
        print(f"  [INFO] Connected on {SERIAL_PORT}")
        return ser
    except Exception as e:
        print(f"  [WARN] Serial failed: {e} — simulation mode")
        return None


def send_cmd(ser, t1, t2, t3, gripper, slow=False):
    """Send T1,T2,T3,G,S to Arduino."""
    s = 1 if slow else 0
    cmd = f"{t1:.2f},{t2:.2f},{t3:.2f},{gripper:.1f},{s}\n"
    if ser is None:
        print(f"    [SIM] {cmd.strip()}")
        return True
    ser.write(cmd.encode())
    ser.flush()
    resp = ser.readline().decode('utf-8', errors='replace').strip()
    if resp.startswith("OK:"):
        return True
    print(f"    [ERR] {resp}")
    return False


def move_ik(ser, x, y, z, gripper, label="", slow=False):
    """Solve IK and move arm."""
    result, err = inverse_kinematics(x, y, z)
    if result is None:
        print(f"    [IK FAIL] {label}: {err}")
        return False
    t1, t2, t3 = result
    speed = "SLOW" if slow else "FAST"
    print(f"    [{label}|{speed}] ({x:.1f},{y:.1f},{z:.1f})cm "
          f"θ1={t1:.1f}° θ2={t2:.1f}° θ3={t3:.1f}°")
    ok = send_cmd(ser, t1, t2, t3, gripper, slow=slow)
    time.sleep(PAUSE_MOVE)
    return ok


def move_home(ser):
    """
    Two-step home to avoid mechanical conflict:
      Step A: Lift arm straight up first (theta1 stays, forearm/upper go to safe angles)
      Step B: Then twist base to HOME_T1 (-45 deg, out of camera view)
    This prevents servo fighting and beeping.
    """
    print("  → HOME step A: lift arm straight up...")
    # First lift to safe vertical position (no base twist yet)
    send_cmd(ser, 0.0, 0.0, 90.0, GRIPPER_OPEN, slow=True)
    time.sleep(1.0)
    print("  → HOME step B: twist base out of camera view...")
    # Then twist base only — forearm/upper already safe
    send_cmd(ser, HOME_T1, HOME_T2, HOME_T3, GRIPPER_OPEN, slow=False)
    time.sleep(0.5)
    return True

# ─────────────────────────────────────────────────────────────
#  PICK AND PLACE SEQUENCE
# ─────────────────────────────────────────────────────────────
def pick_and_place_box(ser, x, y, label):
    """
    Complete pick and place for one box at robot coords (x, y).
    """
    drop_x, drop_y = DROP_ZONES[label]
    print(f"\n  ── Picking {label} box at ({x:.1f}, {y:.1f}) ──")
    print(f"     Drop zone: ({drop_x}, {drop_y})")

    # ── 1. Approach above box ─────────────────────────────────
    print("  [1] Approach above box")
    ok = move_ik(ser, x, y, APPROACH_Z, GRIPPER_OPEN,
                 "Approach", slow=False)
    if not ok: return False

    # ── 2. Descend slowly to grasp height ────────────────────
    print("  [2] Descend to grasp height")
    ok = move_ik(ser, x, y, GRASP_Z, GRIPPER_OPEN,
                 "Descend", slow=True)
    if not ok: return False

    # ── 3. Close gripper — grab box ──────────────────────────
    print("  [3] Close gripper")
    result, _ = inverse_kinematics(x, y, GRASP_Z)
    if result:
        send_cmd(ser, result[0], result[1], result[2],
                 GRIPPER_CLOSE, slow=True)
    time.sleep(PAUSE_GRIP)

    # ── 4. Staged lift — small steps to avoid torque overload ───
    print("  [4] Staged lift (z=1.5 → 4 → 7 → 10 → 13 cm)")
    lift_stages = [4.0, 7.0, 10.0, APPROACH_Z]
    for stage_z in lift_stages:
        print(f"    [lift stage] z={stage_z:.0f} cm...")
        ok = move_ik(ser, x, y, stage_z, GRIPPER_CLOSE,
                     f"Lift z={stage_z:.0f}", slow=True)
        if not ok: return False
        time.sleep(0.1)

    # ── 5. Swing to drop zone approach height ─────────────────
    print(f"  [5] Move to {label} drop zone")
    ok = move_ik(ser, drop_x, drop_y, DROP_APPROACH_Z, GRIPPER_CLOSE,
                 "Swing to drop", slow=False)
    if not ok: return False

    # ── 6. Descend slowly at drop zone ───────────────────────
    print("  [6] Descend to place")
    ok = move_ik(ser, drop_x, drop_y, GRASP_Z, GRIPPER_CLOSE,
                 "Place descend", slow=True)
    if not ok: return False

    # ── 7. Open gripper — release ────────────────────────────
    print("  [7] Open gripper — release")
    result2, _ = inverse_kinematics(drop_x, drop_y, GRASP_Z)
    if result2:
        send_cmd(ser, result2[0], result2[1], result2[2],
                 GRIPPER_OPEN, slow=True)
    time.sleep(PAUSE_GRIP)

    # ── 8. Lift from drop zone ────────────────────────────────
    print("  [8] Lift from drop zone")
    move_ik(ser, drop_x, drop_y, DROP_APPROACH_Z, GRIPPER_OPEN,
            "Lift from drop", slow=True)

    # ── 9. Return home ────────────────────────────────────────
    print("  [9] Return HOME")
    move_home(ser)

    print(f"  ✓ {label} box placed successfully!")
    return True

# ─────────────────────────────────────────────────────────────
#  DETECTION + CLASSIFICATION
# ─────────────────────────────────────────────────────────────
def scan_and_classify(detector):
    """
    Capture one frame, detect all boxes, classify each.
    Returns list of classified box dicts, sorted nearest first.
    """
    # Let camera settle
    for _ in range(5):
        detector.cap.read()

    ret, frame = detector.cap.read()
    if not ret:
        print("[ERR] Frame grab failed")
        return []

    undist = detector.undistort(frame)
    boxes, debug = detector.detect(visualise=True)

    if not boxes:
        print("[SCAN] No boxes detected on board.")
        if debug is not None:
            cv2.imshow("Scan", debug)
            cv2.waitKey(500)
        return []

    # Classify all boxes
    results = classify_all_objects(undist, detector.background, boxes, debug)
    return results


def classify_all_objects(undist, background, boxes, debug_frame):
    """Classify all boxes and annotate the debug frame."""
    results = classify_all_boxes(undist, background, boxes)

    # Sort nearest first (smallest distance from robot origin)
    results.sort(key=lambda r: math.hypot(r['coords'][0], r['coords'][1]))

    # Print summary
    print(f"\n  [SCAN] {len(results)} box(es) found and classified:")
    for i, r in enumerate(results):
        x, y, _ = r['coords']
        dist = math.hypot(x, y)
        col  = LABEL_COLORS.get(r['label'], (120,120,120))
        print(f"    Box {i+1}: ({x:+.1f}, {y:+.1f}) cm  "
              f"dist={dist:.1f}cm  → {r['label']}  "
              f"({r['confidence']*100:.0f}% confidence)")

        # Annotate debug frame
        if debug_frame is not None:
            px, py = int(r['pixel'][0]), int(r['pixel'][1])
            cv2.putText(debug_frame,
                        r['label'],
                        (px - 30, py - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        col, 2)

    if debug_frame is not None:
        cv2.imshow("Scan + Classification", debug_frame)
        cv2.waitKey(800)

    return results

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Autonomous Package Sorting Robot                       ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║   Sticker colors:                                        ║")
    print("║     Red    = FRAGILE    Blue   = ZONE A                  ║")
    print("║     Yellow = ZONE B     Green  = IMMEDIATE               ║")
    print("║     Orange = DAMAGED                                      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║   Prompts:  sort fragile / sort zone a / sort zone b     ║")
    print("║             sort immediate / sort damaged / sort all     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Connect Arduino
    ser = connect_serial()

    # Move to home
    print("[INIT] Moving to home...")
    move_home(ser)

    # Start detector
    detector = BoxDetector()

    # Capture background
    print("\n[BG] Clear the board of ALL boxes.")
    print("     Click camera window → SPACE\n")
    while True:
        ret, frame = detector.cap.read()
        if not ret: continue
        disp = detector.undistort(frame)
        cv2.rectangle(disp, (0,0), (disp.shape[1], 50), (0,0,0), -1)
        cv2.putText(disp, "Clear board → SPACE to start  |  Q=quit",
                    (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,220,80), 2)
        cv2.imshow("Sorting Robot", disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q')):
            detector.close()
            if ser and ser.is_open: ser.close()
            return
        elif key == ord(' '):
            detector.capture_background()
            cv2.destroyAllWindows()
            print("[BG] Background captured!\n")
            break

    # ── Main prompt loop ──────────────────────────────────────
    cv2.destroyAllWindows()   # close camera window so terminal works
    print("Place boxes with stickers on the board.")
    print("Type your sorting command below.\n")

    while True:
        # Get prompt from terminal (no OpenCV window blocking)
        try:
            prompt = input("Command: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.lower() in ('q', 'quit', 'exit'):
            break

        if not prompt:
            continue

        target = parse_prompt(prompt)
        if target is None:
            print(f"  [?] Unrecognised command: '{prompt}'")
            print(f"      Try: sort fragile / sort zone a / sort zone b")
            print(f"           sort immediate / sort damaged / sort all\n")
            continue

        print(f"\n[CMD] Sorting: {target}")
        print("─" * 50)

        # ── Scan board ────────────────────────────────────────
        print("[SCAN] Scanning board...")
        results = scan_and_classify(detector)

        if not results:
            print("[WARN] No boxes found. Place boxes and try again.\n")
            continue

        # ── Filter by prompt ──────────────────────────────────
        if target == "ALL":
            targets = [r for r in results if r['label'] != "UNKNOWN"]
        else:
            targets = [r for r in results if r['label'] == target]

        if not targets:
            labels = [r['label'] for r in results]
            print(f"[WARN] No {target} boxes found.")
            print(f"       Detected labels: {labels}\n")
            continue

        print(f"\n[SORT] {len(targets)} {target} box(es) to sort "
              f"(nearest first):\n")
        for i, r in enumerate(targets):
            x, y, _ = r['coords']
            print(f"  {i+1}. ({x:+.1f}, {y:+.1f}) cm  "
                  f"dist={math.hypot(x,y):.1f} cm  → {r['label']}")
        print()

        # ── Pick and place each ───────────────────────────────
        picked = 0
        for i, box in enumerate(targets):
            x, y, _ = box['coords']
            label    = box['label']

            print(f"\n[PICK {i+1}/{len(targets)}] {label} at ({x:.1f},{y:.1f})")

            # Re-detect to get fresh coordinates before each pick
            if i > 0:
                print("  [RE-DETECT] Refreshing coordinates...")
                fresh = scan_and_classify(detector)
                # Find closest fresh detection to expected position
                if fresh:
                    best = min(fresh,
                               key=lambda r: math.hypot(
                                   r['coords'][0]-x,
                                   r['coords'][1]-y))
                    if math.hypot(best['coords'][0]-x,
                                  best['coords'][1]-y) < 5.0:
                        x, y, _ = best['coords']
                        print(f"  [RE-DETECT] Updated to ({x:.1f},{y:.1f})")

            success = pick_and_place_box(ser, x, y, label)
            if success:
                picked += 1
                if DASHBOARD_ACTIVE:
                    _dashboard_record(label, x, y)
            else:
                print(f"  [WARN] Pick failed — skipping this box.")
                move_home(ser)

            time.sleep(0.3)

        print(f"\n{'='*50}")
        print(f"[DONE] Sorted {picked}/{len(targets)} {target} boxes.")
        print(f"{'='*50}\n")
        print("Ready for next command.\n")

    # ── Cleanup ───────────────────────────────────────────────
    print("\n[END] Moving home and shutting down...")
    move_home(ser)
    detector.close()
    if ser and SERIAL_AVAILABLE and ser.is_open:
        ser.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
