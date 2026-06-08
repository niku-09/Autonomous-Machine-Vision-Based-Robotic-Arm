"""
============================================================
 Pick and Place — Complete System
============================================================

 SEQUENCE (per box):
   1. Arm moves to HOME (out of camera view)
   2. Camera detects box → gets (x, y, z) in robot cm
   3. Move to APPROACH (above box, gripper open)
   4. Descend to GRASP position
   5. Close gripper → pick up box
   6. Lift back to approach height
   7. Move to DROP ZONE (to the right)
   8. Descend to place height
   9. Open gripper → release box
  10. Lift up
  11. Return to HOME
  12. Repeat for next box

 CONFIGURE BELOW:
   HOME_POS     — safe position out of camera view
   DROP_ZONE    — where to place the box (x, y, z)
   APPROACH_Z   — height above box before descending
   SERIAL_PORT  — your Arduino COM port

 REQUIRES:
   detect_boxes.py   (box detection)
   robot_arm_IK.py   (inverse kinematics)
   calibration.npz   (camera calibration)
   robot_arm_pick_place.ino uploaded to Arduino
============================================================
"""

import math
import time
import sys

# ── Import your existing modules ──────────────────────────────
sys.path.append(".")
from detect_boxes import BoxDetector
from robot_arm_IK  import inverse_kinematics, forward_kinematics

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed. Running in simulation mode.")

# ─────────────────────────────────────────────────────────────
#  CONFIGURE THESE
# ─────────────────────────────────────────────────────────────

# Serial port — set to your Arduino port e.g. "COM3"
SERIAL_PORT   = None   # None = auto-detect
BAUD_RATE     = 9600

# Gripper angles
GRIPPER_OPEN   = 90.0   # degrees — open
GRIPPER_CLOSED = 70.0   # degrees — confirmed for 3cm cardboard box

# Home position — arm points UP, out of camera view
# theta1=0, theta2=0, theta3=90 is your safe resting position
HOME_POS = (0.0, 0.0, 13.5 + 24.0)   # x=0, y=0, z = L1+L3 (arm up)

# Approach height — how high above the box to approach before descending (cm)
APPROACH_Z = 13.0

# Drop zone — where to place the box
# "Move to the right" = negative Y in robot frame
# Adjust X and Y based on where you want to drop
DROP_ZONE_X =  0.0    # cm — forward position of drop zone
DROP_ZONE_Y = -35.0   # cm — to the RIGHT of robot (negative Y)
DROP_ZONE_Z =  3.0    # cm — place height

# Grasp Z — height of gripper when grabbing (half box height)
GRASP_Z = 3.0   # cm

# ── Grasp offset corrections (trial and error) ───────────────
# After IK positions the arm above the box, these fine-tune
# the final grasp approach to correct for systematic error.
#
# THETA1_OFFSET: twists the base slightly (+ = left, - = right)
# THETA3_OFFSET: tilts the forearm slightly (+ = down, - = up)
#
# Start at 5.0 and adjust based on where the gripper misses.
# If gripper misses LEFT  → decrease THETA1_OFFSET (more negative)
# If gripper misses RIGHT → increase THETA1_OFFSET (more positive)
# If gripper is too HIGH  → increase THETA3_OFFSET
# If gripper is too LOW   → decrease THETA3_OFFSET
THETA1_OFFSET = 5.0   # degrees — base twist correction
THETA3_OFFSET = 5.0   # degrees — forearm tilt correction

# Pause times (seconds)
PAUSE_AFTER_GRIP   = 0.5
PAUSE_AFTER_MOVE   = 0.3

# ─────────────────────────────────────────────────────────────
#  SERIAL
# ─────────────────────────────────────────────────────────────

def find_arduino_port():
    if not SERIAL_AVAILABLE:
        return None
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(kw in (p.description or "").lower()
               for kw in ["arduino", "ch340", "cp210", "usb serial", "usb-serial"]):
            return p.device
    return ports[0].device if ports else None


def connect_serial():
    if not SERIAL_AVAILABLE:
        return None
    port = SERIAL_PORT or find_arduino_port()
    if not port:
        print("[WARN] No serial port found — simulation mode.")
        return None
    try:
        ser = serial.Serial(port, BAUD_RATE, timeout=10)
        time.sleep(2)
        while ser.in_waiting:
            msg = ser.readline().decode('utf-8', errors='replace').strip()
            if msg:
                print(f"  [Arduino] {msg}")
        print(f"  [INFO] Connected on {port}")
        return ser
    except Exception as e:
        print(f"  [WARN] Serial failed ({e}) — simulation mode.")
        return None


def send_command(ser, t1, t2, t3, gripper, slow=False):
    """Send T1,T2,T3,G,S to Arduino and wait for OK.
       slow=True → gentle descent speed (0.5 deg/step)
       slow=False → normal speed (1.5 deg/step)
    """
    s_flag = 1 if slow else 0
    cmd = f"{t1:.2f},{t2:.2f},{t3:.2f},{gripper:.1f},{s_flag}\n"
    if ser is None:
        print(f"    [SIM] → {cmd.strip()}")
        return True
    ser.write(cmd.encode())
    ser.flush()
    resp = ser.readline().decode('utf-8', errors='replace').strip()
    if resp.startswith("OK:"):
        return True
    elif resp.startswith("ERR:"):
        print(f"    [ERR] Arduino: {resp[4:]}")
        return False
    else:
        print(f"    [Arduino] {resp}")
        return False

# ─────────────────────────────────────────────────────────────
#  MOVE HELPERS
# ─────────────────────────────────────────────────────────────

def move_to(ser, x, y, z, gripper, label="", slow=False):
    """Solve IK and send to robot.
       slow=True → gentle speed for descent/ascent
    """
    result, err = inverse_kinematics(x, y, z)
    if result is None:
        print(f"    [IK FAIL] {label}: {err}")
        return False
    t1, t2, t3 = result
    speed_tag = " [SLOW]" if slow else ""
    print(f"    [{label}{speed_tag}] → ({x:.1f}, {y:.1f}, {z:.1f}) cm  "
          f"| θ1={t1:.1f}° θ2={t2:.1f}° θ3={t3:.1f}°  "
          f"| gripper={'OPEN' if gripper > 50 else 'CLOSED'}")
    ok = send_command(ser, t1, t2, t3, gripper, slow=slow)
    time.sleep(PAUSE_AFTER_MOVE)
    return ok


def move_home(ser):
    """
    Two-step home to prevent servo fighting:
      Step A: Lift arm straight up first (forearm/upper to safe angles)
      Step B: Twist base out of camera view
    """
    print("  → HOME step A: lift arm straight up...")
    send_command(ser, 0.0, 0.0, 90.0, GRIPPER_OPEN, slow=True)
    time.sleep(1.0)
    print("  → HOME step B: twist out of camera view...")
    send_command(ser, -30.0, 0.0, 90.0, GRIPPER_OPEN, slow=False)
    time.sleep(0.5)
    return True

# ─────────────────────────────────────────────────────────────
#  PICK AND PLACE SEQUENCE
# ─────────────────────────────────────────────────────────────

def pick_and_place(ser, x, y, z):
    """
    Full pick-and-place sequence with offset correction.

    SEQUENCE:
      1. Move above box (APPROACH_Z), gripper open
      2. Descend to GRASP_Z via IK → get base angles
      3. Apply THETA1_OFFSET → twist base to correct lateral miss
      4. Apply THETA3_OFFSET → tilt forearm to correct height miss
      5. Close gripper
      6. Lift, move to drop zone, place, return home
    """
    print(f"\n  ── Pick and place: box at ({x:.1f}, {y:.1f}, {z:.1f}) cm ──")
    print(f"     Offsets: θ1={THETA1_OFFSET:+.1f}°  θ3={THETA3_OFFSET:+.1f}°")

    # 1. Approach — above box, gripper open
    print("  Step 1: Approach (above box)")
    ok = move_to(ser, x, y, APPROACH_Z, GRIPPER_OPEN, "Approach")
    if not ok:
        return False

    # 2. Descend to grasp height — get raw IK angles
    print("  Step 2: Descend to grasp height")
    result, err = inverse_kinematics(x, y, GRASP_Z)
    if result is None:
        print(f"    [IK FAIL] Descend: {err}")
        return False
    t1_base, t2_base, t3_base = result
    print(f"    [Descend] θ1={t1_base:.1f}° θ2={t2_base:.1f}° θ3={t3_base:.1f}°")
    ok = send_command(ser, t1_base, t2_base, t3_base, GRIPPER_OPEN, slow=True)
    time.sleep(PAUSE_AFTER_MOVE)
    if not ok:
        return False

    # 3. Apply theta1 offset — twist base to correct lateral miss
    # THETA1_OFFSET: + = left, - = right
    print(f"  Step 3: Base twist correction ({THETA1_OFFSET:+.1f}°)")
    t1_corrected = t1_base + THETA1_OFFSET
    print(f"    [Base correct] θ1={t1_corrected:.1f}°")
    ok = send_command(ser, t1_corrected, t2_base, t3_base, GRIPPER_OPEN, slow=True)
    time.sleep(PAUSE_AFTER_MOVE)
    if not ok:
        return False

    # 4. Apply theta3 offset — tilt forearm down to correct height
    # THETA3_OFFSET: + = down, - = up
    print(f"  Step 4: Forearm correction ({THETA3_OFFSET:+.1f}°)")
    t3_corrected = t3_base + THETA3_OFFSET
    print(f"    [Forearm correct] θ3={t3_corrected:.1f}°")
    ok = send_command(ser, t1_corrected, t2_base, t3_corrected, GRIPPER_OPEN, slow=True)
    time.sleep(PAUSE_AFTER_MOVE)
    if not ok:
        return False

    # 5. Close gripper — grab box
    print("  Step 5: Close gripper (grab)")
    send_command(ser, t1_corrected, t2_base, t3_corrected, GRIPPER_CLOSED, slow=True)
    time.sleep(PAUSE_AFTER_GRIP)

    # 6. Lift up — nudge first to break static friction
    print("  Step 6: Lift")
    result_nudge, _ = inverse_kinematics(x, y, GRASP_Z + 1.0)
    if result_nudge:
        print("    [nudge] Breaking static friction...")
        send_command(ser, result_nudge[0], result_nudge[1], result_nudge[2],
                     GRIPPER_CLOSED, slow=True)
        time.sleep(0.3)
    ok = move_to(ser, x, y, APPROACH_Z, GRIPPER_CLOSED, "Lift", slow=True)
    if not ok:
        return False

    # 7. Move to drop zone
    print("  Step 7: Move to drop zone")
    ok = move_to(ser, DROP_ZONE_X, DROP_ZONE_Y, APPROACH_Z,
                 GRIPPER_CLOSED, "Drop approach")
    if not ok:
        return False

    # 8. Descend to place height
    print("  Step 8: Descend to place")
    ok = move_to(ser, DROP_ZONE_X, DROP_ZONE_Y, DROP_ZONE_Z,
                 GRIPPER_CLOSED, "Place descend")
    if not ok:
        return False

    # 9. Open gripper — release box
    print("  Step 9: Open gripper (release)")
    result2, _ = inverse_kinematics(DROP_ZONE_X, DROP_ZONE_Y, DROP_ZONE_Z)
    if result2:
        send_command(ser, *result2, GRIPPER_OPEN)
    time.sleep(PAUSE_AFTER_GRIP)

    # 10. Lift from drop zone
    print("  Step 10: Lift from drop zone")
    move_to(ser, DROP_ZONE_X, DROP_ZONE_Y, APPROACH_Z, GRIPPER_OPEN, "Lift drop")

    # 11. Return home
    print("  Step 11: Return to HOME")
    move_home(ser)

    print("  ✓ Pick and place complete!")
    return True


def _last_angles(ser, x, y, z):
    """Helper: get IK angles for a position (used for gripper-only commands)."""
    result, _ = inverse_kinematics(x, y, z)
    if result:
        return result
    return (0.0, 0.0, 90.0)

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Autonomous Pick and Place — Package Sorting Robot      ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║   1. Clear board → SPACE to capture background           ║")
    print("║   2. Place box(es) on board                              ║")
    print("║   3. Press S → detect + pick + place automatically       ║")
    print("║   4. Press Q to quit                                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()
    print(f"  Drop zone: ({DROP_ZONE_X}, {DROP_ZONE_Y}, {DROP_ZONE_Z}) cm")
    print(f"  Approach height: {APPROACH_Z} cm")
    print()

    # Connect to Arduino
    ser = connect_serial()

    # Start at home
    print("[INIT] Moving to home position...")
    move_home(ser)

    # Start detector
    import cv2
    detector = BoxDetector()

    # Capture background
    print("\n[BG] Remove ALL boxes from the board.")
    print("     Click camera window → press SPACE\n")

    while True:
        ret, frame = detector.cap.read()
        if not ret:
            continue
        disp = detector.undistort(frame)
        cv2.rectangle(disp, (0,0), (disp.shape[1], 50), (0,0,0), -1)
        cv2.putText(disp, "Clear board → SPACE to start  |  Q=quit",
                    (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 220, 80), 2)
        cv2.imshow("Pick and Place", disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q')):
            detector.close()
            return
        elif key == ord(' '):
            detector.capture_background()
            break

    print("\n[READY] Place box(es) on the board.")
    print("        Click window → S = detect and pick   Q = quit\n")

    while True:
        ret, frame = detector.cap.read()
        if not ret:
            continue
        disp = detector.undistort(frame)
        cv2.putText(disp, "S=detect+pick  B=new background  Q=quit",
                    (12, disp.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 80), 2)
        cv2.imshow("Pick and Place", disp)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        elif key in (ord('b'), ord('B')):
            print("[BG] Re-capturing background — remove all boxes first!")
            detector.capture_background()

        elif key in (ord('s'), ord('S')):
            print("\n[DETECT] Scanning for boxes...")

            # Move arm to home first (out of camera view)
            move_home(ser)
            time.sleep(1)   # wait for arm to settle

            # Detect
            boxes, debug = detector.detect(visualise=True)

            if not boxes:
                print("[WARN] No boxes detected. Try again.")
                if debug is not None:
                    cv2.imshow("Detection", debug)
                continue

            print(f"[DETECT] {len(boxes)} box(es) found:")
            for i, b in enumerate(boxes):
                x, y, z = b['coords']
                print(f"  Box {i+1}: ({x:+.2f}, {y:+.2f}, {z:.2f}) cm")

            if debug is not None:
                cv2.imshow("Detection", debug)
            cv2.waitKey(500)

            # Pick and place each box
            for i, box in enumerate(boxes):
                x, y, z = box['coords']
                z = GRASP_Z   # always use fixed grasp height

                print(f"\n[PICK] Box {i+1} of {len(boxes)}")
                success = pick_and_place(ser, x, y, z)

                if not success:
                    print(f"[WARN] Box {i+1} failed — skipping.")
                    move_home(ser)

                # Small pause between boxes
                time.sleep(0.5)

            print("\n[DONE] All boxes processed. Ready for next batch.")

    # Cleanup
    move_home(ser)
    if ser and ser.is_open:
        ser.close()
    detector.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
