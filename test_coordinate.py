"""
============================================================
 Test: Detect box → pick up → return to home
============================================================
 SEQUENCE (press S):
   1. Arm goes to HOME (out of camera view)
   2. Camera detects box → gets (x, y)
   3. Move to (x, y, APPROACH_Z)  — fast
   4. Lower to (x, y, GRASP_Z)    — slow
   5. Close gripper                — slow
   6. Nudge up 1cm                 — slow (break friction)
   7. Lift to (x, y, APPROACH_Z)  — slow
   8. Return to HOME (0,0,0 pose)  — slow

 CONTROLS:
   SPACE = capture background
   S     = detect + full pick sequence
   H     = return home
   Q     = return home + quit

 CONFIGURE:
   SERIAL_PORT = "COM4"
   GRASP_Z     = 1.5 cm
============================================================
"""

import sys
import time
import cv2

sys.path.append(".")
from detect_boxes import BoxDetector
from robot_arm_IK  import inverse_kinematics

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ─────────────────────────────────────────────────────────────
SERIAL_PORT   = "COM4"    # ← your Arduino COM port
BAUD_RATE     = 9600
APPROACH_Z    = 13.0    # cm — safe travel height (fast)
GRASP_Z       =  1.5    # cm — grasp height (slow lower)
GRIPPER_OPEN  = 90.0    # degrees
GRIPPER_CLOSE = 70.0    # degrees — confirmed for 3cm cardboard box
PAUSE_GRIP    =  0.8    # seconds after grip/release
# ─────────────────────────────────────────────────────────────

def connect_serial():
    if not SERIAL_AVAILABLE:
        print("[ERROR] pyserial not installed. Run: pip install pyserial")
        return None

    # List all available ports
    ports = list(serial.tools.list_ports.comports())
    print(f"[SERIAL] Available ports: {[p.device for p in ports]}")

    if not ports:
        print("[ERROR] No COM ports found. Is Arduino plugged in?")
        return None

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=30)
        time.sleep(2)
        while ser.in_waiting:
            msg = ser.readline().decode('utf-8', errors='replace').strip()
            if msg: print(f"  [Arduino] {msg}")
        print(f"[SERIAL] Connected on {SERIAL_PORT} ✓")
        return ser
    except Exception as e:
        print(f"[ERROR] Could not open {SERIAL_PORT}: {e}")
        print(f"        Available ports: {[p.device for p in ports]}")
        print(f"        Change SERIAL_PORT in the script to the correct port.")
        return None


def send_angles(ser, t1, t2, t3, gripper=90.0, slow=False):
    s_flag = 1 if slow else 0
    cmd = f"{t1:.2f},{t2:.2f},{t3:.2f},{gripper:.1f},{s_flag}\n"
    if ser is None:
        print(f"  [SIM] {cmd.strip()}")
        return True
    ser.write(cmd.encode())
    ser.flush()
    resp = ser.readline().decode('utf-8', errors='replace').strip()
    if resp.startswith("OK:"):
        print(f"  [OK] → {resp[3:]}")
        return True
    else:
        print(f"  [ERR] {resp}")
        return False


def move_to(ser, x, y, z, gripper=90.0, label="", slow=False):
    result, err = inverse_kinematics(x, y, z)
    if result is None:
        print(f"  [IK FAIL] {label}: {err}")
        return False
    t1, t2, t3 = result
    speed_tag = " [SLOW]" if slow else ""
    print(f"  [{label}{speed_tag}] ({x:.1f},{y:.1f},{z:.1f})cm "
          f"θ1={t1:.1f}° θ2={t2:.1f}° θ3={t3:.1f}°")
    return send_angles(ser, t1, t2, t3, gripper=gripper, slow=slow)


def main():
    print()
    print("╔══════════════════════════════════════════════╗")
    print("║   Robot Coordinate Test                      ║")
    print(f"║   Test Z height: {GRASP_Z} cm                    ║")
    print("╠══════════════════════════════════════════════╣")
    print("║   SPACE = capture background                 ║")
    print("║   S     = detect + pick + return home        ║")
    print("║   H     = open gripper + return home         ║")
    print("║   Q     = return home + quit                 ║")
    print("╚══════════════════════════════════════════════╝")
    print()

    ser = connect_serial()

    # Move to home first — two step to avoid mechanical conflict
    print("[INIT] Moving to home...")
    send_angles(ser, 0.0, 0.0, 90.0, gripper=GRIPPER_OPEN, slow=True)
    time.sleep(1.5)

    detector = BoxDetector()

    # Capture background
    print("\n[BG] Clear the board, click window, press SPACE\n")
    while True:
        ret, frame = detector.cap.read()
        if not ret: continue
        disp = detector.undistort(frame)
        cv2.rectangle(disp, (0,0), (disp.shape[1], 50), (0,0,0), -1)
        cv2.putText(disp, "Clear board → SPACE  |  Q=quit",
                    (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,220,80), 2)
        cv2.imshow("Test", disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q')):
            print("[HOME] Returning home before quit...")
            send_angles(ser, 0.0, 0.0, 90.0, slow=True)
            time.sleep(2)
            detector.close(); return
        elif key == ord(' '):
            detector.capture_background()
            print("[BG] Background captured! Now place a box.\n")
            break

    # Detection + move loop
    while True:
        ret, frame = detector.cap.read()
        if not ret: continue
        disp = detector.undistort(frame)
        cv2.putText(disp, "S=detect+move  H=home  Q=quit",
                    (12, disp.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,80), 2)
        cv2.imshow("Test", disp)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q')):
            print("[HOME] Returning home + opening gripper before quit...")
            send_angles(ser, 0.0, 0.0, 90.0,
                        gripper=GRIPPER_OPEN, slow=True)
            time.sleep(2)
            break

        elif key in (ord('h'), ord('H')):
            print("[HOME] Returning home + opening gripper...")
            send_angles(ser, 0.0, 0.0, 90.0,
                        gripper=GRIPPER_OPEN, slow=True)

        elif key in (ord('s'), ord('S')):
            print("\n[DETECT] Scanning...")

            if ser is None:
                print("[ERROR] No serial connection! Robot cannot move.")
                print(f"        Check SERIAL_PORT = '{SERIAL_PORT}' is correct.")
                print("        Plug in Arduino and restart the program.")
                continue

            # Step 1: Home — lift first, then twist out of camera view
            print("  Step 1: Lift arm to safe position...")
            send_angles(ser, 0.0, 0.0, 90.0, gripper=GRIPPER_OPEN, slow=True)
            time.sleep(1.5)

            # Step 2: Detect box
            boxes, debug = detector.detect(visualise=True)
            if debug is not None:
                cv2.imshow("Detection", debug)

            if not boxes:
                print("[WARN] No boxes found. Place a box and try again.")
                continue

            x, y, _ = boxes[0]['coords']
            print(f"\n  Box detected at ({x:.2f}, {y:.2f}) cm")

            # Step 3: Approach above box (fast, gripper open)
            print(f"  Step 3: Approach above box...")
            move_to(ser, x, y, APPROACH_Z,
                    gripper=GRIPPER_OPEN, label="Approach", slow=False)

            # Step 4: Descend slowly to grasp height
            print(f"  Step 4: Descend to grasp height ({GRASP_Z} cm)...")
            move_to(ser, x, y, GRASP_Z,
                    gripper=GRIPPER_OPEN, label="Descend", slow=True)

            # Step 5: Close gripper
            print("  Step 5: Close gripper...")
            result, _ = inverse_kinematics(x, y, GRASP_Z)
            if result:
                send_angles(ser, result[0], result[1], result[2],
                            gripper=GRIPPER_CLOSE, slow=True)
            time.sleep(PAUSE_GRIP)

            # Step 6: Nudge up 1cm to break static friction
            print("  Step 6: Nudge up (break static friction)...")
            result_nudge, _ = inverse_kinematics(x, y, GRASP_Z + 1.0)
            if result_nudge:
                send_angles(ser, result_nudge[0], result_nudge[1], result_nudge[2],
                            gripper=GRIPPER_CLOSE, slow=True)
            time.sleep(0.3)

            # Step 7: Lift to approach height (slow, gripper closed)
            print("  Step 7: Lift to approach height...")
            move_to(ser, x, y, APPROACH_Z,
                    gripper=GRIPPER_CLOSE, label="Lift", slow=True)

            # Step 8: Return to home — lift first then twist
            print("  Step 8: Lift to safe position...")
            result_lift, _ = inverse_kinematics(x, y, APPROACH_Z)
            if result_lift:
                send_angles(ser, result_lift[0], 0.0, 90.0,
                            gripper=GRIPPER_CLOSE, slow=True)
                time.sleep(1.0)
            print("  Step 8b: Return to HOME pose...")
            send_angles(ser, 0.0, 0.0, 90.0,
                        gripper=GRIPPER_CLOSE, slow=True)
            time.sleep(1.5)

            print("\n  ✓ Pick complete! Box should be in gripper at home position.")
            print("    Press H to open gripper and release.")
            print("    Press S to pick another box.")
            print("    Press Q to quit.\n")

    # Cleanup
    print("[END] Moving home and quitting...")
    send_angles(ser, 0.0, 0.0, 90.0)
    if ser and ser.is_open:
        ser.close()
    detector.close()


if __name__ == "__main__":
    main()
