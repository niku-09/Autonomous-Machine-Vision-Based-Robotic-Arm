"""
============================================================
 4-DOF Robot Arm — Inverse Kinematics + Serial Control
 Python side
============================================================

 HOW TO USE
 ----------
   python robot_arm_IK.py
   Enter a target (x, y, z) in cm.
   IK computes theta1, theta2, theta3.
   FK verifies the solution matches the target.
   Angles are sent to Arduino → robot moves smoothly.
   Plot saved to robot_arm_IK.png.

 INVERSE KINEMATICS — GEOMETRIC SOLUTION
 ----------------------------------------
 Your arm is a 3-DOF planar arm + base rotation.
 This gives a clean closed-form geometric solution.

 Step 1 — theta1 (base rotation):
   theta1 = atan2(y, x)
   This points the arm plane toward the target.

 Step 2 — Reduce to 2D planar problem:
   r  = sqrt(x^2 + y^2)   radial distance in XY plane
   z' = z - L1             height above shoulder

 Step 3 — theta3 (elbow angle, law of cosines):
   cos(theta3) = (r^2 + z'^2 - L2^2 - L3^2) / (2*L2*L3)
   theta3 = atan2(+sqrt(1 - cos^2), cos)  → elbow UP
   theta3 = atan2(-sqrt(1 - cos^2), cos)  → elbow DOWN

 Step 4 — theta2 (shoulder angle):
   k1 = L2 + L3*cos(theta3)
   k2 = L3*sin(theta3)
   theta2 = atan2(r, z') - atan2(k2, k1)

 Step 5 — Pick best solution:
   Both elbow-up and elbow-down are computed.
   The one whose joint angles are within safe limits is chosen.
   If both are valid, elbow-up is preferred.

 FK VERIFICATION:
   The computed angles are plugged back into FK.
   If the FK result matches the target within 0.5 cm, solution is accepted.

 WORKSPACE LIMITS
 ----------------
   Max reach : 29 cm forward, +-11.7 cm lateral (checkerboard workspace)
   Z grasp   : 1.5 cm min (mid-height of 3cm box)
   Z must be > 0 (below ground not reachable)

 REQUIRES:  pip install pyserial numpy matplotlib
============================================================
"""

import math
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    import serial
    import serial.tools.list_ports
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False
    print("[WARN] pyserial not installed. Running without serial.")

# ── Robot parameters (from FK) ────────────────────────────────
L1 = 13.5   # cm  ground → shoulder
L2 = 12.0   # cm  shoulder → elbow
L3 = 24.0   # cm  elbow → tip

# Safe kinematic angle limits (degrees)
# Derived from servo limits 5°–175° and horn offsets
LIMITS = {
    "theta1": (-55.0,  55.0),   # base rotation
    "theta2": (-50.0,  85.0),   # shoulder
    "theta3": ( 40.0, 148.0),   # elbow-up only
}

FK_TOLERANCE  = 0.5    # cm — acceptable FK verification error
BAUD_RATE     = 9600
SERIAL_PORT   = None   # None = auto-detect, or set e.g. "COM4"
PLOT_FILE     = "robot_arm_IK.png"


# ════════════════════════════════════════════════════════════
#  FORWARD KINEMATICS  (from FK phase — used for verification)
# ════════════════════════════════════════════════════════════

def forward_kinematics(t1_deg, t2_deg, t3_deg):
    """
    Planar FK + base rotation.
    Returns [p0, p1, p2, p3] joint positions and tip position.
    """
    t1 = math.radians(t1_deg)
    t2 = math.radians(t2_deg)
    t3 = math.radians(t3_deg)

    p0 = np.array([0.0, 0.0, 0.0])
    p1 = np.array([0.0, 0.0, L1])

    # Planar arm in XZ plane
    x_elbow = L2 * math.sin(t2)
    z_elbow = L1 + L2 * math.cos(t2)

    angle23 = t2 + t3
    x_tip   = x_elbow + L3 * math.sin(angle23)
    z_tip   = L1 + L2 * math.cos(t2) + L3 * math.cos(angle23)

    # Apply base rotation
    x2 =  x_elbow * math.cos(t1)
    y2 =  x_elbow * math.sin(t1)
    x3 =  x_tip   * math.cos(t1)
    y3 =  x_tip   * math.sin(t1)

    p2 = np.array([x2, y2, z_elbow])
    p3 = np.array([x3, y3, z_tip])

    return [p0, p1, p2, p3], p3


# ════════════════════════════════════════════════════════════
#  INVERSE KINEMATICS  (geometric closed-form)
# ════════════════════════════════════════════════════════════

def check_limits(t1, t2, t3):
    """Returns True if all angles are within safe kinematic limits."""
    return (LIMITS["theta1"][0] <= t1 <= LIMITS["theta1"][1] and
            LIMITS["theta2"][0] <= t2 <= LIMITS["theta2"][1] and
            LIMITS["theta3"][0] <= t3 <= LIMITS["theta3"][1])


def inverse_kinematics(x, y, z):
    """
    Geometric IK for 3-DOF planar arm + base rotation.

    Parameters
    ----------
    x, y, z : float  target position in cm (world frame)

    Returns
    -------
    (t1, t2, t3) : tuple of floats in degrees, or None if unreachable
    error_msg    : string describing failure reason, or None on success
    """

    # ── Workspace check ───────────────────────────────────────
    if z < 0:
        return None, "Target below ground (z < 0)"

    r  = math.sqrt(x**2 + y**2)     # radial distance
    zp = z - L1                      # height above shoulder joint

    dist = math.sqrt(r**2 + zp**2)  # distance from shoulder to target

    max_reach = L2 + L3
    min_reach = abs(L2 - L3)

    if dist > max_reach:
        return None, f"Target out of reach: distance from shoulder={dist:.2f} cm, max={max_reach:.2f} cm"
    if dist < min_reach:
        return None, f"Target too close: distance from shoulder={dist:.2f} cm, min={min_reach:.2f} cm"

    # ── Step 1: theta1 ────────────────────────────────────────
    t1_rad = math.atan2(y, x)
    t1_deg = math.degrees(t1_rad)

    # ── Step 2: theta3 (law of cosines) ──────────────────────
    cos_t3 = (r**2 + zp**2 - L2**2 - L3**2) / (2 * L2 * L3)
    cos_t3 = max(-1.0, min(1.0, cos_t3))   # clamp for floating point safety

    sin_t3_up   =  math.sqrt(1 - cos_t3**2)   # elbow UP
    sin_t3_down = -math.sqrt(1 - cos_t3**2)   # elbow DOWN

    t3_up   = math.degrees(math.atan2(sin_t3_up,   cos_t3))
    t3_down = math.degrees(math.atan2(sin_t3_down, cos_t3))

    # ── Step 3: theta2 for each t3 solution ───────────────────
    def compute_t2(t3_deg):
        t3 = math.radians(t3_deg)
        k1 = L2 + L3 * math.cos(t3)
        k2 = L3 * math.sin(t3)
        t2 = math.atan2(r, zp) - math.atan2(k2, k1)
        return math.degrees(t2)

    t2_up   = compute_t2(t3_up)
    t2_down = compute_t2(t3_down)

    # ── Step 4: pick valid solution ───────────────────────────
    # Prefer elbow-up; fall back to elbow-down
    solutions = [
        ("elbow-up", t1_deg, t2_up, t3_up),
        # elbow-down excluded — outside servo limits for this workspace
    ]

    for name, t1, t2, t3 in solutions:
        if not check_limits(t1, t2, t3):
            continue

        # FK verification
        _, fk_tip = forward_kinematics(t1, t2, t3)
        err = math.sqrt((fk_tip[0]-x)**2 + (fk_tip[1]-y)**2 + (fk_tip[2]-z)**2)
        if err <= FK_TOLERANCE:
            return (t1, t2, t3), None

    # Both solutions failed — report why
    msgs = []
    for name, t1, t2, t3 in solutions:
        if not check_limits(t1, t2, t3):
            msgs.append(f"{name}: angles ({t1:.1f},{t2:.1f},{t3:.1f}) exceed joint limits")
        else:
            _, fk_tip = forward_kinematics(t1, t2, t3)
            err = math.sqrt((fk_tip[0]-x)**2 + (fk_tip[1]-y)**2 + (fk_tip[2]-z)**2)
            msgs.append(f"{name}: FK error {err:.3f} cm exceeds tolerance")

    return None, " | ".join(msgs)


# ════════════════════════════════════════════════════════════
#  PLOT
# ════════════════════════════════════════════════════════════

def save_plot(positions, target, t1, t2, t3):
    p0, p1, p2, p3 = positions
    pts = np.array([p0, p1, p2, p3])

    fig = plt.figure(figsize=(7, 6))
    ax  = fig.add_subplot(111, projection='3d')

    # Arm links
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2],
            '-o', color='steelblue', linewidth=3,
            markersize=8, markerfacecolor='white', markeredgewidth=2,
            label='Arm')

    # Achieved tip
    ax.scatter(*p3, color='crimson', s=80, zorder=5, label=f'Tip ({p3[0]:.1f},{p3[1]:.1f},{p3[2]:.1f})')

    # Target point
    ax.scatter(*target, color='limegreen', s=100, marker='*', zorder=6,
               label=f'Target ({target[0]:.1f},{target[1]:.1f},{target[2]:.1f})')

    # Joint labels
    for pt, lbl in zip(pts, ['Origin', 'Shoulder', 'Elbow', 'Tip']):
        ax.text(pt[0]+0.5, pt[1]+0.5, pt[2]+0.5, lbl, fontsize=8)

    r = L1 + L2 + L3
    ax.set_xlim(-r, r)
    ax.set_ylim(-r, r)
    ax.set_zlim(0, r)
    ax.set_xlabel('X (cm)')
    ax.set_ylabel('Y (cm)')
    ax.set_zlabel('Z (cm)')
    ax.set_title(f'IK  t1={t1:.1f}  t2={t2:.1f}  t3={t3:.1f}')
    ax.legend(loc='upper left', fontsize=7)
    ax.view_init(elev=20, azim=-60)
    ax.set_box_aspect([1, 1, 0.8])

    fig.tight_layout()
    fig.savefig(PLOT_FILE, dpi=120)
    plt.close(fig)
    print(f"  [PLOT] Saved → {PLOT_FILE}")


# ════════════════════════════════════════════════════════════
#  SERIAL
# ════════════════════════════════════════════════════════════

def find_arduino_port():
    if not SERIAL_AVAILABLE:
        return None
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if any(kw in (p.description or "").lower()
               for kw in ["arduino", "ch340", "cp210", "usb serial", "usb-serial"]):
            return p.device
    return ports[0].device if ports else None


def connect_serial(port=None):
    if not SERIAL_AVAILABLE:
        return None
    target = port or find_arduino_port()
    if not target:
        print("  [WARN] No serial port found. Simulation mode.")
        return None
    try:
        ser = serial.Serial(target, BAUD_RATE, timeout=5)
        time.sleep(2)
        while ser.in_waiting:
            msg = ser.readline().decode('utf-8', errors='replace').strip()
            if msg:
                print(f"  [Arduino] {msg}")
        print(f"  [INFO] Connected on {target}")
        return ser
    except Exception as e:
        print(f"  [WARN] Serial failed ({e}). Simulation mode.")
        return None


def send_to_arduino(ser, t1, t2, t3):
    """Send kinematic angles. Gripper always closed."""
    cmd = f"{t1:.2f},{t2:.2f},{t3:.2f}\n"
    if ser is None:
        print(f"  [SIM] Would send: {cmd.strip()}")
        return True

    ser.write(cmd.encode())
    ser.flush()

    # Wait for confirmation (longer timeout since arm moves smoothly)
    resp = ser.readline().decode('utf-8', errors='replace').strip()
    if resp.startswith("OK:"):
        parts = resp[3:].split(',')
        print(f"  [Arduino] Reached → s1={parts[0]}°  s2={parts[1]}°  s3={parts[2]}°")
        return True
    elif resp.startswith("ERR:"):
        print(f"  [Arduino ERROR] {resp[4:]}")
        return False
    else:
        print(f"  [Arduino] {resp}")
        return False


# ════════════════════════════════════════════════════════════
#  INPUT HELPERS
# ════════════════════════════════════════════════════════════

def get_float(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("    Enter a number.")


# ════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════

def main():
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   4-DOF Robot Arm — Inverse Kinematics  v1           ║")
    print("║   L1=13.5  L2=12  L3=24  (cm)                       ║")
    print("║   Enter target (x,y,z) → robot moves there          ║")
    print("║   Start pose: theta1=0, theta2=0, theta3=90          ║")
    print("║   (upper arm horizontal — safe resting position)     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    ser = connect_serial(SERIAL_PORT)

    while True:
        print("-- Enter target coordinate (cm) ------------------")
        try:
            x = get_float("  x: ")
            y = get_float("  y: ")
            z = get_float("  z: ")
        except (EOFError, KeyboardInterrupt):
            break

        print(f"\n  Target: ({x:.2f}, {y:.2f}, {z:.2f})")

        # Solve IK
        result, err_msg = inverse_kinematics(x, y, z)

        if result is None:
            print(f"  [IK FAILED] {err_msg}")
            print()
            cont = input("  Try again? [y/n]: ").strip().lower()
            if cont != 'y':
                break
            continue

        t1, t2, t3 = result

        # FK verification
        positions, fk_tip = forward_kinematics(t1, t2, t3)
        fk_err = math.sqrt((fk_tip[0]-x)**2 + (fk_tip[1]-y)**2 + (fk_tip[2]-z)**2)

        print(f"\n  IK Solution:")
        print(f"    theta1 = {t1:8.3f} deg  (base)")
        print(f"    theta2 = {t2:8.3f} deg  (forearm)")
        print(f"    theta3 = {t3:8.3f} deg  (upper arm)")
        print(f"\n  FK Verification:")
        print(f"    Target  : ({x:.3f}, {y:.3f}, {z:.3f})")
        print(f"    Achieved: ({fk_tip[0]:.3f}, {fk_tip[1]:.3f}, {fk_tip[2]:.3f})")
        print(f"    Error   : {fk_err:.4f} cm  ({'PASS' if fk_err <= FK_TOLERANCE else 'FAIL'})")

        # Save plot
        save_plot(positions, (x, y, z), t1, t2, t3)

        # Send to Arduino
        print(f"\n  Sending to robot...")
        success = send_to_arduino(ser, t1, t2, t3)
        if success:
            print(f"  [DONE] Robot moving to ({x:.1f}, {y:.1f}, {z:.1f})")

        print()
        try:
            cont = input("  Move to another coordinate? [y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cont != 'y':
            break
        print()

    if ser and SERIAL_AVAILABLE and ser.is_open:
        ser.close()
        print("  [INFO] Serial port closed.")
    print("  Done.")


if __name__ == "__main__":
    main()
