"""
============================================================
 Robot Arm Vision — Step 1: Camera Calibration
 (Inspired by Kai Nakamura's Camera.m — adapted for Python/OpenCV)
============================================================

 SETUP (confirmed):
   Board  : 8×6 inner corners, 2.9 cm squares, glued flat
   Camera : USB desktop camera (OKER, on tall stand)

 TWO-PHASE PROCESS
 -----------------
 PHASE 1 — Intrinsics  (move board around ~20 captures)
   Computes K matrix and distortion coefficients.

 PHASE 2 — Extrinsics  (board in FIXED final position, one shot)
   *** Directly adapted from Kai's calculateCameraPos() ***
   Detects board corners automatically from a single undistorted image.
   Uses solvePnP (≡ Kai's extrinsics()) to compute R and t.
   This gives camera height automatically — no ruler needed for that.
   Only one ruler measurement needed: robot base → board origin.

 WHAT GETS SAVED  →  calibration.npz
   K, new_K, dist          ← intrinsics + undistortion matrix
   cam_R, cam_t            ← camera extrinsics (for projection correction)
   H                       ← pixel → checkerboard cm  (homography)
   T_checker_to_robot      ← 4×4 board frame → robot frame

 CONTROLS (Phase 1)
   SPACE → capture  |  D → undo  |  C → proceed  |  Q → quit

 REQUIRES:  pip install opencv-python numpy
============================================================
"""

import cv2
import numpy as np
import sys

# ─────────────────────────────────────────────────────────────
#  BOARD PARAMETERS  (your confirmed setup)
# ─────────────────────────────────────────────────────────────
BOARD_COLS     = 8      # 8 inner corners along long edge  (7 gaps × 2.9 cm = 20.3 cm)
BOARD_ROWS     = 6      # 6 inner corners along short edge (5 gaps × 2.9 cm = 14.5 cm)
SQUARE_SIZE_CM = 2.9    # measured — each square is 2.9 cm
BOARD_SIZE     = (BOARD_COLS, BOARD_ROWS)
OUTPUT_FILE    = "calibration.npz"
CAMERA_INDEX   = 0
MIN_CAPTURES   = 20

# 3-D world points (z=0, board plane), origin = top-left inner corner
objp = np.zeros((BOARD_COLS * BOARD_ROWS, 3), np.float32)
objp[:, :2] = np.mgrid[0:BOARD_COLS, 0:BOARD_ROWS].T.reshape(-1, 2)
objp        *= SQUARE_SIZE_CM

CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# ─────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────
def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {CAMERA_INDEX}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera opened — {w}×{h}")
    return cap


def detect_corners(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, BOARD_SIZE,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_NORMALIZE_IMAGE +
        cv2.CALIB_CB_FAST_CHECK)
    if found:
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), CRITERIA)
        return corners, gray
    return None, gray


def draw_hud(frame, text, color, n=None, total=None):
    h = frame.shape[0]
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(frame, text, (12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
    cv2.putText(frame, "SPACE=capture  D=undo  C=next phase  Q=quit",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (170, 170, 170), 1)
    if n is not None and total is not None:
        bw = int((frame.shape[1] - 24) * min(n, total) / total)
        cv2.rectangle(frame, (12, h-30), (12+bw, h-20), (30, 200, 30), -1)
        cv2.rectangle(frame, (12, h-30), (frame.shape[1]-12, h-20), (80, 80, 80), 1)

# ─────────────────────────────────────────────────────────────
#  PHASE 1 — INTRINSICS
# ─────────────────────────────────────────────────────────────
def phase1_intrinsics(cap):
    print()
    print("─" * 58)
    print("  PHASE 1 — Intrinsics")
    print("  Move board to ~20 varied positions/tilts/distances.")
    print("  Green overlay = board detected. SPACE to capture.")
    print("─" * 58)

    obj_pts, img_pts = [], []
    img_shape = None

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        corners, gray = detect_corners(frame)
        n = len(obj_pts)
        disp = frame.copy()

        if corners is not None:
            cv2.drawChessboardCorners(disp, BOARD_SIZE, corners, True)
            draw_hud(disp, f"Board FOUND — {n}/{MIN_CAPTURES} saved",
                     (30, 200, 30), n, MIN_CAPTURES)
        else:
            draw_hud(disp, f"Board NOT FOUND — {n}/{MIN_CAPTURES} saved",
                     (30, 30, 220), n, MIN_CAPTURES)

        cv2.imshow("Phase 1 — Intrinsics", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        elif key == ord(' '):
            if corners is None:
                print("[SKIP] Board not detected.")
            else:
                obj_pts.append(objp.copy())
                img_pts.append(corners)
                img_shape = gray.shape[::-1]
                print(f"[CAP]  {n+1:2d}")
        elif key == ord('d'):
            if obj_pts:
                obj_pts.pop(); img_pts.pop()
                print(f"[DEL]  {len(obj_pts)} remain")
        elif key == ord('c'):
            if len(obj_pts) < 6:
                print("[WARN] Need at least 6 captures.")
            else:
                cv2.destroyAllWindows()
                break

    print(f"\n[CAL] Running calibrateCamera ({len(obj_pts)} images)...")
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_pts, img_pts, img_shape, None, None)
    print(f"[CAL] RMS = {rms:.4f} px  (target < 1.0)")
    print(f"[CAL] K:\n{np.round(K, 2)}")

    if rms > 2.0:
        print("[WARN] RMS is high — try more varied angles next time.")

    return K, dist, rms, img_shape

# ─────────────────────────────────────────────────────────────
#  PHASE 2 — EXTRINSICS  (Kai's calculateCameraPos approach)
# ─────────────────────────────────────────────────────────────
def phase2_extrinsics(cap, K, dist):
    """
    Directly mirrors Kai's calculateCameraPos():
      1. Capture undistorted image of board in FINAL position
      2. detectCheckerboardPoints  →  cv2.findChessboardCorners
      3. extrinsics(imagePoints, worldPoints, camIntrinsics)  →  solvePnP
      4. Build homography H = new_K @ [r1 r2 t]
      5. Get T_checker→robot from ruler measurement
    """
    print()
    print("─" * 58)
    print("  PHASE 2 — Extrinsics (Kai's calculateCameraPos method)")
    print("  Place the board in its FINAL working position.")
    print("  Don't move it again after this point.")
    print("  SPACE to capture when the board is fully visible.")
    print("─" * 58)

    ret, probe = cap.read()
    h_f, w_f   = probe.shape[:2]
    new_K, _   = cv2.getOptimalNewCameraMatrix(K, dist, (w_f, h_f), alpha=0)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Undistort first (Kai does this before corner detection)
        undist  = cv2.undistort(frame, K, dist, None, new_K)
        corners, _ = detect_corners(undist)
        disp = undist.copy()

        if corners is not None:
            cv2.drawChessboardCorners(disp, BOARD_SIZE, corners, True)
            draw_hud(disp, "Board FOUND — SPACE to lock this as working position",
                     (30, 200, 30))
        else:
            draw_hud(disp, "Board NOT FOUND — adjust until all corners visible",
                     (30, 30, 220))

        cv2.imshow("Phase 2 — Extrinsics (final board position)", disp)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            cap.release(); cv2.destroyAllWindows(); sys.exit(0)
        elif key == ord(' '):
            if corners is None:
                print("[SKIP] Board not visible — try adjusting.")
                continue

            cv2.destroyAllWindows()

            # ── solvePnP  ≡  Kai's extrinsics() ─────────────
            # Image is already undistorted → pass dist=None
            ok, rvec, tvec = cv2.solvePnP(
                objp, corners, new_K, None,
                flags=cv2.SOLVEPNP_ITERATIVE)

            if not ok:
                print("[ERROR] solvePnP failed. Press SPACE again.")
                continue

            cam_R, _ = cv2.Rodrigues(rvec)
            cam_t    = tvec.ravel()   # [tx, ty, tz] in cm

            # Reprojection check
            proj, _ = cv2.projectPoints(objp, rvec, tvec, new_K, None)
            err = np.mean(np.linalg.norm(
                corners.reshape(-1, 2) - proj.reshape(-1, 2), axis=1))

            print(f"\n[EXT] Camera extrinsics:")
            print(f"      cam_R:\n{np.round(cam_R, 4)}")
            print(f"      cam_t   : {np.round(cam_t, 3)} cm")
            print(f"      Height  : {cam_t[2]:.2f} cm  (auto-computed)")
            print(f"      Reproj  : {err:.3f} px")

            if err > 3.0:
                print("[WARN] Reprojection error high. Phase 1 may need more images.")

            # ── Homography: pixel → board cm ─────────────────
            # H = new_K @ [r1 | r2 | t]
            H = new_K @ np.column_stack([cam_R[:, 0], cam_R[:, 1], cam_t])
            H = H / H[2, 2]
            print(f"\n[HOM] H (pixel→board cm):\n{np.round(H, 4)}")

            break

    # ── T_checker → robot ─────────────────────────────────────
    T = get_T_checker_to_robot(cam_t[2])
    return cam_R, cam_t, H, T, new_K

# ─────────────────────────────────────────────────────────────
#  T_CHECKER → ROBOT
# ─────────────────────────────────────────────────────────────
def get_T_checker_to_robot(auto_camera_height_cm):
    """
    Builds the 4×4 homogeneous transform T_checker→robot.

    KAI'S VERSION (Camera.m, checkerboardPointsToWorldPoints):
        R_0_checker = [0 1 0; 1 0 0; 0 0 -1]   ← board rotated 90° from robot
        t_0_checker = [113; -95; 0]             ← measured physically in mm

    YOUR VERSION:
        R depends on how you placed the board (see options below).
        t = [dx, dy, 0] measured with a ruler in cm.

    NOTE: Camera height (cam_t[2]) was computed automatically
    from solvePnP — no ruler needed for that!
    """
    print()
    print("─" * 58)
    print("  RULER MEASUREMENT — robot base → board origin")
    print("─" * 58)
    print()
    print(f"  Camera height was auto-computed: {auto_camera_height_cm:.2f} cm")
    print("  (no need to measure that manually)")
    print()
    print("  Now measure the board TRANSLATION from the robot base:")
    print("    Robot origin = centre of base servo shaft (servo 1).")
    print("    Board origin = TOP-LEFT INNER CORNER of the glued board.")
    print()
    print("  Stand behind the robot (looking toward the board).")
    print("    +X = rightward,  +Y = forward,  +Z = upward")
    print()
    print("  YOUR setup (confirmed from corners.png):")
    print("    dx is POSITIVE  (board origin is to the robot's RIGHT)")
    print("    dy is POSITIVE  (board origin is FORWARD of robot base)")
    print()
    print("  Enter your measured values:")
    print("    dx = +19.0 cm")
    print("    dy =  +8.7 cm")
    print()

    while True:
        try:
            dx = float(input("  dx (cm): "))
            dy = float(input("  dy (cm): "))
            break
        except ValueError:
            print("  Enter a number.")

    print()
    print("  Board orientation vs robot axes (look at your board from above):")
    print("  [1] Board columns run LEFT→RIGHT  = robot +X  (default)")
    print("      (board is aligned with robot — most common setup)")
    print("  [2] Board columns run AWAY→NEAR   = robot +Y")
    print("      (board rotated 90° — like Kai's setup)")
    print("  [3] Custom angle (degrees, CCW from robot +X)")
    choice = input("  Choice [1]: ").strip() or "1"

    if choice == "2":
        # Kai's exact rotation: board_X→robot_Y, board_Y→robot_X
        R = np.array([[0,  1, 0],
                      [1,  0, 0],
                      [0,  0, 1]], dtype=np.float64)
        print("  Using Kai-style 90° rotation.")
    elif choice == "3":
        ang = float(input("  Angle (deg CCW): "))
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        R = np.array([[ c, -s, 0],
                      [ s,  c, 0],
                      [ 0,  0, 1]], dtype=np.float64)
    else:
        R = np.eye(3, dtype=np.float64)
        print("  Using identity rotation (board aligned with robot).")

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[0,   3] = dx
    T[1,   3] = dy

    print(f"\n  T_checker→robot:\n{np.round(T, 3)}")
    print()
    print("  TIP: If the arm consistently misses by ~N cm in one direction,")
    print("  re-run calibrate.py and adjust dx or dy by that amount.")
    return T

# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────
def main():
    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║   Robot Arm Vision — Camera Calibration                ║")
    print(f"║   Board: {BOARD_COLS}×{BOARD_ROWS} inner corners  |  {SQUARE_SIZE_CM} cm/square        ║")
    print("╠════════════════════════════════════════════════════════╣")
    print("║   Phase 1 — move board around (intrinsics)             ║")
    print("║   Phase 2 — place board in final spot (extrinsics)     ║")
    print("╚════════════════════════════════════════════════════════╝")

    cap = open_camera()

    K, dist, rms, img_shape = phase1_intrinsics(cap)
    cam_R, cam_t, H, T, new_K = phase2_extrinsics(cap, K, dist)

    cap.release()
    cv2.destroyAllWindows()

    np.savez(OUTPUT_FILE,
             K                  = K,
             new_K              = new_K,
             dist               = dist,
             H                  = H,
             T_checker_to_robot = T,
             cam_R              = cam_R,
             cam_t              = cam_t,
             board_cols         = BOARD_COLS,
             board_rows         = BOARD_ROWS,
             square_size_cm     = SQUARE_SIZE_CM,
             rms                = rms,
             img_shape          = np.array(img_shape))

    print()
    print("╔════════════════════════════════════════════════════════╗")
    print("║   CALIBRATION SAVED                                    ║")
    print(f"║   {OUTPUT_FILE:<55}║")
    print(f"║   Intrinsic RMS : {rms:.4f} px                             ║")
    print(f"║   Camera height : {cam_t[2]:.2f} cm  (auto-computed)          ║")
    print(f"║   Board offset  : dx={T[0,3]:.1f}  dy={T[1,3]:.1f} cm              ║")
    print("╠════════════════════════════════════════════════════════╣")
    print("║   Next step: python vision.py                          ║")
    print("╚════════════════════════════════════════════════════════╝")


if __name__ == "__main__":
    main()
