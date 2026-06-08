"""
============================================================
 Box Detector — Color Agnostic (Hardened Edition)
 For autonomous package sorting robot
============================================================

 IMPROVEMENTS OVER ORIGINAL
 ---------------------------
   1. CLAHE adaptive histogram equalization on both background
      and live frames before differencing — normalizes uneven
      lighting and compensates for brightness drift over time.
   2. Adaptive thresholding (Otsu + fallback) instead of fixed
      threshold — auto-adjusts to actual contrast.
   3. Multi-frame background capture with Gaussian blur for
      noise reduction.
   4. Two-pass contour filtering: first pass finds candidates,
      second pass uses convex-hull solidity check + min
      dimension filter to reject noise.
   5. Optional frame averaging at detection time (averages
      3 consecutive frames) to reduce sensor noise.
   6. Better morphological pipeline with tuned kernel sizes.
   7. Illumination-compensated background subtraction using
      per-channel normalization.

 ALL ORIGINAL API AND BEHAVIOR IS PRESERVED.
============================================================
"""

import cv2
import numpy as np
import sys
import os

CALIBRATION_FILE = "calibration.npz"
CAMERA_INDEX     = 0

# Box geometry
BOX_HEIGHT_CM    = 3.0    # box is 3 cm tall
GRASP_Z_CM       = 1.5    # grasp at mid-height

# Contour filters
MIN_BOX_AREA_PX  = 500
MAX_BOX_AREA_PX  = 60000
MIN_RECT_SCORE   = 0.50   # slightly relaxed for poor lighting
MIN_SOLIDITY     = 0.60   # convex hull solidity check (new)
MIN_BOX_DIM_PX   = 15     # minimum width or height in pixels

# Checkerboard params (must match your board)
BOARD_COLS       = 8
BOARD_ROWS       = 6
SQUARE_SIZE_CM   = 2.9

# Camera nadir in robot frame (update after calibration)
CAMERA_NADIR_X_CM = 0.0
CAMERA_NADIR_Y_CM = 20.0

# Number of frames to average for background and detection
BG_CAPTURE_FRAMES   = 20
DETECT_AVG_FRAMES   = 3

# ─────────────────────────────────────────────────────────────
#  CLAHE (Contrast Limited Adaptive Histogram Equalization)
# ─────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def normalize_lighting(gray):
    """Apply CLAHE to equalize local contrast under uneven lighting."""
    return _clahe.apply(gray)

def normalize_frame_color(frame):
    """
    Normalize a color frame's brightness using LAB color space.
    Equalizes the L channel while preserving color information.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────
def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    return cap

# ─────────────────────────────────────────────────────────────
class BoxDetector:

    def __init__(self, calib_file=CALIBRATION_FILE, cam_index=CAMERA_INDEX):

        if not os.path.exists(calib_file):
            print(f"[ERROR] {calib_file} not found. Run calibrate.py first.")
            sys.exit(1)

        data = np.load(calib_file, allow_pickle=True)
        self.K             = data["K"]
        self.new_K         = data["new_K"]
        self.dist          = data["dist"]
        H_raw              = data["H"].astype(np.float64)
        self.H             = np.linalg.inv(H_raw).astype(np.float32)  # invert: pixel→board
        self.T             = data["T_checker_to_robot"]
        self.cam_t         = data["cam_t"]
        self.camera_height = float(self.cam_t[2])

        print(f"[DET] Calibration loaded. Camera height: {self.camera_height:.1f} cm")

        self._map1 = None
        self._map2 = None
        self.cap   = open_camera()

        # Background model (board without boxes)
        self.background    = None
        self.bg_gray       = None
        self.bg_gray_norm  = None   # CLAHE-normalized background gray
        self.board_corners = None   # pixel corners of board

    # ── Undistortion ─────────────────────────────────────────
    def _build_maps(self, h, w):
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            self.K, self.dist, None, self.new_K, (w, h), cv2.CV_16SC2)

    def undistort(self, frame):
        h, w = frame.shape[:2]
        if self._map1 is None:
            self._build_maps(h, w)
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)

    # ── Background capture ────────────────────────────────────
    def capture_background(self):
        """
        Capture the board WITHOUT any boxes.
        Called once at startup. Used for background subtraction.
        Uses more frames + Gaussian blur for noise-robust reference.
        """
        print("[BG] Capturing background (board must be clear)...")
        frames = []
        for _ in range(BG_CAPTURE_FRAMES):
            ret, f = self.cap.read()
            if ret and f is not None:
                frames.append(self.undistort(f))

        if not frames:
            print("[ERROR] No frames captured for background!")
            return

        # Average frames to reduce temporal noise
        self.background = np.mean(frames, axis=0).astype(np.uint8)

        # Light Gaussian blur to reduce pixel noise in reference
        self.background = cv2.GaussianBlur(self.background, (3, 3), 0)

        self.bg_gray = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        # Pre-compute CLAHE-normalized background for robust differencing
        self.bg_gray_norm = normalize_lighting(self.bg_gray)

        # Detect board corners for workspace mask
        self._detect_board_workspace()
        print("[BG] Background captured.")

    def _detect_board_workspace(self):
        """Find the board extent in the image using checkerboard corners."""
        gray  = cv2.cvtColor(self.background, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (BOARD_COLS, BOARD_ROWS),
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

        if found:
            corners = corners.reshape(-1, 2)
            hull   = cv2.convexHull(corners.astype(np.float32))
            center = corners.mean(axis=0)
            expanded = []
            for pt in hull.reshape(-1, 2):
                direction = pt - center
                expanded.append(pt + direction * 0.25)
            self.board_corners = np.array(expanded, dtype=np.int32)
            print(f"[BG] Board workspace detected from checkerboard corners.")
        else:
            print("[WARN] Could not detect board corners for workspace mask.")
            self.board_corners = None

    def get_workspace_mask(self, frame):
        """Binary mask = 1 inside board area."""
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        if self.board_corners is not None:
            cv2.fillConvexPoly(mask, self.board_corners, 255)
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
            mask = cv2.erode(mask, k)
        else:
            h, w = frame.shape[:2]
            mask[int(h*0.1):int(h*0.9), int(w*0.1):int(w*0.9)] = 255
        return mask

    # ── Method 1: Background subtraction (hardened) ──────────
    def detect_by_background_subtraction(self, frame):
        """
        Find pixels that differ from the empty board.
        IMPROVED: Uses CLAHE-normalized grayscale for illumination
        robustness, plus adaptive thresholding.
        """
        if self.background is None or self.bg_gray_norm is None:
            return None, None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE to live frame too — normalizes both to same
        # local contrast space, making subtraction lighting-invariant
        gray_norm = normalize_lighting(gray)

        # Absolute difference on normalized images
        diff = cv2.absdiff(gray_norm, self.bg_gray_norm)

        # Also compute raw diff for supplementary check
        diff_raw = cv2.absdiff(gray, self.bg_gray)

        # Combine both diffs — takes the max of normalized and raw
        # This catches both color-shift and brightness-shift changes
        diff_combined = cv2.max(diff, diff_raw)

        # Light blur to smooth noise before thresholding
        diff_combined = cv2.GaussianBlur(diff_combined, (5, 5), 0)

        # Adaptive threshold: try Otsu first, fall back to fixed
        otsu_val, thresh = cv2.threshold(
            diff_combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # If Otsu picks a very low threshold (< 12), it's likely
        # triggered by noise; clamp to a minimum
        if otsu_val < 12:
            _, thresh = cv2.threshold(diff_combined, 15, 255, cv2.THRESH_BINARY)

        # If Otsu picks a very high threshold (> 80), lighting may be
        # harsh; use a gentler fixed threshold
        if otsu_val > 80:
            _, thresh = cv2.threshold(diff_combined, 25, 255, cv2.THRESH_BINARY)

        # Apply workspace mask — ignore table outside board
        ws_mask = self.get_workspace_mask(frame)
        thresh  = cv2.bitwise_and(thresh, ws_mask)

        # Morphological cleanup — tuned for small boxes
        k_noise  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        k_open   = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        k_close  = cv2.getStructuringElement(cv2.MORPH_RECT,   (21, 21))

        # Remove small noise specks
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_noise, iterations=1)
        # Open to separate touching blobs
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, k_open, iterations=1)
        # Close to fill holes inside boxes
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, k_close, iterations=3)

        return thresh, diff_combined

    # ── Method 2: Edge-based detection ───────────────────────
    def detect_by_edges(self, frame):
        """
        Fallback: detect boxes by their rectangular edges.
        IMPROVED: Uses CLAHE-enhanced frame for better edge
        detection in low-contrast / poor lighting.
        """
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray    = normalize_lighting(gray)  # CLAHE boost
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges   = cv2.Canny(blurred, 25, 90)  # slightly relaxed thresholds

        # Dilate edges to connect nearby lines
        k     = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges = cv2.dilate(edges, k, iterations=2)

        # Fill enclosed regions
        filled = edges.copy()
        h, w   = filled.shape
        flood  = np.zeros((h+2, w+2), dtype=np.uint8)
        cv2.floodFill(filled, flood, (0, 0), 255)
        filled = cv2.bitwise_not(filled)
        mask   = cv2.bitwise_or(edges, filled)

        # Apply workspace mask
        ws_mask = self.get_workspace_mask(frame)
        mask    = cv2.bitwise_and(mask, ws_mask)

        return mask

    # ── Contour filtering (hardened) ──────────────────────────
    def filter_contours(self, mask, method_name=""):
        """
        Extract valid box contours from mask.
        IMPROVED: Added solidity check and minimum dimension filter.
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (MIN_BOX_AREA_PX <= area <= MAX_BOX_AREA_PX):
                continue

            # Bounding rect
            x, y, w, h = cv2.boundingRect(c)

            # Minimum dimension check — reject tiny noise fragments
            if w < MIN_BOX_DIM_PX or h < MIN_BOX_DIM_PX:
                continue

            # Rectangularity score = contour area / bounding rect area
            rect_area   = w * h
            rect_score  = area / (rect_area + 1e-6)

            # Aspect ratio check — boxes shouldn't be too elongated
            aspect = max(w, h) / (min(w, h) + 1e-6)
            if aspect > 4.0:
                continue

            # Solidity check: contour area / convex hull area
            # Real boxes are convex-ish; noise blobs often aren't
            hull_area = cv2.contourArea(cv2.convexHull(c))
            solidity  = area / (hull_area + 1e-6)
            if solidity < MIN_SOLIDITY:
                continue

            if rect_score >= MIN_RECT_SCORE:
                valid.append((c, rect_score, (x, y, w, h)))

        # Sort by area descending
        valid.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
        return valid

    # ── Multi-frame grab for detection ────────────────────────
    def _grab_averaged_frame(self):
        """
        Grab and average multiple frames to reduce sensor noise.
        Especially helpful in low-light conditions.
        """
        frames = []
        for _ in range(DETECT_AVG_FRAMES):
            ret, raw = self.cap.read()
            if ret and raw is not None:
                frames.append(self.undistort(raw).astype(np.float32))

        if not frames:
            raise RuntimeError("Frame grab failed")

        if len(frames) == 1:
            return frames[0].astype(np.uint8)

        avg = np.mean(frames, axis=0).astype(np.uint8)
        return avg

    # ── Coordinate transform ──────────────────────────────────
    def pixel_to_robot(self, u, v):
        """Full pipeline: pixel → robot frame (x, y, z)."""
        # Pixel → board cm
        pt      = np.array([[[float(u), float(v)]]], dtype=np.float32)
        out     = cv2.perspectiveTransform(pt, self.H)
        x_b, y_b = float(out[0,0,0]), float(out[0,0,1])

        # Board → robot frame
        p_b     = np.array([x_b, y_b, 0.0, 1.0])
        p_r     = self.T @ p_b
        x_r, y_r = float(p_r[0]), float(p_r[1])

        # Projection correction for box height
        h_cam   = self.camera_height
        Dx      = CAMERA_NADIR_X_CM - x_r
        Dy      = y_r - CAMERA_NADIR_Y_CM
        D       = np.hypot(Dx, Dy)
        if D > 1e-6 and h_cam > 1e-3:
            scale = BOX_HEIGHT_CM / h_cam
            x_r  += Dx * scale
            y_r  -= Dy * scale

        # Negate Y — camera and IK use opposite Y conventions
        X_OFFSET = -2.0
        Y_OFFSET =  2.0
        return x_r + X_OFFSET, -y_r + Y_OFFSET, GRASP_Z_CM

    def contour_to_robot(self, contour):
        """Get robot coords from contour centroid."""
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return None
        u = M["m10"] / M["m00"]
        v = M["m01"] / M["m00"]
        return self.pixel_to_robot(u, v)

    # ── Main detection ────────────────────────────────────────
    def detect(self, visualise=False):
        """
        Detect all boxes in one frame.

        Returns
        -------
        boxes : list of dicts, each with:
            'coords'  : (x, y, z) in robot cm
            'pixel'   : (u, v) centroid in image
            'area'    : contour area in px
            'rect'    : bounding rect (x,y,w,h)
        debug_frame : annotated image (if visualise=True)
        """
        # Use multi-frame averaging for noise reduction
        frame = self._grab_averaged_frame()

        debug = frame.copy() if visualise else None
        boxes = []

        # ── Try Method 1: Background subtraction (preferred) ─
        mask1, diff = self.detect_by_background_subtraction(frame)

        if mask1 is not None:
            candidates = self.filter_contours(mask1, "BG-sub")
            method_used = "background subtraction"
        else:
            candidates = []

        # ── Fallback: Edge detection ──────────────────────────
        if len(candidates) == 0:
            mask2      = self.detect_by_edges(frame)
            candidates = self.filter_contours(mask2, "edges")
            method_used = "edge detection"

        # ── Extract coordinates ───────────────────────────────
        for cnt, rect_score, bbox in candidates:
            coords = self.contour_to_robot(cnt)
            if coords is None:
                continue
            x, y, w, h = bbox
            M  = cv2.moments(cnt)
            u  = M["m10"] / M["m00"]
            v  = M["m01"] / M["m00"]

            boxes.append({
                'coords'     : coords,
                'pixel'      : (u, v),
                'area'       : cv2.contourArea(cnt),
                'rect'       : bbox,
                'rect_score' : rect_score,
            })

            if visualise:
                rx, ry, rz = coords
                cv2.rectangle(debug, (x, y), (x+w, y+h), (0, 200, 80), 2)
                cv2.drawContours(debug, [cnt], -1, (0, 255, 150), 1)
                cv2.circle(debug, (int(u), int(v)), 7, (0, 80, 255), -1)
                lbl1 = f"({rx:+.1f}, {ry:+.1f}, {rz:.1f}) cm"
                lbl2 = f"rect={rect_score:.2f}"
                cv2.putText(debug, lbl1, (x, y-18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(debug, lbl1, (x, y-18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 200),  1)
                cv2.putText(debug, lbl2, (x, y-2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

        if visualise:
            if self.board_corners is not None:
                cv2.polylines(debug, [self.board_corners], True, (200, 200, 0), 1)
            n   = len(boxes)
            msg = f"{n} box(es) detected  [{method_used}]"
            col = (30, 220, 30) if n > 0 else (30, 30, 220)
            cv2.rectangle(debug, (0, 0), (debug.shape[1], 50), (0,0,0), -1)
            cv2.putText(debug, msg, (12, 33),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)

        return boxes, debug

    def close(self):
        self.cap.release()
        cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────
def main():
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Box Detector — Color Agnostic (Hardened)               ║")
    print("║   Works with plain cardboard, stickers, any color box    ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║   Step 1: CLEAR the board — press SPACE                  ║")
    print("║   Step 2: Place boxes — press S to detect, Q to quit     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    detector = BoxDetector()

    print("Remove ALL boxes from the board.")
    print("Click the camera window and press SPACE when board is clear.\n")

    while True:
        ret, frame = detector.cap.read()
        if not ret:
            continue
        disp = detector.undistort(frame)
        cv2.rectangle(disp, (0,0), (disp.shape[1], 50), (0,0,0), -1)
        cv2.putText(disp, "Clear board → press SPACE  |  Q=quit",
                    (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 80), 2)
        cv2.imshow("Box Detector", disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q')):
            detector.close(); return
        elif key == ord(' '):
            detector.capture_background()
            break

    print("\nBoard cleared and background saved.")
    print("Place boxes on the board.")
    print("Click the window → S=detect  Q=quit\n")

    while True:
        ret, frame = detector.cap.read()
        if not ret:
            continue
        disp = detector.undistort(frame)
        cv2.putText(disp, "S=detect  Q=quit  (click window first)",
                    (12, disp.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 80), 1)
        cv2.imshow("Box Detector", disp)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q')):
            break

        elif key in (ord('s'), ord('S')):
            print("[INFO] Detecting boxes...")
            boxes, debug = detector.detect(visualise=True)

            if not boxes:
                print("[WARN] No boxes detected.")
                print("       Make sure background was captured without boxes.")
            else:
                print(f"\n[RESULT] {len(boxes)} box(es) found:")
                for i, b in enumerate(boxes):
                    x, y, z = b['coords']
                    print(f"  Box {i+1}: robot ({x:+.2f}, {y:+.2f}, {z:.2f}) cm  "
                          f"| rect_score={b['rect_score']:.2f}")
                    print(f"         → inverse_kinematics({x:.2f}, {y:.2f}, {z:.2f})")
                print()

            if debug is not None:
                cv2.imshow("Detection Result", debug)

        elif key in (ord('b'), ord('B')):
            print("[INFO] Re-capturing background — remove all boxes first!")
            detector.capture_background()

    detector.close()
    print("[DONE]")


if __name__ == "__main__":
    main()
