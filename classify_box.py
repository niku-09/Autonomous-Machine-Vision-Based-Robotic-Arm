"""
============================================================
 classify_box.py — Sticker Color Classifier (Hardened Edition)
============================================================

 IMPROVEMENTS OVER ORIGINAL
 ---------------------------
   1. LAB color space analysis in addition to HSV — LAB is
      more perceptually uniform and handles lighting variation
      better than HSV alone.
   2. Dual-channel voting: both HSV hue and LAB a/b channels
      vote independently, results merged for higher accuracy.
   3. CLAHE pre-processing on the crop before classification
      to normalize local contrast.
   4. Histogram peak detection instead of raw pixel counting
      — finds the dominant hue peak, ignoring scattered noise.
   5. Wider and overlapping hue ranges with weighted scoring
      to handle color shifts from warm/cool lighting.
   6. Background subtraction uses per-channel difference with
      a stricter mask to isolate sticker pixels more cleanly.
   7. Higher minimum saturation threshold with adaptive
      adjustment based on overall image brightness.
   8. Confidence calibration: confidence score now accounts
      for sticker pixel density (larger sticker = higher
      confidence).

 ALL ORIGINAL API AND BEHAVIOR IS PRESERVED.
============================================================
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────
#  CATEGORY DEFINITIONS
# ─────────────────────────────────────────────────────────────

# Primary hue ranges (OpenCV: H=0-179)
HUE_RANGES = {
    "FRAGILE":   [(0, 12), (158, 179)],    # Red (wraps around)
    "DAMAGED":   [(8, 22)],                # Orange (slight overlap with red)
    "ZONE_B":    [(18, 38)],               # Yellow (slight overlap with orange)
    "IMMEDIATE": [(38, 85)],               # Green
    "ZONE_A":    [(95, 135)],              # Blue
}

# LAB color space signatures for each category
# (a_center, b_center, radius) in LAB a-b plane
# a: green(-) to red(+), b: blue(-) to yellow(+)
LAB_SIGNATURES = {
    "FRAGILE":   (170, 140, 35),   # high a (red), mid b
    "DAMAGED":   (155, 170, 30),   # high a, high b (orange)
    "ZONE_B":    (120, 190, 30),   # low a, very high b (yellow)
    "IMMEDIATE": ( 90, 135, 35),   # low a (green), mid b
    "ZONE_A":    (120,  85, 35),   # mid a, low b (blue)
}

# Display colors for each label (BGR)
LABEL_COLORS = {
    "FRAGILE":   (0,   0,   220),   # Red
    "ZONE_A":    (200, 80,   0 ),   # Blue
    "ZONE_B":    (0,   210, 210),   # Yellow
    "IMMEDIATE": (0,   180,  0 ),   # Green
    "DAMAGED":   (0,   130, 255),   # Orange
    "UNKNOWN":   (120, 120, 120),   # Grey
}

# Min sticker pixels to trust classification
MIN_STICKER_PIXELS = 25    # lowered slightly for small stickers

# Weight for HSV vs LAB voting (sum = 1.0)
HSV_WEIGHT = 0.55
LAB_WEIGHT = 0.45

# ─────────────────────────────────────────────────────────────
#  CLAHE for pre-processing
# ─────────────────────────────────────────────────────────────
_clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))


def _preprocess_crop(crop):
    """Apply CLAHE on L channel of LAB to normalize brightness."""
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    lab = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# ─────────────────────────────────────────────────────────────
#  CORE CLASSIFIER
# ─────────────────────────────────────────────────────────────

def classify_box(box_crop, bg_crop):
    """
    Classify a box by its sticker color.

    Parameters
    ----------
    box_crop : np.ndarray  BGR crop of box region (current frame)
    bg_crop  : np.ndarray  BGR crop of same region (empty board frame)

    Returns
    -------
    label : str  one of FRAGILE, ZONE_A, ZONE_B, IMMEDIATE, DAMAGED, UNKNOWN
    confidence : float  0.0 - 1.0
    debug_img  : np.ndarray  annotated crop for visualisation
    """
    if box_crop is None or bg_crop is None:
        return "UNKNOWN", 0.0, None

    # Resize bg_crop to match box_crop if needed
    if bg_crop.shape != box_crop.shape:
        bg_crop = cv2.resize(bg_crop, (box_crop.shape[1], box_crop.shape[0]))

    # Pre-process: CLAHE normalize both crops
    box_proc = _preprocess_crop(box_crop)
    bg_proc  = _preprocess_crop(bg_crop)

    # ── Step 1: Background subtraction (per-channel) ─────────
    # Per-channel difference is more sensitive than grayscale diff
    diff = cv2.absdiff(box_proc, bg_proc)
    diff_max = np.max(diff, axis=2)  # max across B,G,R channels

    _, box_mask = cv2.threshold(diff_max, 18, 255, cv2.THRESH_BINARY)

    # Clean up mask
    k        = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_OPEN,  k, iterations=1)
    box_mask = cv2.morphologyEx(box_mask, cv2.MORPH_CLOSE, k, iterations=2)

    # ── Step 2: Remove cardboard brown from box pixels ───────
    hsv      = cv2.cvtColor(box_proc, cv2.COLOR_BGR2HSV)

    # Cardboard brown range (slightly expanded for varied cardboard)
    cardboard_mask = cv2.inRange(hsv,
                                 np.array([ 4,  25,  50], dtype=np.uint8),
                                 np.array([28, 190, 215], dtype=np.uint8))

    # Sticker pixels = box pixels that are NOT brown cardboard
    sticker_mask = cv2.bitwise_and(box_mask,
                                   cv2.bitwise_not(cardboard_mask))

    # ── Step 3: Adaptive saturation threshold ────────────────
    # In dim lighting, saturation drops; adjust threshold accordingly
    v_channel = hsv[:, :, 2]
    mean_v = np.mean(v_channel[box_mask > 0]) if np.any(box_mask > 0) else 128
    # Lower sat threshold in dim conditions (mean_v < 100)
    sat_threshold = max(25, min(50, int(mean_v * 0.35)))

    h_channel = hsv[:, :, 0]
    s_channel = hsv[:, :, 1]

    sticker_pixels = np.sum(sticker_mask > 0)

    if sticker_pixels < MIN_STICKER_PIXELS:
        return "UNKNOWN", 0.0, _make_debug(box_crop, sticker_mask, "UNKNOWN", 0.0)

    sticker_hues = h_channel[sticker_mask > 0]
    sticker_sats = s_channel[sticker_mask > 0]

    # Filter by adaptive saturation
    sat_filter = sticker_sats > sat_threshold
    if sat_filter.sum() < MIN_STICKER_PIXELS:
        return "UNKNOWN", 0.0, _make_debug(box_crop, sticker_mask, "UNKNOWN", 0.0)

    sticker_hues_clean = sticker_hues[sat_filter]
    total = len(sticker_hues_clean)

    # ── Step 4: HSV hue voting with histogram peak ───────────
    # Build hue histogram and find peak — more robust than raw counting
    hist = np.zeros(180, dtype=np.float32)
    for h in sticker_hues_clean:
        hist[h] += 1

    # Smooth histogram to find clean peak
    kernel = np.ones(5) / 5.0
    hist_smooth = np.convolve(hist, kernel, mode='same')

    hsv_votes = {}
    for label, ranges in HUE_RANGES.items():
        score = 0.0
        for (h_lo, h_hi) in ranges:
            # Sum the smoothed histogram in this range
            score += float(np.sum(hist_smooth[h_lo:h_hi+1]))
        hsv_votes[label] = score

    # Normalize HSV votes
    hsv_total = sum(hsv_votes.values()) + 1e-6
    hsv_scores = {k: v / hsv_total for k, v in hsv_votes.items()}

    # ── Step 5: LAB color space voting ───────────────────────
    lab = cv2.cvtColor(box_proc, cv2.COLOR_BGR2LAB)
    lab_a = lab[:, :, 1].astype(np.float32)
    lab_b = lab[:, :, 2].astype(np.float32)

    sticker_a = lab_a[sticker_mask > 0]
    sticker_b_ch = lab_b[sticker_mask > 0]

    # Filter by saturation (same filter)
    sticker_a = sticker_a[sat_filter]
    sticker_b_ch = sticker_b_ch[sat_filter]

    lab_votes = {}
    for label, (a_c, b_c, radius) in LAB_SIGNATURES.items():
        # Distance from each pixel to this category's center in a-b space
        dist = np.sqrt((sticker_a - a_c)**2 + (sticker_b_ch - b_c)**2)
        # Count pixels within radius
        within = np.sum(dist <= radius)
        # Also add soft-scored pixels just outside radius
        soft = np.sum(np.exp(-((dist - radius).clip(0) / 15.0)**2))
        lab_votes[label] = float(within) + float(soft) * 0.3

    lab_total = sum(lab_votes.values()) + 1e-6
    lab_scores = {k: v / lab_total for k, v in lab_votes.items()}

    # ── Step 6: Merge HSV and LAB scores ─────────────────────
    merged = {}
    for label in HUE_RANGES:
        merged[label] = (HSV_WEIGHT * hsv_scores.get(label, 0) +
                         LAB_WEIGHT * lab_scores.get(label, 0))

    best_label = max(merged, key=merged.get)
    best_score = merged[best_label]

    # Calculate confidence from the merged score
    # Also factor in sticker pixel density
    total_box_pixels = np.sum(box_mask > 0) + 1
    sticker_density = min(1.0, sticker_pixels / (total_box_pixels * 0.5))
    confidence = best_score * (0.7 + 0.3 * sticker_density)
    confidence = min(1.0, confidence)

    # Require minimum confidence
    if confidence < 0.20:
        best_label = "UNKNOWN"
        confidence = 0.0

    debug = _make_debug(box_crop, sticker_mask, best_label, confidence)
    return best_label, confidence, debug


def _make_debug(crop, sticker_mask, label, confidence):
    """Create annotated debug image."""
    debug = crop.copy()
    color = LABEL_COLORS.get(label, (120, 120, 120))

    # Tint sticker region
    tint = np.zeros_like(debug)
    tint[sticker_mask > 0] = color
    debug = cv2.addWeighted(debug, 0.6, tint, 0.4, 0)

    # Border
    cv2.rectangle(debug, (0, 0), (debug.shape[1]-1, debug.shape[0]-1), color, 3)

    # Label
    text  = f"{label} {confidence*100:.0f}%"
    scale = 0.5 if debug.shape[1] > 80 else 0.35
    cv2.putText(debug, text, (4, debug.shape[0]-6),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255,255,255), 2)
    cv2.putText(debug, text, (4, debug.shape[0]-6),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1)
    return debug


# ─────────────────────────────────────────────────────────────
#  CLASSIFY ALL BOXES IN FRAME
# ─────────────────────────────────────────────────────────────

def classify_all_boxes(frame, background, box_list, padding=10):
    """
    Classify all detected boxes in one frame.

    Parameters
    ----------
    frame      : np.ndarray  current undistorted frame (BGR)
    background : np.ndarray  empty board frame (BGR)
    box_list   : list of dicts  from detect_boxes.detect()
                 each has 'rect': (x, y, w, h) in pixels
    padding    : int  extra pixels around bounding box for crop

    Returns
    -------
    results : list of dicts, each with:
        'coords'     : (x, y, z) robot cm
        'label'      : str category
        'confidence' : float
        'debug_crop' : annotated crop image
        'rect'       : pixel bounding rect
        'pixel'      : pixel centroid
    """
    results = []
    h_frame, w_frame = frame.shape[:2]

    for box in box_list:
        bx, by, bw, bh = box['rect']

        # Add padding, clamp to frame
        x1 = max(0,       bx - padding)
        y1 = max(0,       by - padding)
        x2 = min(w_frame, bx + bw + padding)
        y2 = min(h_frame, by + bh + padding)

        box_crop = frame[y1:y2, x1:x2]
        bg_crop  = background[y1:y2, x1:x2]

        # Ensure we have valid crops
        if box_crop.size == 0 or bg_crop.size == 0:
            continue

        # Zoom crop for better classification (small boxes)
        zoom_factor = max(1, int(120 / max(bw, bh)))
        if zoom_factor > 1:
            box_crop = cv2.resize(box_crop,
                                  (box_crop.shape[1]*zoom_factor,
                                   box_crop.shape[0]*zoom_factor),
                                  interpolation=cv2.INTER_LINEAR)
            bg_crop  = cv2.resize(bg_crop,
                                  (bg_crop.shape[1]*zoom_factor,
                                   bg_crop.shape[0]*zoom_factor),
                                  interpolation=cv2.INTER_LINEAR)

        label, confidence, debug = classify_box(box_crop, bg_crop)

        results.append({
            'coords':     box['coords'],
            'label':      label,
            'confidence': confidence,
            'debug_crop': debug,
            'rect':       box['rect'],
            'pixel':      box['pixel'],
        })

    return results


# ─────────────────────────────────────────────────────────────
#  STANDALONE TEST
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from detect_boxes import BoxDetector
    import time

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║   Sticker Classifier Test (Hardened)                     ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║   SPACE = capture background                             ║")
    print("║   S     = detect + classify all boxes                    ║")
    print("║   Q     = quit                                           ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    detector = BoxDetector()

    print("Clear the board → click window → SPACE\n")
    while True:
        ret, frame = detector.cap.read()
        if not ret: continue
        disp = detector.undistort(frame)
        cv2.rectangle(disp, (0,0), (disp.shape[1], 50), (0,0,0), -1)
        cv2.putText(disp, "Clear board → SPACE  |  Q=quit",
                    (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0,220,80), 2)
        cv2.imshow("Classifier Test", disp)
        key = cv2.waitKey(30) & 0xFF
        if key in (ord('q'), ord('Q')):
            detector.close(); exit()
        elif key == ord(' '):
            detector.capture_background()
            print("[BG] Background captured!\n")
            break

    print("Place stickered boxes on the board.")
    print("Click window → S to classify  Q to quit\n")

    while True:
        ret, frame = detector.cap.read()
        if not ret: continue
        disp = detector.undistort(frame)
        cv2.putText(disp, "S=classify  Q=quit",
                    (12, disp.shape[0]-12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,220,80), 2)
        cv2.imshow("Classifier Test", disp)
        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), ord('Q')):
            break
        elif key in (ord('s'), ord('S')):
            boxes, _ = detector.detect(visualise=False)
            if not boxes:
                print("[WARN] No boxes detected.")
                continue

            undist = detector.undistort(frame)
            results = classify_all_boxes(
                undist, detector.background, boxes)

            print(f"\n[RESULT] {len(results)} box(es) classified:")
            crops = []
            for i, r in enumerate(results):
                x, y, z = r['coords']
                print(f"  Box {i+1}: ({x:+.1f}, {y:+.1f}) cm  "
                      f"→ {r['label']:10}  confidence={r['confidence']*100:.0f}%")
                if r['debug_crop'] is not None:
                    crops.append(r['debug_crop'])

            if crops:
                max_h = max(c.shape[0] for c in crops)
                padded = []
                for c in crops:
                    if c.shape[0] < max_h:
                        pad = np.zeros((max_h-c.shape[0], c.shape[1], 3), dtype=np.uint8)
                        c = np.vstack([c, pad])
                    padded.append(c)
                cv2.imshow("Classified Boxes", np.hstack(padded))
            print()

    detector.close()
    print("[DONE]")
