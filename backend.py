"""OCR back-end: printed text (Tesseract, optional EasyOCR fallback) and
handwriting (TrOCR on automatically segmented lines).

Public API (unchanged for the GUI / tests):
    preprocess_image(path)            -> (gray, processed, original)
    extract_text(image)               -> str
    process_and_extract(path)         -> (gray, processed, original, text)
    extract_handwritten_text_by_line  -> str
    handle_user_choice(path, choice)  -> dict
"""
import math
import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output
from PIL import Image

import config
from image_utils import (
    ensure_dark_on_light, estimate_skew, fit_scale, normalize_background,
    read_image, rotate_keep_size, to_gray,
)

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


# ===========================================================================
# Printed text
# ===========================================================================
def preprocess_image(image_path):
    """Prepare a page/word image for Tesseract.

    Steps: deskew -> remove shadows -> scale to a good text size -> pad.
    Returns (normalized_gray, processed, original_bgr); (None, None, None)
    when the file cannot be read.
    """
    image = read_image(image_path)
    if image is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None, None, None

    # Light text on a dark background (slides, screenshots, dark mode)
    # must be flipped first: everything below assumes dark ink on paper.
    fixed, _ = ensure_dark_on_light(image)
    gray = to_gray(fixed)

    angle = estimate_skew(gray)
    if abs(angle) > 0.3:
        gray = rotate_keep_size(gray, angle, border=255)

    normalized = normalize_background(gray)
    processed = fit_scale(normalized)
    # Tesseract needs some white margin around the text.
    processed = cv2.copyMakeBorder(processed, 20, 20, 20, 20,
                                   cv2.BORDER_CONSTANT, value=255)
    return normalized, processed, image


def _tesseract_run(image, psm):
    """Run Tesseract once; return (text, mean_confidence, quality_score)."""
    data = pytesseract.image_to_data(
        image,
        lang=config.OCR_LANG,
        config=f"--oem 1 --psm {psm}",
        output_type=Output.DICT,
    )

    lines = {}
    confs = []
    weight = 0.0
    for i, word in enumerate(data["text"]):
        word = word.strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not word or conf < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
        confs.append(conf)
        weight += len(word) * conf / 100.0

    out, prev_par = [], None
    for key in sorted(lines):
        par = key[:2]
        if prev_par is not None and par != prev_par:
            out.append("")
        out.append(" ".join(lines[key]))
        prev_par = par

    mean_conf = float(np.mean(confs)) if confs else 0.0
    return "\n".join(out).strip(), mean_conf, weight


_easyocr_reader = None


def _easyocr_run(image):
    """Optional second engine. Returns (text, mean_confidence 0-100) or None."""
    global _easyocr_reader
    try:
        if _easyocr_reader is None:
            import easyocr
            import torch
            _easyocr_reader = easyocr.Reader(
                config.EASYOCR_LANGS, gpu=torch.cuda.is_available()
            )
        results = _easyocr_reader.readtext(image)
    except Exception as exc:  # not installed / failed -> silently skip
        print(f"[INFO] EasyOCR unavailable: {exc}")
        return None

    if not results:
        return "", 0.0

    items = []
    for box, text, conf in results:
        ys = [p[1] for p in box]
        xs = [p[0] for p in box]
        items.append({"cy": sum(ys) / 4, "h": max(ys) - min(ys),
                      "x": min(xs), "text": text, "conf": conf})
    items.sort(key=lambda d: d["cy"])

    rows, cur = [], [items[0]]
    for it in items[1:]:
        ref = np.mean([c["cy"] for c in cur])
        if abs(it["cy"] - ref) < 0.6 * max(it["h"], 1):
            cur.append(it)
        else:
            rows.append(cur)
            cur = [it]
    rows.append(cur)

    text = "\n".join(
        " ".join(c["text"] for c in sorted(r, key=lambda d: d["x"]))
        for r in rows
    )
    return text.strip(), float(np.mean([i["conf"] for i in items]) * 100)


def extract_text_with_confidence(image, psms=(3, 6), engine="auto"):
    """Return (text, confidence 0-100, engine_name).

    engine: "tesseract", "easyocr" or "auto" (Tesseract, then EasyOCR when the
    Tesseract confidence is low and EasyOCR is installed).
    """
    if image is None:
        return "", 0.0, "none"

    best = ("", 0.0, -1.0)  # text, conf, score
    if engine in ("auto", "tesseract"):
        try:
            for psm in psms:
                text, conf, score = _tesseract_run(image, psm)
                if score > best[2]:
                    best = (text, conf, score)
                if conf >= 85 and text:  # good enough, skip other modes
                    break
        except Exception as exc:
            print("[ERROR] Tesseract failed:", exc)

    name = "tesseract"
    use_easy = engine == "easyocr" or (
        engine == "auto" and best[1] < config.EASYOCR_FALLBACK_BELOW
    )
    if use_easy:
        easy = _easyocr_run(image)
        if easy is not None and (engine == "easyocr" or easy[1] > best[1]):
            best = (easy[0], easy[1], 0.0)
            name = "easyocr"

    text = re.sub(r"\n{3,}", "\n\n", best[0]).strip()
    return text, best[1], name


def extract_text(image, psms=(3, 6), engine="auto"):
    return extract_text_with_confidence(image, psms, engine)[0]


def process_and_extract(image_path, return_info=False):
    gray, processed, original = preprocess_image(image_path)
    if processed is None:
        return (None,) * (5 if return_info else 4)

    text, conf, engine = extract_text_with_confidence(processed)
    if return_info:
        return gray, processed, original, text, {
            "confidence": conf, "engine": engine}
    return gray, processed, original, text


# ===========================================================================
# Handwriting
# ===========================================================================
_hw = {}


def _get_handwriting_model():
    """Load TrOCR lazily (first use) so the GUI starts instantly."""
    if "model" not in _hw:
        import torch
        from transformers import TrOCRProcessor, VisionEncoderDecoderModel

        device = "cuda" if torch.cuda.is_available() else "cpu"
        name = config.HANDWRITING_MODEL
        if name == "auto":
            name = (config.HANDWRITING_LARGE if device == "cuda"
                    else config.HANDWRITING_BASE)
        print(f"[INFO] Loading {name} on {device} ...")

        processor = TrOCRProcessor.from_pretrained(name)
        model = VisionEncoderDecoderModel.from_pretrained(name)
        if device == "cuda":
            model = model.half()
        model.to(device).eval()
        _hw.update(model=model, processor=processor, device=device, name=name)
    return _hw["model"], _hw["processor"], _hw["device"]


def _valley_split(profile, start, end, parts, max_cut_ratio=None):
    """Cut [start, end) into `parts` pieces at the emptiest positions.

    If `max_cut_ratio` is given, a cut is only made where the profile is at
    most that fraction of its mean (i.e. a real gap); otherwise it is skipped.
    """
    mean = float(np.mean(profile[start:end])) if end > start else 0.0
    cuts = [start]
    span = (end - start) / parts
    for k in range(1, parts):
        ideal = start + k * span
        lo = int(max(cuts[-1] + 1, ideal - span * 0.4))
        hi = int(min(end - 1, ideal + span * 0.4))
        if hi <= lo:
            if max_cut_ratio is None:
                cuts.append(int(ideal))
            continue
        pos = lo + int(np.argmin(profile[lo:hi]))
        if max_cut_ratio is not None and profile[pos] > max_cut_ratio * mean:
            continue
        cuts.append(pos)
    cuts.append(end)
    return list(zip(cuts[:-1], cuts[1:]))


def _estimate_line_pitch(profile):
    """Distance between text lines from the autocorrelation of the row
    projection. Works even when lines touch and no row is empty."""
    n = len(profile)
    p = profile - profile.mean()
    if n < 40 or not np.any(p):
        return None
    acf = np.correlate(p, p, mode="full")[n - 1:]
    acf = acf / acf[0]
    lo, hi = max(8, n // 120), n // 2
    seg = acf[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1)
             if seg[i] > seg[i - 1] and seg[i] >= seg[i + 1] and seg[i] > 0.15]
    if not peaks:
        return None
    best = max(seg[i] for i in peaks)
    # smallest lag that is nearly as strong as the best one (avoids 2x pitch)
    pick = min(i for i in peaks if seg[i] >= 0.8 * best)
    return lo + pick, float(seg[pick])


def _comb_segments(smooth, pitch):
    """Split the page into lines for regularly spaced text (ruled paper).

    Cut points are placed one pitch apart, each snapped to the emptiest row
    nearby; the next cut is measured from the previous snapped one so slow
    drift in line spacing is followed. Returns [[top, bottom], ...].
    """
    H = len(smooth)
    P = int(round(pitch))
    phase = min(range(P), key=lambda ph: smooth[ph::P].sum())
    cuts = [0]
    pos = phase if phase > 0.3 * P else phase + P
    while pos < H - 0.3 * P:
        lo = int(max(cuts[-1] + 1, pos - 0.3 * P))
        hi = int(min(H - 1, pos + 0.3 * P))
        if hi <= lo:
            break
        cut = lo + int(np.argmin(smooth[lo:hi + 1]))
        cuts.append(cut)
        pos = cut + P
    cuts.append(H)
    segs = [[a, c] for a, c in zip(cuts[:-1], cuts[1:])]
    mass = np.array([smooth[a:c].sum() for a, c in segs])
    if not mass.any():
        return []
    keep = mass >= 0.2 * np.median(mass[mass > 0])
    return [s for s, k in zip(segs, keep) if k]


def _remove_specks(ink):
    """Delete tiny connected components (paper grain, dust)."""
    h, w = ink.shape
    min_area = max(12, int(1.5e-5 * h * w))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        ink, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_area
    keep[0] = False
    return keep[labels].astype(np.uint8) * 255


def _clean_page(image):
    """Return (clean_gray, ink_mask) with ruled lines suppressed."""
    gray = to_gray(image)
    h, w = gray.shape

    # Light blue / green notebook rules -> white (dark pen is kept).
    if image.ndim == 3:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        ruled = ((hsv[..., 0] >= 40) & (hsv[..., 0] <= 135)
                 & (hsv[..., 1] > 25) & (gray > 140))
        gray = gray.copy()
        gray[ruled] = 255

    paper = float(np.median(gray))
    block = max(31, (min(h, w) // 15) | 1)
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, block, 12)
    ink = cv2.bitwise_and(adaptive, (gray < paper - 35).astype(np.uint8) * 255)

    # Remove long horizontal rules (grey/black ruled paper).
    horiz = cv2.morphologyEx(
        ink, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, w // 8), 1)))
    horiz = cv2.dilate(horiz, np.ones((3, 1), np.uint8))
    ink = cv2.bitwise_and(ink, cv2.bitwise_not(horiz))
    gray = gray.copy()
    gray[horiz > 0] = 255

    # Grain / dust is removed by size (an opening would also erase thin pen
    # strokes).
    return gray, _remove_specks(ink)


def segment_handwriting_lines(image):
    """Find text lines. Returns (clean_gray, [ [ (x1,y1,x2,y2), ... ], ... ])
    where each inner list holds the crop boxes of one text line."""
    gray, ink = _clean_page(image)
    H, W = ink.shape

    rows = np.count_nonzero(ink, axis=1).astype(np.float32)
    win = max(3, H // 200)
    smooth = np.convolve(rows, np.ones(win, np.float32) / win, mode="same")
    if not np.any(smooth > 0):
        return gray, []

    # Threshold relative to the typical *ink* density, so grain in the
    # gaps cannot glue lines together.
    thr = max(2.0, 0.12 * float(np.percentile(smooth, 90)))
    active = smooth >= thr

    bands, y = [], 0
    while y < H:
        if not active[y]:
            y += 1
            continue
        top = y
        while y < H and active[y]:
            y += 1
        if y - top >= max(6, H // 150):
            bands.append([top, y])
    if not bands:
        return gray, []

    med = float(np.median([b - t for t, b in bands]))
    merged = []
    for top, bottom in bands:
        if merged and top - merged[-1][1] <= 0.35 * med:
            merged[-1][1] = bottom
        else:
            merged.append([top, bottom])

    # Typical line spacing: from the autocorrelation when possible (works
    # for touching lines), else from the band heights.
    est = _estimate_line_pitch(smooth)
    pitch, strength = est if est else (None, 0.0)

    if pitch and strength >= 0.35 and H / pitch >= 3:
        # Regularly spaced lines (ruled paper / dense paragraph).
        final = _comb_segments(smooth, pitch)
        # trim each segment to its actual ink rows
        trimmed = []
        for a, c in final:
            idx = np.where(smooth[a:c] >= thr)[0]
            if idx.size:
                trimmed.append([a + int(idx[0]), a + int(idx[-1]) + 1])
        final = trimmed
    else:
        heights = [b - t for t, b in merged]
        ref = float(pitch) if pitch else float(np.median(heights))
        can_split = pitch is not None or len(merged) >= 3
        final = []
        for top, bottom in merged:
            if can_split and (bottom - top) > 1.6 * ref:
                parts = max(2, int(round((bottom - top) / ref)))
                final.extend([list(p) for p in
                              _valley_split(smooth, top, bottom, parts)])
            else:
                final.append([top, bottom])
    if not final:
        return gray, []
    ref_h = float(np.median([b - t for t, b in final]))
    final = [b for b in final if (b[1] - b[0]) >= 0.3 * ref_h]

    lines = []
    for top, bottom in final:
        band = ink[top:bottom]
        cols = np.where(np.count_nonzero(band, axis=0) > 0)[0]
        if cols.size == 0:
            continue
        x1, x2 = int(cols[0]), int(cols[-1]) + 1
        bh = bottom - top
        py, px = int(0.25 * bh), int(0.5 * bh)
        y1, y2 = max(0, top - py), min(H, bottom + py)

        boxes = []
        n = math.ceil((x2 - x1) / (config.HANDWRITING_MAX_ASPECT * bh))
        if n > 1:
            col_profile = np.convolve(
                np.count_nonzero(band, axis=0).astype(np.float32),
                np.ones(5, np.float32) / 5, mode="same")
            # Only cut at genuine gaps between words, never through a word.
            pieces = _valley_split(col_profile, x1, x2, n, max_cut_ratio=0.2)
            for k, (a, b) in enumerate(pieces):
                left = max(0, a - px) if k == 0 else a
                right = min(W, b + px) if k == len(pieces) - 1 else b
                boxes.append((left, y1, right, y2))
        else:
            boxes.append((max(0, x1 - px), y1, min(W, x2 + px), y2))
        lines.append(boxes)

    return gray, lines


def _crop_to_pil(gray, box):
    x1, y1, x2, y2 = box
    crop = gray[y1:y2, x1:x2]
    lo, hi = np.percentile(crop, 1), np.percentile(crop, 99)
    if hi - lo > 10:  # stretch contrast so faint ink becomes dark
        crop = np.clip((crop.astype(np.float32) - lo) * 255.0 / (hi - lo),
                       0, 255).astype(np.uint8)
    crop = cv2.copyMakeBorder(crop, 10, 10, 10, 10,
                              cv2.BORDER_CONSTANT, value=255)
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_GRAY2RGB))


def _recognize_batch(images):
    import torch
    model, processor, device = _get_handwriting_model()
    pixel_values = processor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    if device == "cuda":
        pixel_values = pixel_values.half()
    with torch.inference_mode():
        ids = model.generate(
            pixel_values,
            max_new_tokens=config.HANDWRITING_MAX_TOKENS,
            num_beams=config.HANDWRITING_NUM_BEAMS,
            early_stopping=True,
        )
    return [t.strip() for t in
            processor.batch_decode(ids, skip_special_tokens=True)]


def recognize_handwriting(image_path, progress_cb=None):
    """Full handwriting pipeline. Returns (text, preview_bgr)."""
    image = read_image(image_path)
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    image, _ = ensure_dark_on_light(image)

    # Straighten photographed notebook pages.
    angle = estimate_skew(to_gray(image))
    if abs(angle) > 0.3:
        image = rotate_keep_size(image, angle, border=255)

    gray, lines = segment_handwriting_lines(image)
    preview = image.copy()
    print(f"[INFO] Detected {len(lines)} text lines")
    if not lines:
        return "", preview

    # Flatten pieces so they can be recognised in batches.
    pieces = [(li, box) for li, boxes in enumerate(lines) for box in boxes]
    for li, boxes in enumerate(lines):
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 160, 0), 2)
        cv2.putText(preview, str(li + 1), (max(0, boxes[0][0]), boxes[0][1] + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    crops = [_crop_to_pil(gray, box) for _, box in pieces]
    texts = []
    bs = config.HANDWRITING_BATCH_SIZE
    for start in range(0, len(crops), bs):
        texts.extend(_recognize_batch(crops[start:start + bs]))
        if progress_cb:
            progress_cb(min(start + bs, len(crops)), len(crops),
                        "Recognizing handwriting")

    per_line = {}
    for (li, _), text in zip(pieces, texts):
        per_line.setdefault(li, []).append(text)
    result = [" ".join(t for t in per_line[i] if t) for i in sorted(per_line)]
    return "\n".join(r for r in result if r), preview


def extract_handwritten_text_by_line(image_path, progress_cb=None):
    try:
        text, _ = recognize_handwriting(image_path, progress_cb)
        return text if text else "No handwriting detected."
    except Exception as exc:
        print(f"[ERROR] Handwriting OCR failed: {exc}")
        raise


# ===========================================================================
# Dispatcher used by the GUI
# ===========================================================================
def handle_user_choice(image_path, choice, progress_cb=None):
    """choice: 'printed' or 'handwritten'"""
    if choice == "printed":
        gray, processed, original, text, info = process_and_extract(
            image_path, return_info=True)
        return {"gray": gray, "processed": processed, "original": original,
                "text": text, "info": info}

    if choice == "handwritten":
        original = read_image(image_path)
        if original is None:
            raise ValueError(f"Could not load image: {image_path}")
        text, preview = recognize_handwriting(image_path, progress_cb)
        return {"gray": to_gray(original), "processed": preview,
                "original": original,
                "text": text or "No handwriting detected.",
                "info": {"model": _hw.get("name", "")}}

    raise ValueError("Invalid choice. Must be 'printed' or 'handwritten'.")
