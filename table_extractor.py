"""Table detection + cell OCR.

Pipeline: deskew -> detect ruling lines -> find every enclosed cell as a
connected component -> cluster cells into rows/columns -> OCR each cell
(in parallel). Tables without ruling lines fall back to a text-layout
analysis (best effort).

Public API:
    extract_tables_from_image(path) -> (annotated_bgr_image, cells_data)
        cells_data = [((col, row), text), ...]
    cells_to_grid / cells_to_dataframe / cells_to_csv / cells_to_xlsx
"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd
import pytesseract
from pytesseract import Output

import config
from image_utils import estimate_skew, read_image, rotate_keep_size, to_gray

pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _cluster(values, tol):
    """Group sorted 1-D values; returns the list of group centres."""
    values = sorted(values)
    if not values:
        return []
    groups, cur = [], [values[0]]
    for v in values[1:]:
        if v - cur[-1] <= tol:
            cur.append(v)
        else:
            groups.append(cur)
            cur = [v]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def _nearest(centres, value):
    return int(np.argmin([abs(c - value) for c in centres]))


def _ocr_cell(gray_cell, lang):
    """OCR one cell image (grayscale). Returns cleaned text."""
    if gray_cell.size == 0:
        return ""
    # Skip empty cells: almost no dark pixels.
    otsu = cv2.threshold(gray_cell, 0, 255,
                         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if np.count_nonzero(otsu) < 0.004 * otsu.size or gray_cell.std() < 8:
        return ""

    h, w = gray_cell.shape
    scale = 3.0 if h < 40 else 2.0 if h < 90 else 1.0
    if scale != 1.0:
        gray_cell = cv2.resize(gray_cell, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_CUBIC)
    gray_cell = cv2.copyMakeBorder(gray_cell, 12, 12, 12, 12,
                                   cv2.BORDER_CONSTANT, value=255)

    for psm in (6, 7, 8):  # block -> single line -> single word
        try:
            text = pytesseract.image_to_string(
                gray_cell, lang=lang, config=f"--oem 1 --psm {psm}"
            ).strip()
        except Exception:
            text = ""
        if text:
            return " ".join(text.split())
    return ""


# ---------------------------------------------------------------------------
# bordered tables
# ---------------------------------------------------------------------------
def _detect_cells(gray):
    """Return list of (x, y, w, h) boxes of enclosed cells, or []."""
    h, w = gray.shape
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 31, 10)

    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(25, w // 40), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(25, h // 40)))
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, hk)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vk)

    # Bridge small breaks in the lines, then thicken a little so that
    # cells are properly sealed.
    horizontal = cv2.morphologyEx(
        horizontal, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 120), 1)))
    vertical = cv2.morphologyEx(
        vertical, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, h // 120))))
    grid = cv2.dilate(cv2.bitwise_or(horizontal, vertical),
                      np.ones((3, 3), np.uint8))

    if np.count_nonzero(grid) < 0.002 * grid.size:
        return []

    # Frame the mask so cells touching the image edge are still closed.
    framed = cv2.copyMakeBorder(grid, 3, 3, 3, 3,
                                cv2.BORDER_CONSTANT, value=255)
    inverted = cv2.bitwise_not(framed)
    count, _, stats, _ = cv2.connectedComponentsWithStats(inverted,
                                                          connectivity=4)
    boxes = []
    H, W = framed.shape
    for i in range(1, count):
        x, y, cw, ch, area = stats[i]
        if cw < 15 or ch < 10:
            continue
        if cw > 0.95 * W and ch > 0.95 * H:
            continue
        if area / float(cw * ch) < 0.8:  # not rectangular -> background
            continue
        boxes.append((int(x - 3), int(y - 3), int(cw), int(ch)))

    # Reject results that are clearly not a table (e.g. a lone rectangle).
    if len(boxes) < 2:
        return []
    return boxes


def _assign_grid(boxes):
    """Map each box to (col, row) using clustered x/y positions."""
    med_w = float(np.median([b[2] for b in boxes]))
    med_h = float(np.median([b[3] for b in boxes]))
    xs = _cluster([b[0] for b in boxes], max(6, 0.25 * med_w))
    ys = _cluster([b[1] for b in boxes], max(6, 0.35 * med_h))
    return [(_nearest(xs, b[0]), _nearest(ys, b[1])) for b in boxes]


def _bordered_table(gray, lang):
    boxes = _detect_cells(gray)
    if not boxes:
        return None
    positions = _assign_grid(boxes)

    crops = []
    for (x, y, w, h) in boxes:
        pad = 3
        crops.append(gray[y + pad:y + h - pad, x + pad:x + w - pad])

    workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        texts = list(pool.map(lambda c: _ocr_cell(c, lang), crops))

    # If two boxes landed in the same slot keep the one with text.
    slots = {}
    for pos, box, text in zip(positions, boxes, texts):
        if pos not in slots or (text and not slots[pos][1]):
            slots[pos] = (box, text)
    return [(pos, box, text) for pos, (box, text) in slots.items()]


# ---------------------------------------------------------------------------
# borderless tables (best effort)
# ---------------------------------------------------------------------------
def _borderless_table(gray, lang):
    data = pytesseract.image_to_data(
        gray, lang=lang, config="--oem 1 --psm 6", output_type=Output.DICT)

    lines = {}
    for i, text in enumerate(data["text"]):
        text = text.strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1
        if not text or conf < 30:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(
            (data["left"][i], data["top"][i],
             data["width"][i], data["height"][i], text))
    if len(lines) < 2:
        return None

    heights = [w[3] for ws in lines.values() for w in ws]
    gap_limit = 1.8 * float(np.median(heights))

    # Split every line into "cells" wherever there is a wide horizontal gap.
    row_cells = []
    for key in sorted(lines, key=lambda k: min(w[1] for w in lines[k])):
        words = sorted(lines[key])
        cells, cur = [], [words[0]]
        for prev, nxt in zip(words, words[1:]):
            if nxt[0] - (prev[0] + prev[2]) > gap_limit:
                cells.append(cur)
                cur = [nxt]
            else:
                cur.append(nxt)
        cells.append(cur)
        row_cells.append(cells)

    # Need a repeated multi-column structure to call it a table.
    if sum(len(c) >= 2 for c in row_cells) < max(2, len(row_cells) // 2):
        return None

    def box_of(ws):
        x1 = min(w[0] for w in ws)
        y1 = min(w[1] for w in ws)
        x2 = max(w[0] + w[2] for w in ws)
        y2 = max(w[1] + w[3] for w in ws)
        return x1, y1, x2 - x1, y2 - y1

    lefts = [box_of(c)[0] for cells in row_cells for c in cells]
    col_centres = _cluster(lefts, 1.5 * float(np.median(heights)))

    result = []
    for r, cells in enumerate(row_cells):
        for ws in cells:
            box = box_of(ws)
            col = _nearest(col_centres, box[0])
            result.append(((col, r), box, " ".join(w[4] for w in ws)))
    return result


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def extract_tables_from_image(image_path, lang=None, allow_borderless=True):
    lang = lang or config.OCR_LANG
    image = read_image(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = to_gray(image)
    angle = estimate_skew(gray)
    if abs(angle) > 0.3:
        gray = rotate_keep_size(gray, angle, border=255)
        image = rotate_keep_size(image, angle, border=255)

    # Work at a size where lines and text are easy to see.
    longest = max(gray.shape)
    scale = 1.0
    if longest < 1600:
        scale = min(2.0, 1600.0 / longest)
        gray_s = cv2.resize(gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_CUBIC)
    elif longest > 4000:
        scale = 4000.0 / longest
        gray_s = cv2.resize(gray, None, fx=scale, fy=scale,
                            interpolation=cv2.INTER_AREA)
    else:
        gray_s = gray

    found = _bordered_table(gray_s, lang)
    mode = "bordered"
    if not found and allow_borderless:
        found = _borderless_table(gray_s, lang)
        mode = "borderless"

    annotated = image.copy()
    if not found:
        return annotated, []

    color = (0, 170, 0) if mode == "bordered" else (200, 120, 0)
    cells_data = []
    for (col, row), (x, y, w, h), text in found:
        x1, y1 = int(x / scale), int(y / scale)
        x2, y2 = int((x + w) / scale), int((y + h) / scale)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated, f"{row},{col}", (x1 + 3, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cells_data.append(((col, row), text))

    cells_data.sort(key=lambda item: (item[0][1], item[0][0]))
    return annotated, cells_data


def cells_to_grid(cells_data):
    if not cells_data:
        raise ValueError(
            "No table cells were detected. Try a clearer image or "
            "one with more visible grid lines.")
    max_row = max(p[1] for p, _ in cells_data)
    max_col = max(p[0] for p, _ in cells_data)
    rows = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for (col, row), text in cells_data:
        rows[row][col] = text
    return rows


def cells_to_dataframe(cells_data):
    return pd.DataFrame(cells_to_grid(cells_data))


def cells_to_csv(cells_data, output_csv="table.csv"):
    df = cells_to_dataframe(cells_data)
    df.to_csv(output_csv, index=False, header=False, encoding="utf-8-sig")
    return df


def cells_to_xlsx(cells_data, output_xlsx="table.xlsx"):
    df = cells_to_dataframe(cells_data)
    df.to_excel(output_xlsx, index=False, header=False)
    return df


def cells_to_file(cells_data, path):
    """Save as .xlsx or .csv depending on the extension."""
    if path.lower().endswith(".xlsx"):
        return cells_to_xlsx(cells_data, path)
    return cells_to_csv(cells_data, path)
