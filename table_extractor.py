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
from image_utils import (
    ensure_dark_on_light, estimate_skew, needs_inversion, read_image,
    rotate_keep_size, to_gray,
)

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
    if gray_cell.size == 0 or gray_cell.std() < 8:
        return ""  # empty / flat-coloured cell

    # White text on a dark fill (typical header cells) -> dark on light.
    if needs_inversion(gray_cell):
        gray_cell = cv2.bitwise_not(gray_cell)
    gray_cell = cv2.normalize(gray_cell, None, 0, 255, cv2.NORM_MINMAX)

    # Skip cells with almost no ink.
    otsu = cv2.threshold(gray_cell, 0, 255,
                         cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if np.count_nonzero(otsu) < 0.004 * otsu.size:
        return ""

    h, w = gray_cell.shape
    # One resampling step only: make the cell ~110 px high (x1 .. x4).
    scale = float(np.clip(110.0 / max(h, 1), 1.0, 4.0))
    if scale > 1.05:
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
def _grid_mask(gray):
    """Binary mask (255 on ruling lines) or None when there are none."""
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
    # A table needs at least two long ruling lines in some direction;
    # stray letter strokes must not be mistaken for a grid.
    def long_lines(mask, axis_len, min_len):
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        return sum(1 for i in range(1, n)
                   if stats[i, axis_len] >= min_len)

    n_h = long_lines(horizontal, cv2.CC_STAT_WIDTH, 0.12 * w)
    n_v = long_lines(vertical, cv2.CC_STAT_HEIGHT, 0.12 * h)
    if n_h < 2 and n_v < 2:
        return None

    return cv2.dilate(cv2.bitwise_or(horizontal, vertical),
                      np.ones((3, 3), np.uint8))


def _detect_cells(grid):
    """Return list of (x, y, w, h) boxes of enclosed cells, or []."""
    sealed = grid.copy()
    # Many tables have no outer border (e.g. no left edge). Draw the outline
    # of everything that was detected so edge cells are closed, instead of
    # leaking into the background and being thrown away.
    ys, xs = np.where(grid > 0)
    cv2.rectangle(sealed, (int(xs.min()), int(ys.min())),
                  (int(xs.max()), int(ys.max())), 255, 3)

    # Frame the mask so cells touching the image edge are still closed.
    framed = cv2.copyMakeBorder(sealed, 3, 3, 3, 3,
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


def _crop_full(gray_full, box, scale, pad):
    """Crop `box` (given in scaled coordinates) from the full-resolution
    image, shrunk by `pad` scaled pixels on every side."""
    x, y, w, h = box
    x1, y1 = int((x + pad) / scale), int((y + pad) / scale)
    x2, y2 = int((x + w - pad) / scale), int((y + h - pad) / scale)
    return gray_full[max(0, y1):max(y1, y2), max(0, x1):max(x1, x2)]


def _bordered_table(gray, gray_full, scale, lang, grid):
    boxes = _detect_cells(grid) if grid is not None else []
    if not boxes:
        return None
    positions = _assign_grid(boxes)

    # Crop from the original-resolution image (one resampling only) and
    # keep clear of the ruling lines.
    crops = [_crop_full(gray_full, b, scale, 5) for b in boxes]

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
def _borderless_table(gray, gray_full, scale, lang):
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
             data["width"][i], data["height"][i], text, conf))
    if len(lines) < 2:
        return None

    heights = [w[3] for ws in lines.values() for w in ws]
    med_h = float(np.median(heights))
    # Word gaps inside a cell are ~0.3 x text height, gaps between columns
    # are larger. (Splitting each text line on a big gap, as before, merged
    # neighbouring columns of tightly packed tables - e.g. "Year" and
    # "Gross" - so whole columns went missing.)
    gap_min = 0.45 * med_h

    ordered = sorted(lines, key=lambda k: min(w[1] for w in lines[k]))
    line_words = [sorted(lines[k]) for k in ordered]

    def has_gap(ws):
        return any(n[0] - (p[0] + p[2]) >= gap_min for p, n in zip(ws, ws[1:]))

    # Only lines that show a column gap vote on where the columns are, so a
    # title or a paragraph above the table cannot glue the columns together.
    gap_lines = [ws for ws in line_words if has_gap(ws)]
    if len(gap_lines) < max(2, len(line_words) // 2):
        return None

    def n_cells(ws):
        return 1 + sum(n[0] - (p[0] + p[2]) >= gap_min
                       for p, n in zip(ws, ws[1:]))

    # Headers are often wider than their data and overlap the neighbouring
    # column, so the columns are measured on the typical data rows only
    # (the lines with the most common number of cells). Other lines are
    # assigned to the nearest column afterwards.
    counts = [n_cells(ws) for ws in gap_lines]
    mode = max(set(counts), key=lambda c: (counts.count(c), c))
    voting = [ws for ws, c in zip(gap_lines, counts) if c == mode]
    if len(voting) < 2:
        voting = gap_lines

    def box_of(ws):
        x1 = min(w[0] for w in ws)
        y1 = min(w[1] for w in ws)
        x2 = max(w[0] + w[2] for w in ws)
        y2 = max(w[1] + w[3] for w in ws)
        return x1, y1, x2 - x1, y2 - y1

    # Columns = x-ranges covered by words of the voting lines (union of the
    # word spans, bridged across ordinary word gaps). Works for left-,
    # right- and centre-aligned columns alike.
    half = gap_min / 2.0
    spans = sorted((w[0] - half, w[0] + w[2] + half)
                   for ws in voting for w in ws)
    columns = []
    for s_, e_ in spans:
        if columns and s_ < columns[-1][1]:
            columns[-1][1] = max(columns[-1][1], e_)
        else:
            columns.append([s_, e_])
    columns = [[s_ + half, e_ - half] for s_, e_ in columns]
    if len(columns) < 2:
        return None

    def column_of(box):
        """Column with the largest horizontal overlap with `box`; if none
        overlaps, the nearest one."""
        x1, x2 = box[0], box[0] + box[2]
        overlaps = [min(x2, e_) - max(x1, s_) for s_, e_ in columns]
        if max(overlaps) > 0:
            return int(np.argmax(overlaps))
        return int(np.argmin([min(abs(x1 - e_), abs(x2 - s_))
                              for s_, e_ in columns]))

    def split_cells(ws):
        cells, cur = [], [ws[0]]
        for p, n in zip(ws, ws[1:]):
            if n[0] - (p[0] + p[2]) >= gap_min:
                cells.append(cur)
                cur = [n]
            else:
                cur.append(n)
        cells.append(cur)
        return cells

    result = []                       # (pos, box, text, confidence)
    for r, ws in enumerate(line_words):
        by_col = {}
        # whole cells (not single words) are assigned, so "New Moon" can
        # never be torn apart by a neighbouring column.
        for cell in split_cells(ws):
            by_col.setdefault(column_of(box_of(cell)), []).extend(cell)
        for c, cws in by_col.items():
            cws = sorted(cws)
            result.append(((c, r), box_of(cws),
                           " ".join(w[4] for w in cws),
                           float(np.mean([w[5] for w in cws]))))

    # Geometry comes from the layout analysis; weak cells are read again on
    # their own, and slots where a word was missed are filled in.
    row_span = [(min(w[1] for w in ws), max(w[1] + w[3] for w in ws))
                for ws in line_words]

    jobs = []                         # (kind, pos, box, old_text)
    for pos, box, text, conf in result:
        if conf < 60:
            jobs.append(("redo", pos, box, text))
    present = {pos for pos, _, _, _ in result}
    for r, (y1, y2) in enumerate(row_span):
        for c, (s, e) in enumerate(columns):
            if (c, r) not in present:
                jobs.append(("fill", (c, r),
                             (int(s), int(y1), int(e - s), int(y2 - y1)), ""))

    def work(job):
        kind, pos, (x, y, w, h), old = job
        m = max(4, int(0.3 * h))
        crop = _crop_full(gray_full, (x, y, w, h), scale, -m)
        return kind, pos, (x, y, w, h), _ocr_cell(crop, lang) or old

    workers = min(8, (os.cpu_count() or 2))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        done = list(pool.map(work, jobs))

    final = {pos: (box, text) for pos, box, text, _ in result}
    for kind, pos, box, text in done:
        if text:
            final[pos] = (box, text)
    return [(pos, box, text) for pos, (box, text) in final.items()]


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def extract_tables_from_image(image_path, lang=None, allow_borderless=True):
    lang = lang or config.OCR_LANG
    image = read_image(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = to_gray(ensure_dark_on_light(image)[0])
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

    grid = _grid_mask(gray_s)
    found = _bordered_table(gray_s, gray, scale, lang, grid)
    mode = "bordered"

    def n_cols(res):
        return max(pos[0] for pos, _, _ in res) + 1

    # Tables that only have row lines (or none at all) give <= 2 "columns"
    # from the ruling lines; try the text-layout analysis on the page with
    # the lines removed and keep whichever finds more columns.
    if allow_borderless and (not found or n_cols(found) <= 2):
        clean = gray_s.copy()
        if grid is not None:
            clean[cv2.dilate(grid, np.ones((3, 3), np.uint8)) > 0] = 255
        alt = _borderless_table(clean, gray, scale, lang)
        if alt and (not found or n_cols(alt) > n_cols(found)):
            found, mode = alt, "borderless"

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
