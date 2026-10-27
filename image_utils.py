"""Image helpers shared by the printed-text, handwriting and table code."""
import cv2
import numpy as np


def to_gray(image):
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def read_image(path):
    """cv2.imread that also works with non-ASCII Windows paths."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        image = None
    return image


def estimate_skew(gray, max_angle=10.0):
    """Estimate page skew in degrees using the projection-profile method.

    The page is rotated through a range of angles; the angle where the text
    rows are sharpest (highest variance of the row histogram) is the skew.
    This is far more stable than Hough lines on Canny edges.
    """
    h, w = gray.shape[:2]
    scale = 600.0 / max(h, w) if max(h, w) > 600 else 1.0
    small = cv2.resize(gray, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA) if scale != 1.0 else gray
    binary = cv2.threshold(small, 0, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if np.count_nonzero(binary) < 50:
        return 0.0

    sh, sw = binary.shape
    center = (sw / 2, sh / 2)

    def score(angle):
        m = cv2.getRotationMatrix2D(center, angle, 1.0)
        rot = cv2.warpAffine(binary, m, (sw, sh), flags=cv2.INTER_NEAREST)
        profile = rot.sum(axis=1, dtype=np.float64)
        return np.var(np.diff(profile))

    coarse = np.arange(-max_angle, max_angle + 0.01, 0.5)
    best = max(coarse, key=score)
    fine = np.arange(best - 0.5, best + 0.51, 0.1)
    best = max(fine, key=score)
    return float(best)


def rotate_keep_size(image, angle, border=255):
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    value = (border,) * 3 if image.ndim == 3 else border
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=value)


def normalize_background(gray):
    """Remove shadows / uneven lighting while keeping the ink dark."""
    k = max(21, (min(gray.shape[:2]) // 25) | 1)
    bg = cv2.medianBlur(cv2.dilate(gray, np.ones((7, 7), np.uint8)), k)
    diff = 255 - cv2.absdiff(gray, bg)
    return cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)


def fit_scale(gray, target=1800, max_scale=3.0, max_side=3500):
    """Upscale small images toward `target` px (capped); shrink huge ones."""
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        s = max_side / longest
    elif longest < target:
        s = min(max_scale, target / longest)
    else:
        return gray
    interp = cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA
    return cv2.resize(gray, None, fx=s, fy=s, interpolation=interp)


def needs_inversion(gray):
    """True when the background is darker than the text (light-on-dark).

    Otsu splits pixels into two classes; the text is the minority class, so
    if the *dark* class is the majority the page is light text on dark.
    """
    t, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return float(np.mean(gray <= t)) > 0.5


def page_needs_inversion(gray):
    """True only for a genuinely dark page (slides, dark-mode screenshots).

    needs_inversion() alone is NOT safe for whole photos: Otsu may split
    "lit paper" from "shadowed paper" and call the shadowed half the
    background, which flipped normal grey notebook photos and destroyed
    the text. A real light-on-dark page also has a dark MEDIAN and dark
    border, so require those as well.
    """
    small = cv2.resize(gray, None, fx=0.25, fy=0.25,
                       interpolation=cv2.INTER_AREA) \
        if max(gray.shape[:2]) > 800 else gray
    if float(np.median(small)) >= 100:
        return False
    border = np.concatenate([small[0], small[-1], small[:, 0], small[:, -1]])
    if float(np.median(border)) >= 100:
        return False
    return needs_inversion(small)


def ensure_dark_on_light(image):
    """Return (image, was_inverted) with dark text on a light background."""
    if page_needs_inversion(to_gray(image)):
        return cv2.bitwise_not(image), True
    return image, False
