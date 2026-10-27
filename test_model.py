"""Evaluate the OCR pipeline on a labelled dataset.

dataset/img/<id>.png|jpg   images
dataset/text/<id>.txt      ground-truth text

Examples
    python test_model.py                         # printed, word images
    python test_model.py --mode handwritten
    python test_model.py --psm 7 6 --limit 200 --show
Outputs: results/summary.json, results/predictions.csv, results/confusions.png
"""
import argparse
import json
import os
from collections import Counter

import pandas as pd
from rapidfuzz.distance import Levenshtein
from tqdm import tqdm

import backend


def normalize(text, lowercase=True):
    text = " ".join(text.split())          # collapse all whitespace
    return text.lower() if lowercase else text


def evaluate(img_dir, gt_dir, mode="printed", psms=(8, 7, 6),
             limit=None, lowercase=True):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
    files = sorted(f for f in os.listdir(img_dir) if f.lower().endswith(exts))
    if limit:
        files = files[:limit]
    print(f"Images found: {len(files)}")

    rows = []
    confusions = Counter()
    tot_chars = tot_char_err = tot_words = tot_word_err = 0

    for name in tqdm(files):
        stem = os.path.splitext(name)[0]
        gt_path = os.path.join(gt_dir, stem + ".txt")
        if not os.path.exists(gt_path):
            print(f"Missing ground truth: {name}")
            continue
        with open(gt_path, encoding="utf-8") as f:
            gt = normalize(f.read(), lowercase)

        img_path = os.path.join(img_dir, name)
        conf = None
        if mode == "printed":
            _, processed, _ = backend.preprocess_image(img_path)
            if processed is None:
                print(f"Could not read: {name}")
                continue
            pred, conf, _ = backend.extract_text_with_confidence(
                processed, psms=psms)
        else:
            pred = backend.extract_handwritten_text_by_line(img_path)
            if pred == "No handwriting detected.":
                pred = ""
        pred = normalize(pred, lowercase)

        # Character errors (edit distance) - the standard CER definition.
        char_err = Levenshtein.distance(pred, gt)
        tot_chars += len(gt)
        tot_char_err += char_err

        # Word errors (edit distance over word tokens) - the standard WER.
        gt_w, pr_w = gt.split(), pred.split()
        word_err = Levenshtein.distance(pr_w, gt_w)
        tot_words += len(gt_w)
        tot_word_err += word_err

        # Which characters get confused? (from the edit operations)
        for op in Levenshtein.editops(gt, pred):
            if op.tag == "replace":
                confusions[(gt[op.src_pos], pred[op.dest_pos])] += 1
            elif op.tag == "delete":
                confusions[(gt[op.src_pos], "∅")] += 1
            else:
                confusions[("∅", pred[op.dest_pos])] += 1

        rows.append({"file": name, "ground_truth": gt, "prediction": pred,
                     "exact": pred == gt, "char_errors": char_err,
                     "cer": char_err / max(len(gt), 1),
                     "confidence": conf})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("Nothing was evaluated - check the dataset paths.")

    summary = {
        "images_evaluated": int(len(df)),
        "exact_match_accuracy_%": round(100 * df["exact"].mean(), 2),
        "CER_%": round(100 * tot_char_err / max(tot_chars, 1), 2),
        "WER_%": round(100 * tot_word_err / max(tot_words, 1), 2),
        "char_accuracy_%": round(
            100 * (1 - tot_char_err / max(tot_chars, 1)), 2),
        "avg_edit_distance": round(df["char_errors"].mean(), 3),
    }
    return df, summary, confusions


def plot_confusions(confusions, path, top=25, show=False):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not confusions:
        return
    items = confusions.most_common(top)
    labels = [f"'{a}' → '{b}'" for (a, b), _ in items][::-1]
    counts = [c for _, c in items][::-1]
    plt.figure(figsize=(8, 0.35 * len(items) + 1.5))
    plt.barh(labels, counts, color="#3B6FD4")
    plt.xlabel("Count")
    plt.title(f"Top {len(items)} character confusions (truth → predicted)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--img-dir", default="dataset/img")
    ap.add_argument("--gt-dir", default="dataset/text")
    ap.add_argument("--mode", choices=["printed", "handwritten"],
                    default="printed")
    ap.add_argument("--psm", type=int, nargs="+", default=[8, 7, 6],
                    help="Tesseract page-segmentation modes to try "
                         "(8/7 = single word/line, 6 = block, 3 = page)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-case", action="store_true",
                    help="case-sensitive comparison")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--show", action="store_true",
                    help="open the confusion chart in a window")
    args = ap.parse_args()

    df, summary, confusions = evaluate(
        args.img_dir, args.gt_dir, args.mode, tuple(args.psm),
        args.limit, not args.keep_case)

    os.makedirs(args.out_dir, exist_ok=True)
    df.to_csv(os.path.join(args.out_dir, "predictions.csv"),
              index=False, encoding="utf-8-sig")
    with open(os.path.join(args.out_dir, "summary.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    plot_confusions(confusions, os.path.join(args.out_dir, "confusions.png"),
                    show=args.show)

    print("\nOCR evaluation")
    for k, v in summary.items():
        print(f"  {k:<26}{v}")

    wrong = df[~df["exact"]]
    print(f"\n{len(wrong)} incorrect predictions (first 20):")
    for _, r in wrong.head(20).iterrows():
        print(f"  {r['file']}: predicted={r['prediction']!r} "
              f"truth={r['ground_truth']!r}")
    print(f"\nFull report saved in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
