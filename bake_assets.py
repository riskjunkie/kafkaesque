#!/usr/bin/env python3
"""
Pre-bakes every image the game currently processes at runtime via
canvas.getImageData() — the exact call the browser blocks when the page is
opened as file:// instead of served over http, which is the recurring
"artwork can't load" issue.

This replicates keyOutBackground(), stripRoadLines() and extractIconSheet()
from the-castle.html pixel-for-pixel (same thresholds, same corner-sampling,
same dilate-then-label approach) and writes the RESULTS as static files, so
the live page can just load pre-processed PNGs directly with zero canvas
reads for these assets. Run this once whenever the source art in
Game Imagery/ changes; the baked outputs are committed alongside it.
"""
import json
import math
import numpy as np
from PIL import Image

SRC = "Game Imagery"
OUT = "Game Imagery/baked"
import os
os.makedirs(OUT, exist_ok=True)


def key_out_background(path, out_name):
    """Mirrors keyOutBackground(): corner-sampled chroma key with a soft
    falloff, plus the resulting content bounding box."""
    im = Image.open(path).convert("RGBA")
    arr = np.array(im).astype(np.float64)
    h, w = arr.shape[0], arr.shape[1]
    r, g, b, a0 = arr[..., 0], arr[..., 1], arr[..., 2], arr[..., 3]

    def px(x, y):
        return arr[y, x, :3]

    corners = [px(2, 2), px(w - 3, 2), px(2, h - 3), px(w - 3, h - 3)]
    bg = np.mean(corners, axis=0)

    inner, outer = 16.0, 70.0
    dist = np.sqrt((r - bg[0]) ** 2 + (g - bg[1]) ** 2 + (b - bg[2]) ** 2)
    alpha = np.clip((dist - inner) / (outer - inner) * 255.0, 0, 255)
    alpha = np.where(dist <= inner, 0.0, np.where(dist >= outer, 255.0, alpha))
    new_alpha = np.minimum(a0, alpha)

    mask = new_alpha > 40
    ys, xs = np.where(mask)
    if len(xs) == 0:
        minX, maxX, minY, maxY = 0, w, 0, h
    else:
        minX, maxX = int(xs.min()), int(xs.max())
        minY, maxY = int(ys.min()), int(ys.max())

    out = arr.copy()
    out[..., 3] = new_alpha
    Image.fromarray(out.astype(np.uint8), "RGBA").save(f"{OUT}/{out_name}.png")
    return {"w": w, "h": h, "bbox": {"minX": minX, "maxX": maxX, "minY": minY, "maxY": maxY}}


def strip_road_lines(path, out_name):
    """Mirrors stripRoadLines(): erases the two baked-in lane stripes by
    cross-fading a mirror of the clean asphalt on either side of each band."""
    im = Image.open(path).convert("RGB")
    arr = np.array(im).astype(np.float64)
    h, w = arr.shape[0], arr.shape[1]
    bands = [(0.243, 0.286), (0.716, 0.758)]
    for b0, b1 in bands:
        a = math.floor(b0 * w)
        z = math.ceil(b1 * w)
        span = z - a
        if span <= 0 or a - span < 0 or z + span > w - 1:
            continue
        for xi in range(a, z):
            t = (xi - a) / span
            li = a - 1 - (xi - a)
            ri = z + (z - xi)
            arr[:, xi, :] = arr[:, li, :] * (1 - t) + arr[:, ri, :] * t
    Image.fromarray(arr.astype(np.uint8), "RGB").save(f"{OUT}/{out_name}.png")


def dilate(mask, radius):
    """Separable box dilation, matches the JS two-pass sliding-window version."""
    h, w = mask.shape
    tmp = np.zeros_like(mask, dtype=np.int32)
    out = np.zeros_like(mask, dtype=np.uint8)
    m = mask.astype(np.int32)
    for y in range(h):
        row = m[y]
        count = 0
        acc = np.zeros(w, dtype=np.int32)
        for x in range(-radius, w):
            if x + radius < w:
                count += row[x + radius]
            if x - radius - 1 >= 0:
                count -= row[x - radius - 1]
            if x >= 0:
                acc[x] = 1 if count > 0 else 0
        tmp[y] = acc
    for x in range(w):
        col = tmp[:, x]
        count = 0
        acc = np.zeros(h, dtype=np.uint8)
        for y in range(-radius, h):
            if y + radius < h:
                count += col[y + radius]
            if y - radius - 1 >= 0:
                count -= col[y - radius - 1]
            if y >= 0:
                acc[y] = 1 if count > 0 else 0
        out[:, x] = acc
    return out


def label_components(mask):
    """4-connected flood fill labeling (no scipy available offline)."""
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=np.int32)
    next_label = 1
    for y0 in range(h):
        for x0 in range(w):
            if mask[y0, x0] and labels[y0, x0] == 0:
                lbl = next_label
                next_label += 1
                stack = [(y0, x0)]
                labels[y0, x0] = lbl
                while stack:
                    y, x = stack.pop()
                    if x > 0 and mask[y, x - 1] and labels[y, x - 1] == 0:
                        labels[y, x - 1] = lbl
                        stack.append((y, x - 1))
                    if x < w - 1 and mask[y, x + 1] and labels[y, x + 1] == 0:
                        labels[y, x + 1] = lbl
                        stack.append((y, x + 1))
                    if y > 0 and mask[y - 1, x] and labels[y - 1, x] == 0:
                        labels[y - 1, x] = lbl
                        stack.append((y - 1, x))
                    if y < h - 1 and mask[y + 1, x] and labels[y + 1, x] == 0:
                        labels[y + 1, x] = lbl
                        stack.append((y + 1, x))
    return labels, next_label - 1


def extract_icon_sheet(path, order):
    """Mirrors extractIconSheet(): black-keyed alpha, dilate-then-label blob
    detection, reading-order sort, per-icon crop with padding=5."""
    im = Image.open(path).convert("RGB")
    arr = np.array(im).astype(np.float64)
    h, w = arr.shape[0], arr.shape[1]
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    inner, outer = 14.0, 60.0
    dist = np.sqrt(r * r + g * g + b * b)
    alpha = np.clip((dist - inner) / (outer - inner) * 255.0, 0, 255)
    alpha = np.where(dist <= inner, 0.0, np.where(dist >= outer, 255.0, alpha))
    solid = alpha > 90

    dilated = dilate(solid, 8)
    labels, n_labels = label_components(dilated)

    blobs = []
    for lbl in range(1, n_labels + 1):
        ys, xs = np.where((labels == lbl) & solid)
        if len(xs) <= 500:
            continue
        blobs.append({
            "minX": int(xs.min()), "maxX": int(xs.max()),
            "minY": int(ys.min()), "maxY": int(ys.max()),
            "cy": (int(ys.min()) + int(ys.max())) / 2,
            "cx": (int(xs.min()) + int(xs.max())) / 2,
        })

    blobs.sort(key=lambda b: b["cy"])
    rows, cur, last_cy = [], [], None
    for bl in blobs:
        if last_cy is not None and bl["cy"] - last_cy > h * 0.08:
            rows.append(cur)
            cur = []
        cur.append(bl)
        last_cy = bl["cy"]
    if cur:
        rows.append(cur)
    ordered = []
    for row in rows:
        row.sort(key=lambda b: b["cx"])
        ordered.extend(row)

    print(f"  icon sheet: expected {len(order)}, found {len(ordered)}")

    rgba = np.dstack([arr, alpha]).astype(np.uint8)
    full = Image.fromarray(rgba, "RGBA")
    meta = {}
    pad = 5
    for i, bl in enumerate(ordered):
        if i >= len(order):
            break
        sx = max(0, bl["minX"] - pad)
        sy = max(0, bl["minY"] - pad)
        sw = min(w, bl["maxX"] + pad) - sx
        sh = min(h, bl["maxY"] + pad) - sy
        crop = full.crop((sx, sy, sx + sw, sy + sh))
        name = order[i]
        crop.save(f"{OUT}/icon-{name}.png")
        meta[name] = {"w": sw, "h": sh, "bbox": {"minX": pad, "maxX": sw - pad, "minY": pad, "maxY": sh - pad}}
    return meta


ICON_ORDER = ['cabinet', 'stack', 'stamp', 'deskSlow',
              'redtape', 'server', 'auditor', 'barricade',
              'subpoena', 'loophole', 'tip', 'appeal', 'refund']

if __name__ == "__main__":
    meta = {}

    print("Keying character sprites...")
    for key, fname in [
        ("roachRunA", "cockroach1.png"), ("roachRunB", "cockroach2.png"),
        ("roachJump", "cockroach3.png"), ("roachDuck", "cockroach4.png"),
        ("roachFlyA", "cockroach_fly1.png"), ("roachFlyB", "cockroach_fly2.png"),
        ("roachHitA", "hit1.png"), ("roachHitB", "hit2.png"),
        ("roachSplat", "splat.png"),
    ]:
        print(" ", fname)
        meta[key] = key_out_background(f"{SRC}/{fname}", key)

    print("Cleaning road texture...")
    strip_road_lines(f"{SRC}/background3.png", "roadTex")

    print("Extracting icon sheet...")
    icon_meta = extract_icon_sheet(f"{SRC}/icons.png", ICON_ORDER)

    with open(f"{OUT}/manifest.json", "w") as f:
        json.dump({"characters": meta, "icons": icon_meta}, f, indent=1)

    print("Done ->", OUT)
