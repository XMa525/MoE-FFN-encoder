import os
import math
import argparse

import pandas as pd
import openslide
from PIL import Image, ImageDraw, ImageFont, ImageOps
import sys
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def canonicalize_path(path: str) -> str:
    return os.path.normpath(str(path)).replace("\\", "/")


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def read_patch(svs_path, x, y, patch_level, patch_size, out_size=224):
    slide = openslide.OpenSlide(svs_path)
    try:
        patch = slide.read_region((int(x), int(y)), int(patch_level), (int(patch_size), int(patch_size))).convert("RGB")
    finally:
        slide.close()

    if out_size is not None and int(out_size) > 0:
        patch = patch.resize((out_size, out_size))
    return patch


def get_font(size=20):
    # 尽量使用系统字体；失败就退回默认
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def draw_patch_with_text(img, title_lines, border_color=(0, 0, 0), border_width=3, text_height=54):
    """
    在 patch 上方加文字说明
    """
    w, h = img.size
    canvas = Image.new("RGB", (w, h + text_height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = get_font(18)

    # 贴 patch
    canvas.paste(img, (0, text_height))

    # 边框
    draw.rectangle(
        [0, text_height, w - 1, text_height + h - 1],
        outline=border_color,
        width=border_width
    )

    y = 4
    for line in title_lines:
        draw.text((6, y), line, fill=(20, 20, 20), font=font)
        y += 22

    return canvas


def make_patch_grid(patch_items, title, patch_out_size=224, cols=5, side_name="baseline"):
    """
    patch_items: list of dict rows
    """
    title_font = get_font(28)
    small_font = get_font(18)

    patch_panels = []
    for row in patch_items:
        patch = read_patch(
            svs_path=row["svs_path"],
            x=row["coord_x"],
            y=row["coord_y"],
            patch_level=row["patch_level"],
            patch_size=row["patch_size"],
            out_size=patch_out_size,
        )

        score = float(row["score"])
        rank = int(row["rank"])
        xy_line = f"({int(row['coord_x'])}, {int(row['coord_y'])})"
        score_line = f"rank={rank}  score={score:.4f}"

        border_color = (60, 90, 170) if side_name == "baseline" else (180, 70, 70)

        panel = draw_patch_with_text(
            patch,
            title_lines=[score_line, xy_line],
            border_color=border_color,
            border_width=3,
            text_height=50,
        )
        patch_panels.append(panel)

    if len(patch_panels) == 0:
        patch_panels = [Image.new("RGB", (patch_out_size, patch_out_size + 50), (245, 245, 245))]

    n = len(patch_panels)
    cols = min(cols, n)
    rows = math.ceil(n / cols)

    panel_w, panel_h = patch_panels[0].size
    gap = 12
    title_h = 48

    grid_w = cols * panel_w + (cols - 1) * gap
    grid_h = title_h + rows * panel_h + (rows - 1) * gap

    canvas = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill=(20, 20, 20), font=title_font)

    for i, p in enumerate(patch_panels):
        r = i // cols
        c = i % cols
        x = c * (panel_w + gap)
        y = title_h + r * (panel_h + gap)
        canvas.paste(p, (x, y))

    return canvas


def stack_lr(left_img, right_img, gap=30, bg=(255, 255, 255)):
    h = max(left_img.height, right_img.height)
    w = left_img.width + gap + right_img.width
    canvas = Image.new("RGB", (w, h), bg)
    canvas.paste(left_img, (0, 0))
    canvas.paste(right_img, (left_img.width + gap, 0))
    return canvas


def add_header(img, lines, header_h=90):
    font_title = get_font(32)
    font_small = get_font(22)

    canvas = Image.new("RGB", (img.width, img.height + header_h), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    y = 8
    for i, line in enumerate(lines):
        font = font_title if i == 0 else font_small
        draw.text((12, y), line, fill=(10, 10, 10), font=font)
        y += 32

    canvas.paste(img, (0, header_h))
    return canvas


def select_slides(compare_df, num_negative=5, num_positive=5):
    """
    优先选变化最大的 slide：
    negative / positive 分开选
    """
    selected = []

    neg_df = compare_df[compare_df["slide_label"] == 0].copy()
    pos_df = compare_df[compare_df["slide_label"] == 1].copy()

    if "delta_topk_mean_score_wsi_minus_baseline" in neg_df.columns:
        neg_df = neg_df.sort_values("delta_topk_mean_score_wsi_minus_baseline", ascending=True)
    if "delta_topk_mean_score_wsi_minus_baseline" in pos_df.columns:
        pos_df = pos_df.sort_values("delta_topk_mean_score_wsi_minus_baseline", ascending=True)

    selected.extend(neg_df.head(num_negative)["slide_id"].tolist())
    selected.extend(pos_df.head(num_positive)["slide_id"].tolist())
    return selected


def build_slide_summary_line(compare_row):
    sid = compare_row["slide_id"]
    label = int(compare_row["slide_label"])

    b = compare_row["topk_mean_score_baseline"]
    w = compare_row["topk_mean_score_wsi"]
    d = compare_row["delta_topk_mean_score_wsi_minus_baseline"]

    label_name = "negative" if label == 0 else "positive"
    line1 = f"slide_id={sid}   label={label_name}"
    line2 = f"topk_mean: baseline={b:.4f}   wsi={w:.4f}   delta={d:.4f}"
    return line1, line2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-top-csv", type=str, required=True)
    parser.add_argument("--wsi-top-csv", type=str, required=True)
    parser.add_argument("--compare-csv", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)

    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--num-negative", type=int, default=5)
    parser.add_argument("--num-positive", type=int, default=5)

    args = parser.parse_args()

    safe_mkdir(args.out_dir)

    baseline_top = pd.read_csv(args.baseline_top_csv)
    wsi_top = pd.read_csv(args.wsi_top_csv)
    compare_df = pd.read_csv(args.compare_csv)

    # 规范路径
    if "svs_path" in baseline_top.columns:
        baseline_top["svs_path"] = baseline_top["svs_path"].map(canonicalize_path)
    if "svs_path" in wsi_top.columns:
        wsi_top["svs_path"] = wsi_top["svs_path"].map(canonicalize_path)

    selected_slides = select_slides(
        compare_df,
        num_negative=args.num_negative,
        num_positive=args.num_positive,
    )

    selected_csv = os.path.join(args.out_dir, "selected_slides.csv")
    compare_df[compare_df["slide_id"].isin(selected_slides)].to_csv(selected_csv, index=False)
    print(f"[Saved] {selected_csv}")

    for slide_id in selected_slides:
        cmp_row = compare_df[compare_df["slide_id"] == slide_id]
        if len(cmp_row) == 0:
            continue
        cmp_row = cmp_row.iloc[0]

        b_rows = (
            baseline_top[baseline_top["slide_id"] == slide_id]
            .sort_values("rank")
            .head(args.topk)
            .to_dict("records")
        )
        w_rows = (
            wsi_top[wsi_top["slide_id"] == slide_id]
            .sort_values("rank")
            .head(args.topk)
            .to_dict("records")
        )

        if len(b_rows) == 0 or len(w_rows) == 0:
            print(f"[Skip] slide_id={slide_id}, missing top patches in one side")
            continue

        left = make_patch_grid(
            b_rows,
            title=f"Baseline Top-{args.topk}",
            patch_out_size=args.patch_size,
            cols=args.cols,
            side_name="baseline",
        )
        right = make_patch_grid(
            w_rows,
            title=f"WSI-Bag Top-{args.topk}",
            patch_out_size=args.patch_size,
            cols=args.cols,
            side_name="wsi",
        )

        merged = stack_lr(left, right, gap=30)

        line1, line2 = build_slide_summary_line(cmp_row)
        merged = add_header(merged, [line1, line2], header_h=84)

        label_name = "neg" if int(cmp_row["slide_label"]) == 0 else "pos"
        delta = cmp_row["delta_topk_mean_score_wsi_minus_baseline"]
        save_name = f"{label_name}__{slide_id}__delta_{delta:.4f}.png".replace("/", "_")
        save_path = os.path.join(args.out_dir, save_name)
        merged.save(save_path)
        print(f"[Saved] {save_path}")


if __name__ == "__main__":
    main()