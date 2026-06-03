import os
import argparse
import openslide
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slide_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--ext", type=str, default=".svs")
    parser.add_argument("--thumb_size", type=int, default=2048)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(args.slide_dir) if f.lower().endswith(args.ext)])
    print(f"Found {len(files)} slides")

    for fname in files:
        slide_path = os.path.join(args.slide_dir, fname)
        try:
            slide = openslide.OpenSlide(slide_path)
            w, h = slide.dimensions

            scale = args.thumb_size / max(w, h)
            new_size = (max(1, int(w * scale)), max(1, int(h * scale)))

            thumb = slide.get_thumbnail(new_size).convert("RGB")
            out_path = os.path.join(args.out_dir, os.path.splitext(fname)[0] + ".jpg")
            thumb.save(out_path, quality=90)

            slide.close()
            print(f"[OK] {fname} -> {out_path}")
        except Exception as e:
            print(f"[FAIL] {fname}: {e}")

if __name__ == "__main__":
    main()