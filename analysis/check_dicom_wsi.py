import os
import argparse
from pathlib import Path

import pydicom


def inspect_dicom(path):
    ds = pydicom.dcmread(path, stop_before_pixels=True)

    print("=" * 80)
    print(f"File: {path}")
    print("-" * 80)

    fields = [
        "SOPClassUID",
        "Modality",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "Rows",
        "Columns",
        "TotalPixelMatrixRows",
        "TotalPixelMatrixColumns",
        "NumberOfFrames",
        "PhotometricInterpretation",
        "ImageType",
        "SamplesPerPixel",
        "BitsAllocated",
        "TransferSyntaxUID",
    ]

    for k in fields:
        v = ds.get(k, None)
        print(f"{k}: {v}")

    sop = str(ds.get("SOPClassUID", ""))
    modality = str(ds.get("Modality", ""))

    # DICOM WSI / Slide microscopy 粗判断
    is_slide_microscopy = False
    if modality == "SM":
        is_slide_microscopy = True
    if "1.2.840.10008.5.1.4.1.1.77.1.6" in sop:
        # VL Whole Slide Microscopy Image Storage
        is_slide_microscopy = True

    print(f"\nLikely DICOM WSI / Slide Microscopy: {is_slide_microscopy}")

    return ds, is_slide_microscopy


def try_openslide(path):
    try:
        import openslide
    except Exception as e:
        print(f"\n[OpenSlide] import failed: {e}")
        return False

    try:
        slide = openslide.OpenSlide(path)
        print("\n[OpenSlide] SUCCESS")
        print(f"level_count: {slide.level_count}")
        print(f"dimensions: {slide.dimensions}")
        print(f"level_dimensions: {slide.level_dimensions}")
        slide.close()
        return True
    except Exception as e:
        print(f"\n[OpenSlide] FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser("Check whether DICOM files are WSI and whether OpenSlide can open them")
    parser.add_argument("--input", type=str, required=True,
                        help="A single .dcm file or a directory containing .dcm files")
    parser.add_argument("--max_files", type=int, default=3,
                        help="How many files to inspect if input is a directory")
    args = parser.parse_args()

    input_path = Path(args.input)

    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.rglob("*.dcm"))[:args.max_files]

    if len(files) == 0:
        raise FileNotFoundError(f"No .dcm files found in {input_path}")

    for f in files:
        inspect_dicom(str(f))
        try_openslide(str(f))
        print()


if __name__ == "__main__":
    main()