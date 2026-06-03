#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageFile
from tqdm import tqdm
import openslide
import random
from collections import defaultdict

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

ImageFile.LOAD_TRUNCATED_IMAGES = True


DEFAULT_PROMPT_BANKS: Dict[str, Dict[str, List[str]]] = {
    "tissue_parotid_v1": {
        "tumor": [
            "neoplastic salivary gland epithelial proliferation in a histopathology image",
            "tumor-forming epithelial nests or sheets in a salivary gland pathology patch",
            "atypical neoplastic salivary gland epithelial region in an H&E tile",
            "infiltrative or expansile salivary gland tumor epithelium in histology",
            "oncocytic or neoplastic epithelial tumor component in a salivary gland lesion"
        ],
        "stroma": [
            "fibrous supporting stroma in a salivary gland histopathology image",
            "collagenous connective tissue stroma between salivary gland structures",
            "myxoid or chondromyxoid stromal matrix in a salivary gland lesion",
            "hypocellular stromal background in salivary gland histology",
            "fibrotic or myxoid salivary gland stromal tissue in an H&E patch"
        ],
        "immune": [
            "dense lymphoid aggregate in a salivary gland histopathology image",
            "lymphocyte-rich inflammatory infiltrate in a salivary gland pathology patch",
            "compact inflammatory cell cluster in salivary gland histology",
            "prominent lymphoid stroma in a salivary gland lesion",
            "dense small round immune cells in an H&E salivary gland tile"
        ],
        "necrosis": [
            "necrotic tissue and cell debris in a salivary gland histopathology image",
            "coagulative necrosis in a salivary gland pathology patch",
            "necrotic tumor-associated region in salivary gland histology",
            "cell death with necrotic debris in an H&E salivary gland tile"
        ],
        "normal_epithelium": [
            "normal salivary gland acini and ducts in a histopathology image",
            "benign serous acinar epithelium in a parotid gland pathology patch",
            "organized salivary gland ductal and acinar structures in histology",
            "non-neoplastic salivary gland parenchyma with acini and ducts in an H&E tile",
            "benign orderly salivary gland epithelial structures in histology"
        ],
        "background": [
            "empty slide background with no tissue in a pathology patch",
            "whitespace or blank background in a histopathology image",
            "tissue edge artifact or low-information blank region in histology",
            "out-of-focus or nearly empty background area in a pathology tile"
        ],
        "ambiguous": [
            "mixed epithelial and stromal salivary gland lesion in a histopathology image",
            "epithelial stromal interface in a salivary gland pathology patch",
            "heterogeneous salivary gland tumor region with mixed tissue patterns",
            "oncocytic epithelial and lymphoid mixed region in salivary gland histology",
            "indeterminate mixed salivary gland tissue region in an H&E tile"
        ]
    },
    "tissue_kich_v1": {
        "tumor": [
            "viable chromophobe renal cell carcinoma with large polygonal tumor cells in a kidney histopathology image",
            "sheets of chromophobe renal carcinoma cells with distinct plant-like cell borders in an H&E kidney tile",
            "renal tumor cells with pale reticular cytoplasm, perinuclear halos, and intact wrinkled nuclei",
            "chromophobe renal cell carcinoma arranged in solid or trabecular epithelial sheets",
            "viable renal epithelial tumor with abundant pale to eosinophilic cytoplasm and raisinoid nuclei",
            "chromophobe renal carcinoma cells with prominent cell membranes and granular eosinophilic cytoplasm"
        ],
        "stroma": [
            "fibrovascular stroma separating nests of renal carcinoma cells in a kidney histopathology image",
            "thin collagenous septa and small blood vessels between chromophobe renal tumor sheets",
            "renal tumor capsule or fibrotic stromal band adjacent to chromophobe carcinoma",
            "hyalinized fibrous stroma around renal epithelial tumor nests in an H&E patch",
            "connective tissue stroma with spindle fibroblasts and vessels within a kidney tumor tile",
            "non-epithelial stromal tissue between viable renal carcinoma cell groups"
        ],
        "immune": [
            "dense lymphocytic inflammatory infiltrate in a renal tumor histopathology image",
            "compact cluster of small round immune cells adjacent to chromophobe renal carcinoma",
            "lymphocyte-rich inflammatory focus in a kidney tumor pathology patch",
            "tumor-associated lymphoid aggregate within renal carcinoma histology",
            "dense mononuclear inflammatory cells without renal epithelial tumor morphology"
        ],
        "necrosis": [
            "definite coagulative tumor necrosis with ghost cell outlines in renal carcinoma histology",
            "acellular necrotic renal tumor area with karyorrhectic nuclear debris and absent viable nuclei",
            "geographic necrosis with complete loss of intact chromophobe tumor cell borders",
            "dead tumor tissue with granular eosinophilic debris and no viable renal epithelial nuclei",
            "necrotic kidney carcinoma region lacking viable tumor cells, tubules, vessels, or stromal nuclei",
            "large acellular necrotic center with ghost architecture and nuclear debris in an H&E renal tumor tile"
        ],
        "normal_epithelium": [
            "benign renal tubules lined by orderly epithelial cells in non-neoplastic kidney parenchyma",
            "normal kidney cortex with renal tubules and glomeruli in an H&E pathology patch",
            "non-neoplastic renal tubular epithelium with preserved tubular lumens and bland nuclei",
            "benign distal renal tubules and collecting ducts with organized epithelial lining",
            "normal renal parenchyma containing glomeruli, tubules, interstitium, and small vessels"
        ],
        "background": [
            "empty slide background with no tissue in a pathology patch",
            "whitespace or blank background in a histopathology image",
            "out-of-focus or nearly empty background area in an H&E renal tile",
            "fold artifact, blur, pen mark, or non-diagnostic tissue edge in kidney histology",
            "blood, hemorrhage, crush artifact, or staining artifact without diagnostic renal tumor cells",
            "low-information tissue fragment without intact epithelial, stromal, or inflammatory morphology"
        ],
        "ambiguous": [
            "partially degenerated chromophobe renal carcinoma that is not definite viable tumor or definite necrosis",
            "pale eosinophilic renal tumor area with uncertain viability and preserved partial cell outlines",
            "oncocytic eosinophilic renal epithelial region that is difficult to distinguish from chromophobe carcinoma",
            "mixed kidney tumor region containing viable tumor cells, degeneration, stroma, inflammation, or artifact",
            "renal tumor-normal interface with carcinoma cells, benign tubules, stroma, and inflammatory cells",
            "low-quality renal tumor tissue with uncertain histologic content and ambiguous diagnostic category",
            "indeterminate renal pathology patch not confidently assigned to tumor, stroma, immune, necrosis, normal epithelium, or background"
        ]
    },
    "tissue_kich_v2": {
        "tumor": [
            "viable chromophobe renal cell carcinoma composed of large polygonal epithelial tumor cells",
            "solid sheets of chromophobe renal carcinoma cells with distinct cell borders and intact nuclei",
            "viable renal epithelial tumor cells with pale reticular cytoplasm, perinuclear halos, and wrinkled nuclei",
            "chromophobe renal cell carcinoma with cohesive nests of polygonal tumor cells and preserved cell membranes",
            "cellular chromophobe renal carcinoma area dominated by viable tumor epithelium rather than stroma or necrosis",
            "highly cellular viable kidney carcinoma patch with abundant chromophobe tumor cells and minimal background"
        ],
        "fibrovascular_stroma": [
            "fibrovascular connective tissue stroma separating renal carcinoma nests",
            "collagenous fibrous septa with spindle stromal cells between kidney tumor cell groups",
            "hyalinized fibrotic stroma adjacent to renal epithelial tumor nests",
            "renal tumor-associated fibrous stroma with collagen, fibroblasts, and small vessels",
            "non-epithelial stromal tissue around chromophobe renal carcinoma cells",
            "fibrous capsule or stromal band with spindle cells and collagen in a kidney tumor section"
        ],
        "normal_kidney_parenchyma": [
            "normal non-neoplastic kidney parenchyma in an H&E histopathology image",
            "benign renal cortex with organized renal tubules and glomeruli in a kidney pathology patch",
            "normal kidney tissue containing renal tubules, glomeruli, interstitium, and small blood vessels",
            "non-neoplastic renal tubular epithelium with preserved tubular lumens and bland nuclei",
            "benign renal tubules lined by orderly cuboidal epithelial cells in an H&E kidney tile",
            "normal renal medulla with collecting ducts and bland tubular epithelial lining",
            "adjacent normal kidney parenchyma without renal carcinoma cells in a histology patch",
            "benign kidney parenchyma with preserved tubulointerstitial architecture and no tumor"
        ],
        "vascular_hemorrhage": [
            "blood-filled vascular lumen or hemorrhage in a kidney tumor histopathology patch",
            "red blood cells and vascular space without dominant renal tumor epithelium",
            "hemorrhagic area with erythrocytes, fibrin, or blood clot in renal tumor tissue",
            "small vessel, capillary, or blood-filled space within a renal carcinoma section",
            "patch dominated by red blood cells or vascular tissue rather than tumor cells",
            "vascular or hemorrhagic tissue fragment in an H&E kidney tumor tile"
        ],
        "background_artifact": [
            "empty white slide background with no tissue in a pathology patch",
            "nearly blank or low-information background area in an H&E kidney histology tile",
            "out-of-focus, blurred, folded, crushed, or poorly stained non-diagnostic tissue",
            "tissue edge artifact with mostly whitespace and little diagnostic histology",
            "pen mark, staining artifact, torn tissue, or non-diagnostic artifact in a pathology patch",
            "low-quality kidney tissue patch without reliable tumor, stroma, or necrosis morphology"
        ],
        "ambiguous_mixed": [
            "mixed renal tumor and stromal interface without a dominant histologic component",
            "partially viable chromophobe renal carcinoma adjacent to degeneration, stroma, or artifact",
            "pale eosinophilic renal tumor area with uncertain viability but not definite necrosis",
            "renal tumor patch containing tumor cells, vessels, stroma, blood, and background together",
            "indeterminate kidney tumor tissue that is not confidently pure tumor, stroma, necrosis, hemorrhage, or artifact",
            "small fragmented tissue region with mixed epithelial cells, stromal tissue, blood, or empty space"
        ]
    },
    "tissue_camelyon17_v1": {
        "tumor_metastasis": [
            "metastatic breast carcinoma cells in a lymph node histopathology image",
            "breast cancer metastasis within lymph node tissue in an H&E pathology patch",
            "cohesive epithelial tumor nests replacing lymphoid tissue in a sentinel lymph node",
            "clusters or sheets of malignant breast carcinoma cells in a lymph node section",
            "viable metastatic carcinoma with atypical epithelial cells in lymph node histology",
            "tumor embolus or metastatic epithelial cell cluster in lymph node parenchyma",
            "macrometastatic or micrometastatic breast cancer focus in an H&E lymph node tile",
            "glandular or solid metastatic breast carcinoma cells among lymphocytes",
            "cytologically atypical epithelial tumor cells contrasting with lymphoid background"
        ],
        "lymphoid_tissue": [
            "benign lymph node parenchyma with dense small lymphocytes in an H&E histology image",
            "normal lymphoid tissue with lymphocyte-rich cortex or paracortex in a lymph node patch",
            "reactive lymphoid follicles and germinal centers without metastatic carcinoma",
            "benign lymph node tissue composed of small round lymphocytes and stromal meshwork",
            "dense lymphocytic tissue in a sentinel lymph node section without epithelial tumor cells",
            "normal or reactive lymph node architecture with lymphoid follicles and sinuses",
            "lymphocyte-rich background tissue in an H&E lymph node tile",
            "benign lymphoid aggregate without cohesive malignant epithelial nests"
        ],
        "adipose_stroma": [
            "adipose tissue and fibrous capsule adjacent to lymph node in a histopathology image",
            "perinodal fat with mature adipocytes in an H&E lymph node section",
            "fibrous stromal tissue, capsule, trabeculae, or connective tissue in a lymph node patch",
            "collagenous stroma and small blood vessels without metastatic carcinoma cells",
            "fatty tissue or fibroadipose tissue surrounding lymph node parenchyma",
            "lymph node capsule or trabecular fibrous tissue with bland stromal cells",
            "non-neoplastic fibrous connective tissue and adipocytes in a sentinel lymph node tile"
        ],
        "background_artifact": [
            "empty white slide background with no tissue in a pathology patch",
            "nearly blank or low-information background area in an H&E histology tile",
            "out-of-focus, blurred, folded, crushed, or poorly stained non-diagnostic tissue",
            "tissue edge artifact with mostly whitespace and little diagnostic histology",
            "pen mark, staining artifact, torn tissue, or non-diagnostic artifact in a pathology patch",
            "blood, debris, or processing artifact without reliable lymph node or tumor morphology"
        ],
        "ambiguous_mixed": [
            "mixed lymph node tissue containing lymphocytes, stroma, fat, vessels, or possible tumor cells",
            "lymph node tumor interface with metastatic carcinoma adjacent to lymphoid tissue",
            "small suspicious epithelial cluster in lymph node tissue that is not confidently metastatic carcinoma",
            "indeterminate lymph node patch with mixed lymphoid tissue, stroma, artifact, or possible tumor",
            "low-quality lymph node tissue with uncertain tumor status in an H&E tile",
            "heterogeneous sentinel lymph node region not confidently assigned to tumor, lymphoid tissue, adipose stroma, or background"
        ]
    },
    "tissue_bracs_bt_at_v1": {
        "normal_breast_epithelium": [
            "normal breast terminal duct lobular unit with orderly ducts and lobules in an H&E histopathology image",
            "benign normal breast ducts and acini with preserved architecture and bland epithelial lining",
            "normal breast glandular epithelium with open lumina and organized duct lobular structures",
            "non-neoplastic breast parenchyma containing bland ducts, lobules, and acini",
            "well-organized normal breast epithelial structures without ductal hyperplasia or atypia",
            "bland breast ductal and lobular epithelium with low cellularity and preserved myoepithelial layer"
        ],

        "benign_proliferative_epithelium": [
            "benign breast epithelial proliferation without atypia in an H&E histopathology patch",
            "usual ductal hyperplasia with crowded but heterogeneous benign epithelial cells in a breast duct",
            "benign proliferative breast lesion with irregular epithelial crowding and slit-like ductal spaces",
            "pathological benign breast lesion with non-atypical epithelial hyperplasia",
            "expanded breast duct or lobule containing benign epithelial proliferation with mixed cell population",
            "fibrocystic or papillomatous benign breast epithelial change without cytologic atypia",
            "usual ductal hyperplasia-like breast epithelium with streaming heterogeneous cells and preserved benign appearance"
        ],

        "atypical_epithelial_lesion": [
            "atypical breast epithelial lesion suspicious for flat epithelial atypia or atypical ductal hyperplasia",
            "flat epithelial atypia with dilated terminal duct lobular units lined by monotonous atypical epithelial cells",
            "atypical ductal hyperplasia with monotonous epithelial proliferation partially involving a breast duct",
            "breast duct lined by cytologically atypical cuboidal or columnar epithelial cells",
            "low-grade atypical breast epithelial proliferation with enlarged round nuclei and architectural rigidity",
            "monotonous atypical epithelial cells forming micropapillary, rigid, or cribriform-like structures in a breast duct",
            "ductal epithelial atypia more organized and cytologically atypical than usual ductal hyperplasia"
        ],

        "fibrocollagenous_stroma": [
            "fibrocollagenous breast stroma with pink collagen and sparse spindle fibroblasts",
            "breast connective tissue stroma without dominant epithelial lesion",
            "dense collagenous stromal tissue in a breast histopathology patch",
            "fibrous or hyalinized breast stroma with few epithelial structures",
            "fibrovascular breast stromal tissue with collagen, fibroblasts, and small vessels",
            "stromal sclerosis or collagen-rich supporting tissue surrounding benign breast glands"
        ],

        "adipose_tissue": [
            "mature breast adipose tissue with large clear adipocytes in an H&E patch",
            "fat lobules composed of large empty adipocyte spaces and sparse nuclei",
            "breast fibroadipose tissue dominated by mature adipocytes",
            "adipose-rich breast tissue with minimal epithelial or stromal diagnostic content",
            "large clear fat cells occupying most of the breast histology image",
            "benign mammary fat tissue without dominant epithelial lesion"
        ],

        "background_artifact": [
            "empty white slide background with no breast tissue in a pathology patch",
            "nearly blank low-information background area in an H&E breast histology tile",
            "out-of-focus blurred folded crushed or poorly stained non-diagnostic breast tissue",
            "tissue edge artifact with mostly whitespace and little interpretable histology",
            "pen mark staining artifact torn tissue or processing artifact in a breast pathology patch",
            "blood mucus debris or fragmented tissue without reliable breast epithelial or stromal morphology"
        ],

        "ambiguous_mixed": [
            "mixed breast epithelium and stroma without a single dominant histologic component",
            "breast epithelial stromal interface with uncertain dominant tissue role",
            "heterogeneous breast lesion patch containing normal ducts, proliferative epithelium, stroma, fat, or artifact together",
            "small fragmented breast tissue region that is difficult to classify as normal, benign proliferative, atypical, stromal, adipose, or background",
            "indeterminate breast histology patch with uncertain atypia or mixed benign and atypical features",
            "low-quality or mixed breast tissue patch not confidently assigned to a pure prototype category"
        ]
    }
}


# =========================================================
# args / prompt
# =========================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Fast CONCH semantic labeling for WSI svs/tif + h5 patches")
    parser.add_argument("--svs-root", type=str, required=True)
    parser.add_argument("--h5-root", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--class-set", type=str, default="tissue_camelyon17_v1",
                        choices=sorted(DEFAULT_PROMPT_BANKS.keys()))
    parser.add_argument("--prompt-json", type=str, default=None)
    parser.add_argument("--model-name", type=str, default="conch_ViT-B-16")

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--batch-size", type=int, default=256)

    parser.add_argument(
        "--projects",
        nargs="+",
        default=None,
        help="Optional project subfolders under svs-root. If omitted, search svs-root recursively without project constraint."
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="all",
        help="Project name used in output when --projects is not specified."
    )
    parser.add_argument(
        "--wsi-suffixes",
        nargs="+",
        default=[".svs", ".tif", ".tiff"],
        help="WSI file suffixes to search. Example: --wsi-suffixes .tif .svs"
    )

    parser.add_argument("--max-slides", type=int, default=None)
    parser.add_argument("--max-patches-per-slide", type=int, default=None)

    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--patch-level", type=int, default=None)

    parser.add_argument("--temperature", type=float, default=100.0)

    parser.add_argument("--high-entropy-to-ambiguous", action="store_true")
    parser.add_argument("--entropy-threshold", type=float, default=1.5)
    parser.add_argument("--margin-threshold", type=float, default=0.08)
    parser.add_argument("--background-score-threshold", type=float, default=0.75)

    parser.add_argument("--prefilter-white", action="store_true")
    parser.add_argument("--white-threshold", type=float, default=235.0,
                        help="Mean RGB threshold above which a patch is treated as near-white")
    parser.add_argument("--save-image-features", action="store_true")
    parser.add_argument("--skip-done", action="store_true",
                        help="Skip slides whose per-slide csv already exists")
    parser.add_argument("--merge-at-end", action="store_true",
                        help="Merge all per-slide csv into one file at the end")
    return parser.parse_args()


def load_prompt_bank(args) -> Dict[str, List[str]]:
    if args.prompt_json is not None:
        with open(args.prompt_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_PROMPT_BANKS[args.class_set]


# =========================================================
# tokenization / text feature
# =========================================================
def safe_tokenize_texts(tokenizer, texts, device, max_length: int = 128):
    """
    Always return a torch.LongTensor of shape [B, L] on device.
    Compatible with:
    - HF tokenizer with batch_encode_plus
    - HF tokenizer callable returning BatchEncoding
    - tokenizers backend with encode_batch
    """
    if hasattr(tokenizer, "batch_encode_plus"):
        out = tokenizer.batch_encode_plus(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if hasattr(out, "input_ids"):
            return out.input_ids.to(device)
        if isinstance(out, dict) and "input_ids" in out:
            return out["input_ids"].to(device)
        raise KeyError("batch_encode_plus output missing input_ids")

    if callable(tokenizer):
        out = tokenizer(
            texts,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        if hasattr(out, "input_ids"):
            return out.input_ids.to(device)
        if isinstance(out, dict) and "input_ids" in out:
            return out["input_ids"].to(device)
        if torch.is_tensor(out):
            return out.to(device)
        raise TypeError(f"Callable tokenizer returned unsupported type: {type(out)}")

    if hasattr(tokenizer, "encode_batch"):
        encs = tokenizer.encode_batch(texts)
        input_ids = [e.ids for e in encs]
        return torch.tensor(input_ids, dtype=torch.long, device=device)

    raise TypeError(f"Unsupported tokenizer type: {type(tokenizer)}")


def build_text_features(model, tokenizer, prompt_bank, device):
    class_names = list(prompt_bank.keys())
    class_features = []
    for cname in class_names:
        prompts = prompt_bank[cname]
        text_tokens = safe_tokenize_texts(tokenizer, prompts, device)
        with torch.inference_mode():
            text_features = model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
            class_feature = F.normalize(text_features.mean(dim=0, keepdim=True), dim=-1)
        class_features.append(class_feature)
    return torch.cat(class_features, dim=0), class_names


# =========================================================
# scores
# =========================================================
def compute_entropy(prob: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    return -(prob * np.log(prob + eps)).sum(axis=1)


def top1_top2_margin(prob: np.ndarray) -> np.ndarray:
    if prob.shape[1] == 1:
        return np.ones((prob.shape[0],), dtype=np.float32)
    part = np.partition(prob, kth=-2, axis=1)
    top2 = part[:, -2:]
    return top2[:, 1] - top2[:, 0]


# =========================================================
# path matching
# =========================================================
def normalize_stem_for_match(path: Path) -> str:
    """
    Default matching key: file stem.

    Handles common names:
      patient_001_node_0.tif       -> patient_001_node_0
      patient_001_node_0.h5        -> patient_001_node_0
      patient_001_node_0_patches.h5 -> patient_001_node_0
    """
    stem = path.stem
    for suffix in ["_patches", ".ome"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def find_files_by_stem_multi(root_dir: str, suffixes: List[str]) -> Dict[str, str]:
    root = Path(root_dir).resolve()
    out: Dict[str, str] = {}

    suffixes = [s.lower() if s.startswith(".") else f".{s.lower()}" for s in suffixes]

    if not root.exists():
        print(f"[Warn] root not found: {root}")
        return out

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in suffixes:
            continue

        key = normalize_stem_for_match(p)
        if key in out:
            print(f"[Warn] duplicate stem key={key}")
            print(f"  keep: {out[key]}")
            print(f"  skip: {str(p)}")
            continue
        out[key] = str(p)

    print(f"[Search] found {len(out)} files under {root} with suffixes={suffixes}")
    return out


def find_files_by_stem_under_project(
    root_dir: str,
    suffixes: List[str],
    projects: List[str],
) -> Dict[str, Dict[str, str]]:
    root = Path(root_dir).resolve()
    out: Dict[str, Dict[str, str]] = {}

    suffixes = [s.lower() if s.startswith(".") else f".{s.lower()}" for s in suffixes]

    for proj in projects:
        proj_dir = root / proj
        proj_map: Dict[str, str] = {}

        if not proj_dir.exists():
            print(f"[Warn] project dir not found: {proj_dir}")
            out[proj] = proj_map
            continue

        for p in proj_dir.rglob("*"):
            if not p.is_file():
                continue
            if p.suffix.lower() not in suffixes:
                continue

            key = normalize_stem_for_match(p)
            if key in proj_map:
                print(f"[Warn] duplicate stem in project={proj}, key={key}")
                print(f"  keep: {proj_map[key]}")
                print(f"  skip: {str(p)}")
                continue
            proj_map[key] = str(p)

        out[proj] = proj_map
        print(f"[{proj}] found {len(proj_map)} WSI files under {proj_dir}")

    return out


def pair_svs_h5(
    svs_root: str,
    h5_root: str,
    projects: Optional[List[str]] = None,
    wsi_suffixes: Optional[List[str]] = None,
    project_name: str = "all",
) -> List[dict]:
    """
    Supports two modes:

    1) Project mode:
       --projects BRCA KIRC
       Search:
         svs_root/BRCA/**/*.svs|tif
         svs_root/KIRC/**/*.svs|tif
       h5 is searched globally under h5_root.

    2) Flat/global mode:
       no --projects
       Search:
         svs_root/**/*.svs|tif
         h5_root/**/*.h5
       output project = project_name
    """
    if wsi_suffixes is None:
        wsi_suffixes = [".svs", ".tif", ".tiff"]

    h5_map = find_files_by_stem_multi(h5_root, [".h5"])
    pairs: List[dict] = []

    if projects is None or len(projects) == 0:
        wsi_map = find_files_by_stem_multi(svs_root, wsi_suffixes)

        wsi_stems = set(wsi_map.keys())
        h5_stems = set(h5_map.keys())
        common_stems = sorted(wsi_stems & h5_stems)

        print(f"[Flat/global] wsi={len(wsi_stems)} h5={len(h5_stems)} common={len(common_stems)}")

        if len(common_stems) == 0:
            wsi_only = sorted(wsi_stems - h5_stems)[:20]
            h5_only = sorted(h5_stems - wsi_stems)[:20]

            if len(wsi_only) > 0:
                print("[Flat/global] WSI-only examples:")
                for x in wsi_only:
                    print(f"  {x}")

            if len(h5_only) > 0:
                print("[Flat/global] H5-only examples:")
                for x in h5_only:
                    print(f"  {x}")

        for stem in common_stems:
            pairs.append({
                "project": project_name,
                "slide_id": stem,
                "svs_path": wsi_map[stem],
                "h5_path": h5_map[stem],
            })

        print(f"[Total] matched slide pairs = {len(pairs)}")
        return pairs

    wsi_map_by_proj = find_files_by_stem_under_project(
        svs_root,
        suffixes=wsi_suffixes,
        projects=projects,
    )

    total_common = 0
    for proj in projects:
        wsi_map = wsi_map_by_proj.get(proj, {})

        wsi_stems = set(wsi_map.keys())
        h5_stems = set(h5_map.keys())
        common_stems = sorted(wsi_stems & h5_stems)

        print(f"[{proj}] wsi={len(wsi_stems)} h5={len(h5_stems)} common={len(common_stems)}")

        if len(common_stems) == 0:
            wsi_only = sorted(wsi_stems - h5_stems)[:10]
            h5_only = sorted(h5_stems - wsi_stems)[:10]

            if len(wsi_only) > 0:
                print(f"  [{proj}] WSI-only examples:")
                for x in wsi_only:
                    print(f"    {x}")

            if len(h5_only) > 0:
                print(f"  [{proj}] H5-only examples:")
                for x in h5_only:
                    print(f"    {x}")

        for stem in common_stems:
            pairs.append({
                "project": proj,
                "slide_id": stem,
                "svs_path": wsi_map[stem],
                "h5_path": h5_map[stem],
            })

        total_common += len(common_stems)

    print(f"[Total] matched slide pairs = {total_common}")
    return pairs


# =========================================================
# h5 / image IO
# =========================================================
def read_h5_coords_and_attrs(h5_path: str):
    with h5py.File(h5_path, "r") as f:
        if "coords" not in f:
            raise KeyError(f"'coords' not found in {h5_path}")
        coords = f["coords"][:]
        attrs = dict(f["coords"].attrs.items())
    return coords, attrs


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def read_patch_from_slide(slide, x: int, y: int, level: int, patch_size: int) -> Image.Image:
    return slide.read_region((x, y), level, (patch_size, patch_size)).convert("RGB")


def is_near_white_patch(pil_img: Image.Image, white_threshold: float) -> bool:
    arr = np.asarray(pil_img, dtype=np.float32)
    return float(arr.mean()) >= white_threshold


# =========================================================
# one slide process
# =========================================================
def process_one_slide(
    slide_info: dict,
    model,
    preprocess,
    text_features,
    class_names: List[str],
    device,
    args,
    slide_csv_path: str,
    slide_feat_path: Optional[str] = None,
):
    coords, attrs = read_h5_coords_and_attrs(slide_info["h5_path"])

    patch_size = args.patch_size if args.patch_size is not None else int(attrs.get("patch_size", 256))
    patch_level = args.patch_level if args.patch_level is not None else int(attrs.get("patch_level", 0))

    if args.max_patches_per_slide is not None and len(coords) > args.max_patches_per_slide:
        sel = np.linspace(0, len(coords) - 1, args.max_patches_per_slide, dtype=int)
        coords = coords[sel]

    background_idx = None
    for bg_name in ["background", "background_artifact"]:
        if bg_name in class_names:
            background_idx = class_names.index(bg_name)
            break

    ambiguous_idx = None
    for amb_name in ["ambiguous", "ambiguous_mixed"]:
        if amb_name in class_names:
            ambiguous_idx = class_names.index(amb_name)
            break

    rows = []
    feat_chunks = [] if args.save_image_features else None

    slide = openslide.OpenSlide(slide_info["svs_path"])
    autocast_enabled = bool(args.amp and device.type == "cuda")

    batch_images = []
    batch_meta = []

    def flush_batch():
        nonlocal batch_images, batch_meta, rows, feat_chunks
        if len(batch_images) == 0:
            return

        images = torch.stack(batch_images, dim=0).to(device, non_blocking=True)

        with torch.inference_mode():
            if autocast_enabled:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    image_features = model.encode_image(images, proj_contrast=True, normalize=True)
            else:
                image_features = model.encode_image(images, proj_contrast=True, normalize=True)

            image_features = F.normalize(image_features, dim=-1)
            logits = args.temperature * (image_features @ text_features.T)
            prob = torch.softmax(logits, dim=-1)

        prob_np = prob.detach().cpu().numpy().astype(np.float32)
        feat_np = image_features.detach().cpu().numpy().astype(np.float32)

        pred_idx = prob_np.argmax(axis=1)
        pred_conf = prob_np.max(axis=1)
        ent = compute_entropy(prob_np)
        margin = top1_top2_margin(prob_np)

        for i, meta in enumerate(batch_meta):
            final_idx = int(pred_idx[i])

            if args.high_entropy_to_ambiguous and ambiguous_idx is not None:
                if ent[i] >= args.entropy_threshold and margin[i] <= args.margin_threshold:
                    final_idx = ambiguous_idx

            if background_idx is not None:
                if prob_np[i, background_idx] >= args.background_score_threshold:
                    final_idx = background_idx

            row = {
                "project": slide_info["project"],
                "slide_id": slide_info["slide_id"],
                "svs_path": slide_info["svs_path"],
                "h5_path": slide_info["h5_path"],
                "coord_x": meta["x"],
                "coord_y": meta["y"],
                "coord_idx": meta["coord_idx"],
                "patch_level": patch_level,
                "patch_size": patch_size,
                "pred_label": class_names[final_idx],
                "pred_confidence": float(pred_conf[i]),
                "entropy": float(ent[i]),
                "margin_top1_top2": float(margin[i]),
                "prefilter_white": int(meta["prefilter_white"]),
            }
            for c, cname in enumerate(class_names):
                row[f"score_{cname}"] = float(prob_np[i, c])
            rows.append(row)

        if args.save_image_features:
            feat_chunks.append(feat_np)

        batch_images = []
        batch_meta = []

    coord_iter = tqdm(
        enumerate(coords),
        total=len(coords),
        desc=f"{slide_info['slide_id']}",
        leave=False,
        dynamic_ncols=True,
    )

    for coord_idx, xy in coord_iter:
        x, y = int(xy[0]), int(xy[1])
        patch = read_patch_from_slide(slide, x, y, patch_level, patch_size)

        if args.prefilter_white and is_near_white_patch(patch, args.white_threshold):
            label_name = class_names[background_idx] if background_idx is not None else "background"
            row = {
                "project": slide_info["project"],
                "slide_id": slide_info["slide_id"],
                "svs_path": slide_info["svs_path"],
                "h5_path": slide_info["h5_path"],
                "coord_x": x,
                "coord_y": y,
                "coord_idx": coord_idx,
                "patch_level": patch_level,
                "patch_size": patch_size,
                "pred_label": label_name,
                "pred_confidence": 1.0,
                "entropy": 0.0,
                "margin_top1_top2": 1.0,
                "prefilter_white": 1,
            }
            for cname in class_names:
                row[f"score_{cname}"] = 1.0 if cname == label_name else 0.0
            rows.append(row)
            continue

        batch_images.append(preprocess(patch))
        batch_meta.append({
            "x": x,
            "y": y,
            "coord_idx": coord_idx,
            "prefilter_white": 0,
        })

        if len(batch_images) >= args.batch_size:
            flush_batch()

    flush_batch()
    slide.close()

    df = pd.DataFrame(rows)
    df.to_csv(slide_csv_path, index=False)

    if args.save_image_features and feat_chunks is not None and len(feat_chunks) > 0:
        feat_array = np.concatenate(feat_chunks, axis=0)
        np.save(slide_feat_path, feat_array)

    return {
        "num_patches": int(len(df)),
        "class_counts": df["pred_label"].value_counts().to_dict(),
        "mean_confidence": float(df["pred_confidence"].mean()),
        "mean_entropy": float(df["entropy"].mean()),
    }


# =========================================================
# merge / sampling
# =========================================================
def merge_slide_csvs(slide_csv_dir: str, merged_csv_path: str):
    csv_files = sorted(Path(slide_csv_dir).glob("*.csv"))
    if not csv_files:
        return None
    dfs = [pd.read_csv(p) for p in csv_files]
    merged = pd.concat(dfs, axis=0, ignore_index=True)
    merged.to_csv(merged_csv_path, index=False)
    return merged


def balance_sample_slide_infos(slide_infos: List[dict], max_slides: Optional[int], seed: int = 42) -> List[dict]:
    """
    Evenly sample slides across projects.
    In flat/global mode all slides have the same project name, so this becomes random sampling.
    """
    if max_slides is None or max_slides >= len(slide_infos):
        return slide_infos

    rng = random.Random(seed)

    proj_to_infos = defaultdict(list)
    for s in slide_infos:
        proj_to_infos[s["project"]].append(s)

    projects = sorted(proj_to_infos.keys())
    num_projects = len(projects)
    if num_projects == 0:
        return slide_infos

    for proj in projects:
        rng.shuffle(proj_to_infos[proj])

    base = max_slides // num_projects
    rem = max_slides % num_projects

    selected = []

    for i, proj in enumerate(projects):
        need = base + (1 if i < rem else 0)
        take = min(need, len(proj_to_infos[proj]))
        selected.extend(proj_to_infos[proj][:take])
        proj_to_infos[proj] = proj_to_infos[proj][take:]

    if len(selected) < max_slides:
        leftovers = []
        for proj in projects:
            leftovers.extend(proj_to_infos[proj])
        rng.shuffle(leftovers)
        need_more = max_slides - len(selected)
        selected.extend(leftovers[:need_more])

    rng.shuffle(selected)
    return selected


# =========================================================
# main
# =========================================================
def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    slide_csv_dir = os.path.join(args.output_dir, "per_slide_csv")
    ensure_dir(slide_csv_dir)

    if args.save_image_features:
        slide_feat_dir = os.path.join(args.output_dir, "per_slide_features")
        ensure_dir(slide_feat_dir)
    else:
        slide_feat_dir = None

    slide_infos = pair_svs_h5(
        svs_root=args.svs_root,
        h5_root=args.h5_root,
        projects=args.projects,
        wsi_suffixes=args.wsi_suffixes,
        project_name=args.project_name,
    )
    if len(slide_infos) == 0:
        raise RuntimeError("No matched WSI-h5 pairs found.")

    print(f"[Before balancing] Matched slide pairs: {len(slide_infos)}")
    print(pd.Series([x["project"] for x in slide_infos]).value_counts())

    slide_infos = balance_sample_slide_infos(
        slide_infos=slide_infos,
        max_slides=args.max_slides,
        seed=42,
    )

    print(f"[After balancing] Selected slide pairs: {len(slide_infos)}")
    print(pd.Series([x["project"] for x in slide_infos]).value_counts())

    prompt_bank = load_prompt_bank(args)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model, preprocess = create_model_from_pretrained(args.model_name, checkpoint_path=args.ckpt)
    model.eval().to(device)
    tokenizer = get_tokenizer()
    text_features, class_names = build_text_features(model, tokenizer, prompt_bank, device)

    np.save(
        os.path.join(args.output_dir, "text_features.npy"),
        text_features.detach().cpu().numpy().astype(np.float32)
    )
    with open(os.path.join(args.output_dir, "class_names.json"), "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.output_dir, "prompt_bank.json"), "w", encoding="utf-8") as f:
        json.dump(prompt_bank, f, ensure_ascii=False, indent=2)

    slide_summaries = []

    processed_ok = 0
    processed_fail = 0
    processed_skip = 0

    slide_total = len(slide_infos)
    slide_pbar = tqdm(total=slide_total, desc="Slides", dynamic_ncols=True)

    for slide_info in slide_infos:
        slide_csv_path = os.path.join(slide_csv_dir, f"{slide_info['slide_id']}.csv")
        slide_feat_path = (
            os.path.join(slide_feat_dir, f"{slide_info['slide_id']}.npy")
            if slide_feat_dir is not None else None
        )

        if args.skip_done and os.path.exists(slide_csv_path):
            print(f"[Skip done] {slide_info['slide_id']}")
            processed_skip += 1
            slide_pbar.update(1)
            slide_pbar.set_postfix(ok=processed_ok, fail=processed_fail, skip=processed_skip)
            continue

        if not os.path.exists(slide_info["svs_path"]):
            print(f"[Missing WSI] {slide_info['slide_id']} -> {slide_info['svs_path']}")
            processed_fail += 1
            slide_pbar.update(1)
            slide_pbar.set_postfix(ok=processed_ok, fail=processed_fail, skip=processed_skip)
            continue

        if not os.path.exists(slide_info["h5_path"]):
            print(f"[Missing h5] {slide_info['slide_id']} -> {slide_info['h5_path']}")
            processed_fail += 1
            slide_pbar.update(1)
            slide_pbar.set_postfix(ok=processed_ok, fail=processed_fail, skip=processed_skip)
            continue

        try:
            stats = process_one_slide(
                slide_info=slide_info,
                model=model,
                preprocess=preprocess,
                text_features=text_features,
                class_names=class_names,
                device=device,
                args=args,
                slide_csv_path=slide_csv_path,
                slide_feat_path=slide_feat_path,
            )
            stats["project"] = slide_info["project"]
            stats["slide_id"] = slide_info["slide_id"]
            slide_summaries.append(stats)
            processed_ok += 1
            slide_pbar.update(1)
            slide_pbar.set_postfix(ok=processed_ok, fail=processed_fail, skip=processed_skip)

        except Exception as e:
            print(f"[Error] slide {slide_info['slide_id']} failed: {e}")
            slide_summaries.append({
                "project": slide_info["project"],
                "slide_id": slide_info["slide_id"],
                "error": str(e),
            })
            processed_fail += 1
            slide_pbar.update(1)
            slide_pbar.set_postfix(ok=processed_ok, fail=processed_fail, skip=processed_skip)

    slide_pbar.close()

    print(f"[Summary] processed_ok={processed_ok}, processed_fail={processed_fail}, processed_skip={processed_skip}")

    summary_df = pd.DataFrame(slide_summaries)
    summary_df.to_csv(os.path.join(args.output_dir, "per_slide_summary.csv"), index=False)

    run_summary = {
        "num_matched_slides": len(slide_infos),
        "num_processed_slides": int(summary_df["slide_id"].nunique()) if "slide_id" in summary_df.columns else 0,
        "settings": vars(args),
    }
    with open(os.path.join(args.output_dir, "run_summary.json"), "w", encoding="utf-8") as f:
        json.dump(run_summary, f, ensure_ascii=False, indent=2)

    if args.merge_at_end:
        merged_csv_path = os.path.join(args.output_dir, "patch_semantic_predictions.csv")
        merged = merge_slide_csvs(slide_csv_dir, merged_csv_path)
        if merged is not None:
            print(f"[Merged] {merged_csv_path}")
            print(merged["pred_label"].value_counts())
            if "project" in merged.columns:
                print(pd.crosstab(merged["project"], merged["pred_label"]))

    print("Done.")


if __name__ == "__main__":
    main()
