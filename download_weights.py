"""Download MiDaS weights to the sibling `../weights` folder."""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT.parent / "weights"

MODEL_URLS = {
    "dpt_beit_large_512": ["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt"],
    "dpt_beit_large_384": ["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_384.pt"],
    "dpt_beit_base_384": ["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_base_384.pt"],
    "dpt_swin2_large_384": ["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_large_384.pt"],
    "dpt_swin2_base_384": ["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_base_384.pt"],
    "dpt_swin2_tiny_256": ["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin2_tiny_256.pt"],
    "dpt_swin_large_384": ["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_swin_large_384.pt"],
    "dpt_next_vit_large_384": ["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_next_vit_large_384.pt"],
    "dpt_levit_224": ["https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_levit_224.pt"],
    "dpt_large_384": ["https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt"],
    "dpt_hybrid_384": ["https://github.com/isl-org/MiDaS/releases/download/v3/dpt_hybrid_384.pt"],
    "midas_v21_384": ["https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_384.pt"],
    "midas_v21_small_256": ["https://github.com/isl-org/MiDaS/releases/download/v2_1/midas_v21_small_256.pt"],
    "openvino_midas_v21_small_256": [
        "https://github.com/isl-org/MiDaS/releases/download/v3_1/openvino_midas_v21_small_256.xml",
        "https://github.com/isl-org/MiDaS/releases/download/v3_1/openvino_midas_v21_small_256.bin",
    ],
}


def download(url: str, output_path: Path, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        print(f"Exists, skip: {output_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".part")

    print(f"Downloading:\n  {url}\n  -> {output_path}")
    with urllib.request.urlopen(url) as response, tmp_path.open("wb") as file:
        total = int(response.headers.get("Content-Length", "0") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            file.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                sys.stdout.write(f"\r  {downloaded / 1024 / 1024:.1f}/{total / 1024 / 1024:.1f} MB ({pct:.1f}%)")
            else:
                sys.stdout.write(f"\r  {downloaded / 1024 / 1024:.1f} MB")
            sys.stdout.flush()
    print()
    tmp_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download MiDaS model weights.")
    parser.add_argument("--model_type", default="midas_v21_small_256", choices=sorted(MODEL_URLS))
    parser.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Defaults to ../weights beside MyCode.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()

    for url in MODEL_URLS[args.model_type]:
        download(url, output_dir / url.rsplit("/", 1)[-1], overwrite=args.overwrite)


if __name__ == "__main__":
    main()
