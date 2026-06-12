"""Run Torch MiDaS inference on a folder of images.

This file is a small, handoff-friendly wrapper around the original MiDaS
`run.py` logic. The model implementation itself is kept in `midas/`.
"""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path

import cv2
import numpy as np
import torch

import utils
from midas.model_loader import DEFAULT_MODEL_PATH, MODEL_TYPE, load_model


ROOT = Path(__file__).resolve().parent

IMAGE_EXTENSIONS = (".bmp", ".dib", ".jpeg", ".jpg", ".jpe", ".jp2", ".png", ".webp", ".tif", ".tiff")

first_execution = True


def resolve_path(path_text: str) -> Path:
    """Resolve a CLI path relative to this file's folder."""
    path = Path(path_text)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def process(device, model, image, input_size, target_size, optimize, use_camera=False):
    """Run the MiDaS forward pass and interpolate to the original image size.

    This is intentionally the same Torch path as the author's `run.py`:
    numpy CHW image -> torch tensor -> model.forward -> bicubic interpolation.
    """
    global first_execution

    sample = torch.from_numpy(image).to(device).unsqueeze(0)

    if optimize and device == torch.device("cuda"):
        if first_execution:
            print(
                "  Optimization to half-floats activated. Use with caution, because models like Swin require\n"
                "  float precision to work properly and may yield non-finite depth values to some extent for\n"
                "  half-floats."
            )
        sample = sample.to(memory_format=torch.channels_last)
        sample = sample.half()

    if first_execution or not use_camera:
        height, width = sample.shape[2:]
        print(f"    Input resized to {width}x{height} before entering the encoder")
        first_execution = False

    prediction = model.forward(sample)
    prediction = (
        torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=target_size[::-1],
            mode="bicubic",
            align_corners=False,
        )
        .squeeze()
        .cpu()
        .numpy()
    )

    return prediction


def create_side_by_side(image, depth, grayscale):
    """Create the same RGB/depth side-by-side visualization as the original script."""
    depth_min = depth.min()
    depth_max = depth.max()
    normalized_depth = 255 * (depth - depth_min) / (depth_max - depth_min)
    normalized_depth *= 3

    right_side = np.repeat(np.expand_dims(normalized_depth, 2), 3, axis=2) / 3
    if not grayscale:
        right_side = cv2.applyColorMap(np.uint8(right_side), cv2.COLORMAP_INFERNO)

    if image is None:
        return right_side
    return np.concatenate((image, right_side), axis=1)


def iter_images(input_path: Path) -> list[Path]:
    image_names = [Path(name) for name in glob.glob(str(input_path / "*"))]
    return sorted(path for path in image_names if path.suffix.lower() in IMAGE_EXTENSIONS)


def run(
    input_path: Path,
    output_path: Path,
    model_path: Path,
    optimize: bool = False,
    side: bool = False,
    height: int | None = None,
    square: bool = False,
    grayscale: bool = False,
    device_name: str = "auto",
) -> None:
    """Compute MiDaS relative depth maps for all images in `input_path`."""
    print("Initialize")

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    print(f"Device: {device}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"Cannot find weights: {model_path}\n"
            "Run `python download_weights.py` from MyCode, or pass --model_weights to an existing .pt file."
        )

    model, transform, net_w, net_h = load_model(device, str(model_path), optimize, height, square)

    image_names = iter_images(input_path)
    if not image_names:
        raise FileNotFoundError(f"No input images found in: {input_path}")

    os.makedirs(output_path, exist_ok=True)

    print("Start processing")
    for index, image_name in enumerate(image_names):
        print(f"  Processing {image_name} ({index + 1}/{len(image_names)})")

        original_image_rgb = utils.read_image(str(image_name))
        image = transform({"image": original_image_rgb})["image"]

        with torch.no_grad():
            prediction = process(
                device,
                model,
                image,
                (net_w, net_h),
                original_image_rgb.shape[1::-1],
                optimize,
                False,
            )

        filename = output_path / f"{image_name.stem}-{MODEL_TYPE}"
        if not side:
            utils.write_depth(str(filename), prediction, grayscale, bits=2)
        else:
            original_image_bgr = np.flip(original_image_rgb, 2)
            content = create_side_by_side(original_image_bgr * 255, prediction, grayscale)
            cv2.imwrite(str(filename) + ".png", content)

        utils.write_pfm(str(filename) + ".pfm", prediction.astype(np.float32))

    print("Finished")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Torch MiDaS inference for handoff code.")
    parser.add_argument("--input_path", default="data", help="Folder with input images, relative to MyCode by default.")
    parser.add_argument("--output_path", default="output", help="Folder for depth outputs, relative to MyCode by default.")
    parser.add_argument(
        "--model_weights",
        default=None,
        help="Path to midas_v21_small_256.pt. Defaults to ../weights/midas_v21_small_256.pt relative to MyCode.",
    )
    parser.add_argument("--side", action="store_true", help="Write RGB and depth visualization side by side.")
    parser.add_argument("--optimize", action="store_true", help="Use half-float optimization on CUDA.")
    parser.add_argument(
        "--height",
        type=int,
        default=None,
        help="Preferred encoder input height. Leave unset to use the author's model default.",
    )
    parser.add_argument("--square", action="store_true", help="Resize to square encoder input when supported.")
    parser.add_argument("--grayscale", action="store_true", help="Write grayscale depth PNG instead of inferno colormap.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    model_weights = args.model_weights or DEFAULT_MODEL_PATH

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    run(
        input_path=resolve_path(args.input_path),
        output_path=resolve_path(args.output_path),
        model_path=resolve_path(model_weights),
        optimize=args.optimize,
        side=args.side,
        height=args.height,
        square=args.square,
        grayscale=args.grayscale,
        device_name=args.device,
    )


if __name__ == "__main__":
    main()
