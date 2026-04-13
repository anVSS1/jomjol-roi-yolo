from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def rotate_keep_bounds(image, angle_degrees: float):
    """Rotate an image around its center while expanding canvas to avoid cropping."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)

    rotation = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    abs_cos = abs(rotation[0, 0])
    abs_sin = abs(rotation[0, 1])

    new_w = int((h * abs_sin) + (w * abs_cos))
    new_h = int((h * abs_cos) + (w * abs_sin))

    rotation[0, 2] += (new_w / 2) - center[0]
    rotation[1, 2] += (new_h / 2) - center[1]

    return cv2.warpAffine(image, rotation, (new_w, new_h), borderMode=cv2.BORDER_REPLICATE)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rotate first N images in a folder by given angles and save beside originals."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=Path("test maroc rotation"),
        help="Folder containing input images.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=4,
        help="Maximum number of input images to process.",
    )
    parser.add_argument(
        "--angles",
        type=float,
        nargs="+",
        default=[10, 30, 45],
        help="Rotation angles in degrees.",
    )
    args = parser.parse_args()

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    folder = args.folder

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    images = sorted([p for p in folder.iterdir() if p.suffix.lower() in image_exts])

    if not images:
        raise FileNotFoundError(f"No images found in folder: {folder}")

    selected = images[: args.num_images]
    if len(selected) < args.num_images:
        print(
            f"Warning: requested {args.num_images} images but found {len(selected)}. Processing available images only."
        )

    for img_path in selected:
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Skipping unreadable image: {img_path.name}")
            continue

        stem = img_path.stem
        suffix = img_path.suffix

        for angle in args.angles:
            rotated = rotate_keep_bounds(image, angle)
            angle_tag = int(angle) if float(angle).is_integer() else str(angle).replace(".", "p")
            out_path = folder / f"{stem}_rot{angle_tag}{suffix}"
            cv2.imwrite(str(out_path), rotated)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
