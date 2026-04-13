from __future__ import annotations

import argparse
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
YOLO_ROTATION_RETRY_ANGLES = [10, -10, 20, -20, 30, -30, 45, -45]


@dataclass
class DetectionResult:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    method: str
    rotation_angle: float = 0.0
    rotated_box: Optional[tuple[int, int, int, int]] = None


def resolve_yolo_model_path(requested_path: str) -> Path:
    requested = Path(requested_path)
    candidates = [
        requested,
        Path("models/roi_model/yolo_model.pt"),
        Path("models/yolo_model.pt"),
        Path("yolo_model.pt"),
    ]

    seen = set()
    for candidate in candidates:
        key = str(candidate.resolve()) if candidate.exists() else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(
        "Could not find YOLO model weights. Checked:\n"
        f"{searched}\n\n"
        "Expected filename: yolo_model.pt"
    )


def recvall(sock: socket.socket, size: int) -> Optional[bytes]:
    data = bytearray()
    while len(data) < size:
        packet = sock.recv(size - len(data))
        if not packet:
            return None
        data.extend(packet)
    return bytes(data)


def recv_message(sock: socket.socket) -> Optional[bytes]:
    header = recvall(sock, 4)
    if header is None:
        return None
    body_size = struct.unpack("!I", header)[0]
    if body_size == 0:
        return b""
    return recvall(sock, body_size)


def send_message(sock: socket.socket, payload: bytes) -> None:
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def clamp_box(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    width: int,
    height: int,
) -> Optional[tuple[int, int, int, int]]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def list_images(input_path: Path) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]

    images = []
    for ext in SUPPORTED_EXTENSIONS:
        images.extend(input_path.glob(f"*{ext}"))
        images.extend(input_path.glob(f"*{ext.upper()}"))

    images = sorted(set(images))
    if not images:
        raise ValueError(f"No supported images found in folder: {input_path}")
    return images


def detect_roi_yolo(
    model: YOLO,
    image_bgr: np.ndarray,
    conf_threshold: float,
    imgsz: int,
    device: str,
) -> Optional[DetectionResult]:
    results = model.predict(
        source=image_bgr,
        conf=conf_threshold,
        imgsz=imgsz,
        device=device,
        verbose=False,
    )

    if not results:
        return None

    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None

    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
    confs = boxes.conf.detach().cpu().numpy()
    coords = boxes.xyxy.detach().cpu().numpy()

    candidate_indices = np.where(cls_ids == 0)[0]
    if len(candidate_indices) == 0:
        candidate_indices = np.arange(len(confs))

    best_idx = candidate_indices[int(np.argmax(confs[candidate_indices]))]
    x1, y1, x2, y2 = coords[best_idx]
    return DetectionResult(
        x1=int(round(x1)),
        y1=int(round(y1)),
        x2=int(round(x2)),
        y2=int(round(y2)),
        confidence=float(confs[best_idx]),
        method="yolo",
        rotation_angle=0.0,
        rotated_box=None,
    )


def rotate_keep_bounds(image_bgr: np.ndarray, angle_degrees: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_bgr.shape[:2]
    center = (w / 2.0, h / 2.0)

    rotation = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)
    abs_cos = abs(rotation[0, 0])
    abs_sin = abs(rotation[0, 1])

    new_w = int((h * abs_sin) + (w * abs_cos))
    new_h = int((h * abs_cos) + (w * abs_sin))

    rotation[0, 2] += (new_w / 2.0) - center[0]
    rotation[1, 2] += (new_h / 2.0) - center[1]

    rotated = cv2.warpAffine(
        image_bgr,
        rotation,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated, rotation


def _transform_points(points: np.ndarray, matrix_2x3: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    hom = np.hstack([points.astype(np.float32), ones])
    out = hom @ matrix_2x3.T
    return out


def map_detection_to_original(
    det_rot: DetectionResult,
    rot_matrix: np.ndarray,
    original_shape: tuple[int, int, int],
    angle_degrees: float,
) -> Optional[DetectionResult]:
    inv = cv2.invertAffineTransform(rot_matrix)

    corners = np.array(
        [
            [det_rot.x1, det_rot.y1],
            [det_rot.x2, det_rot.y1],
            [det_rot.x2, det_rot.y2],
            [det_rot.x1, det_rot.y2],
        ],
        dtype=np.float32,
    )
    mapped = _transform_points(corners, inv)

    x_vals = mapped[:, 0]
    y_vals = mapped[:, 1]

    x1 = int(np.floor(np.min(x_vals)))
    y1 = int(np.floor(np.min(y_vals)))
    x2 = int(np.ceil(np.max(x_vals)))
    y2 = int(np.ceil(np.max(y_vals)))

    h, w = original_shape[:2]
    clamped = clamp_box(x1, y1, x2, y2, width=w, height=h)
    if clamped is None:
        return None

    cx1, cy1, cx2, cy2 = clamped
    return DetectionResult(
        x1=cx1,
        y1=cy1,
        x2=cx2,
        y2=cy2,
        confidence=det_rot.confidence,
        method=f"yolo_rot{angle_degrees:g}",
        rotation_angle=float(angle_degrees),
        rotated_box=(det_rot.x1, det_rot.y1, det_rot.x2, det_rot.y2),
    )


def crop_from_rotated_detection(
    image_bgr: np.ndarray,
    rotation_angle: float,
    rotated_box: tuple[int, int, int, int],
) -> Optional[np.ndarray]:
    rotated_img, _ = rotate_keep_bounds(image_bgr, rotation_angle)
    rh, rw = rotated_img.shape[:2]
    rx1, ry1, rx2, ry2 = rotated_box
    clamped = clamp_box(rx1, ry1, rx2, ry2, width=rw, height=rh)
    if clamped is None:
        return None

    cx1, cy1, cx2, cy2 = clamped
    roi = rotated_img[cy1:cy2, cx1:cx2]
    if roi.size == 0:
        return None
    return roi


def detect_roi_yolo_with_rotation_retry(
    model: YOLO,
    image_bgr: np.ndarray,
    conf_threshold: float,
    imgsz: int,
    device: str,
    rotation_angles: list[float],
) -> Optional[DetectionResult]:
    first_try = detect_roi_yolo(
        model=model,
        image_bgr=image_bgr,
        conf_threshold=conf_threshold,
        imgsz=imgsz,
        device=device,
    )
    if first_try is not None:
        return first_try

    for angle in rotation_angles:
        rotated, rot_matrix = rotate_keep_bounds(image_bgr, angle)
        det_rot = detect_roi_yolo(
            model=model,
            image_bgr=rotated,
            conf_threshold=conf_threshold,
            imgsz=imgsz,
            device=device,
        )
        if det_rot is None:
            continue

        mapped = map_detection_to_original(
            det_rot=det_rot,
            rot_matrix=rot_matrix,
            original_shape=image_bgr.shape,
            angle_degrees=angle,
        )
        if mapped is not None:
            return mapped

    return None


def _find_best_horizontal_rect(mask: np.ndarray, image_shape: tuple[int, int, int]) -> Optional[DetectionResult]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_shape[:2]
    image_area = float(h * w)

    best_score = -1.0
    best_rect: Optional[DetectionResult] = None

    for cnt in contours:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh <= 0:
            continue

        aspect = bw / float(bh)
        area = float(bw * bh)

        if aspect <= 2.5:
            continue
        if bw < 0.15 * w:
            continue
        if bh < 0.05 * h:
            continue

        score = area * min(aspect, 8.0)
        if score > best_score:
            best_score = score
            pseudo_conf = min(0.99, area / max(1.0, image_area))
            best_rect = DetectionResult(
                x1=x,
                y1=y,
                x2=x + bw,
                y2=y + bh,
                confidence=float(pseudo_conf),
                method="opencv_fallback",
            )

    return best_rect


def detect_roi_fallback(image_bgr: np.ndarray) -> Optional[DetectionResult]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, thresh = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)),
        iterations=2,
    )

    best = _find_best_horizontal_rect(thresh, image_bgr.shape)
    if best is not None:
        return best

    edges = cv2.Canny(blur, 60, 160)
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)
    return _find_best_horizontal_rect(edges, image_bgr.shape)


def send_roi_and_get_reading(sock: socket.socket, roi_bgr: np.ndarray) -> str:
    ok, enc = cv2.imencode(".jpg", roi_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        return "ENCODE_ERROR"

    send_message(sock, enc.tobytes())
    response = recv_message(sock)
    if response is None:
        return "SOCKET_CLOSED"

    try:
        return response.decode("utf-8")
    except UnicodeDecodeError:
        return "BAD_RESPONSE"


def make_visualization(
    image_bgr: np.ndarray,
    roi_bgr: Optional[np.ndarray],
    detection: Optional[DetectionResult],
    reading: str,
) -> np.ndarray:
    vis_top = image_bgr.copy()

    if detection is not None:
        cv2.rectangle(vis_top, (detection.x1, detection.y1), (detection.x2, detection.y2), (0, 220, 0), 2)
        cv2.putText(
            vis_top,
            f"{detection.method} | conf={detection.confidence:.3f}",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 255),
            2,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            vis_top,
            "No ROI detected",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 220),
            2,
            cv2.LINE_AA,
        )

    cv2.putText(
        vis_top,
        f"Reading: {reading}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 0, 220),
        2,
        cv2.LINE_AA,
    )

    if roi_bgr is None or roi_bgr.size == 0:
        return vis_top

    target_w = vis_top.shape[1]
    roi_h = max(80, int(roi_bgr.shape[0] * (target_w / max(1, roi_bgr.shape[1]))))
    roi_panel = cv2.resize(roi_bgr, (target_w, roi_h), interpolation=cv2.INTER_CUBIC)

    caption = np.full((40, target_w, 3), 255, dtype=np.uint8)
    cv2.putText(
        caption,
        "Cropped ROI sent to Machine B",
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )

    return np.vstack([vis_top, caption, roi_panel])


def connect_to_server(host: str, port: int, timeout: float) -> socket.socket:
    sock = socket.create_connection((host, port), timeout=timeout)
    sock.settimeout(None)
    return sock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Machine A client: detects meter ROI, sends to Machine B, receives reading."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one image file or a folder with test images.",
    )
    parser.add_argument(
        "--model-path",
        default="models/roi_model/yolo_model.pt",
        help="Path to YOLO model weights file (yolo_model.pt).",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Machine B server host.")
    parser.add_argument("--port", type=int, default=9999, help="Machine B server port.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Ultralytics device string, e.g. cpu, 0, 0,1.",
    )
    parser.add_argument(
        "--output-dir",
        default="machine_a/output",
        help="Directory to save visualization outputs.",
    )
    parser.add_argument(
        "--save-roi",
        action="store_true",
        help="Also save each cropped ROI image.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="Socket connection timeout in seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    roi_save_dir = output_dir / "roi_crops"
    if args.save_roi:
        roi_save_dir.mkdir(parents=True, exist_ok=True)

    image_paths = list_images(input_path)

    model_path = resolve_yolo_model_path(args.model_path)

    print(f"[Machine A] Loading YOLO model from: {model_path}")
    yolo_model = YOLO(str(model_path))

    print(
        f"[Machine A] Connecting to Machine B at {args.host}:{args.port} "
        f"(timeout={args.connect_timeout}s)..."
    )
    sock = connect_to_server(args.host, args.port, args.connect_timeout)
    print("[Machine A] Connected to Machine B.")

    processed = 0
    try:
        for image_path in image_paths:
            processed += 1
            print(f"\n[Machine A] ---- Image #{processed}: {image_path.name} ----")

            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                print("[Machine A] Could not read image, skipping.")
                continue

            h, w = image.shape[:2]

            detection = detect_roi_yolo_with_rotation_retry(
                model=yolo_model,
                image_bgr=image,
                conf_threshold=args.conf,
                imgsz=args.imgsz,
                device=args.device,
                rotation_angles=YOLO_ROTATION_RETRY_ANGLES,
            )

            if detection is None:
                print(
                    "[Machine A] YOLO found no ROI on original + rotated views. "
                    "Trying OpenCV fallback..."
                )
                detection = detect_roi_fallback(image)

            reading = "NO_ROI"
            roi = None

            if detection is not None:
                clamped = clamp_box(
                    detection.x1,
                    detection.y1,
                    detection.x2,
                    detection.y2,
                    width=w,
                    height=h,
                )
                if clamped is None:
                    print("[Machine A] Invalid ROI bounds after clamping.")
                    detection = None
                else:
                    x1, y1, x2, y2 = clamped
                    detection.x1, detection.y1, detection.x2, detection.y2 = x1, y1, x2, y2
                    roi = image[y1:y2, x1:x2]

                    # If detection came from a rotated YOLO pass, send ROI cropped from
                    # that rotated frame so digits are upright for downstream CNN logic.
                    if detection.rotation_angle != 0.0 and detection.rotated_box is not None:
                        upright_roi = crop_from_rotated_detection(
                            image_bgr=image,
                            rotation_angle=detection.rotation_angle,
                            rotated_box=detection.rotated_box,
                        )
                        if upright_roi is not None:
                            roi = upright_roi

                    if roi.size == 0:
                        print("[Machine A] ROI crop is empty after detection.")
                        detection = None
                        roi = None
                    else:
                        try:
                            reading = send_roi_and_get_reading(sock, roi)
                        except OSError as ex:
                            print(f"[Machine A] Socket error ({ex}). Reconnecting and retrying once...")
                            sock.close()
                            sock = connect_to_server(args.host, args.port, args.connect_timeout)
                            reading = send_roi_and_get_reading(sock, roi)

            if detection is None:
                box_txt = "None"
                conf_txt = "0.000"
                method_txt = "none"
            else:
                box_txt = f"({detection.x1},{detection.y1},{detection.x2},{detection.y2})"
                conf_txt = f"{detection.confidence:.3f}"
                method_txt = detection.method

            print(
                f"[Machine A] {image_path.name} | box={box_txt} | "
                f"method={method_txt} | conf={conf_txt} | reading={reading}"
            )

            vis = make_visualization(
                image_bgr=image,
                roi_bgr=roi,
                detection=detection,
                reading=reading,
            )

            vis_name = f"{image_path.stem}_result.jpg"
            vis_path = output_dir / vis_name
            cv2.imwrite(str(vis_path), vis)
            print(f"[Machine A] Visualization saved: {vis_path}")

            if args.save_roi and roi is not None and roi.size > 0:
                roi_path = roi_save_dir / f"{image_path.stem}_roi.jpg"
                cv2.imwrite(str(roi_path), roi)
                print(f"[Machine A] ROI crop saved: {roi_path}")

    finally:
        sock.close()
        print("\n[Machine A] Finished. Socket closed.")


if __name__ == "__main__":
    main()
