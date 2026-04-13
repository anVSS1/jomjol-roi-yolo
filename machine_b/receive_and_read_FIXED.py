from __future__ import annotations

import argparse
import socket
import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tensorflow as tf


@dataclass
class DigitSegment:
    x1: int
    x2: int
    crop: np.ndarray
    pred_class: int
    confidence: float


def resolve_model_path(requested_path: str) -> Path:
    """Resolve CNN model path using cnn_model.keras naming."""
    requested = Path(requested_path)
    candidates = [
        requested,
        Path("models/cnn_model.keras"),
        Path("cnn_model.keras"),
    ]

    seen = set()
    for candidate in candidates:
        normalized = candidate.resolve() if candidate.exists() else candidate
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate.resolve()

    searched = "\n".join(f"- {c}" for c in candidates)
    raise FileNotFoundError(
        "Could not find the CNN model. Checked:\n"
        f"{searched}\n\n"
        "Expected filename: cnn_model.keras"
    )


def recvall(sock: socket.socket, size: int) -> Optional[bytes]:
    """Receive exactly size bytes, or None if the socket closes."""
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


def preprocess_roi(roi_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if roi_bgr.ndim == 3:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    else:
        gray = roi_bgr.copy()

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Prefer white foreground (digits) for stable vertical projection.
    white_ratio = float(np.count_nonzero(thresh)) / float(thresh.size)
    if white_ratio > 0.60:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    return gray, thresh


def find_digit_boundaries(binary: np.ndarray) -> list[tuple[int, int]]:
    h, w = binary.shape
    col_activity = np.sum(binary > 0, axis=0)

    min_foreground_per_col = max(1, int(0.12 * h))
    active_cols = col_activity >= min_foreground_per_col

    boundaries: list[tuple[int, int]] = []
    start = None
    for x, is_active in enumerate(active_cols):
        if is_active and start is None:
            start = x
        elif not is_active and start is not None:
            boundaries.append((start, x))
            start = None
    if start is not None:
        boundaries.append((start, w))

    min_width = max(2, int(0.025 * w))
    boundaries = [(s, e) for (s, e) in boundaries if (e - s) >= min_width]

    if not boundaries:
        return []

    max_gap = max(1, int(0.01 * w))
    merged = [boundaries[0]]
    for s, e in boundaries[1:]:
        ps, pe = merged[-1]
        if s - pe <= max_gap:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))

    return merged


def prepare_digit_for_model(digit_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    resized = cv2.resize(digit_gray, (20, 32), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_GRAY2BGR)

    # The model was trained on raw pixel values (no /255 normalization).
    model_input = rgb.astype(np.float32)[None, ...]
    return model_input, rgb


def classify_segments(
    gray: np.ndarray,
    boundaries: list[tuple[int, int]],
    model: tf.keras.Model,
) -> list[DigitSegment]:
    segments: list[DigitSegment] = []

    for x1, x2 in boundaries:
        crop_gray = gray[:, x1:x2]
        if crop_gray.size == 0:
            continue

        model_input, resized_crop = prepare_digit_for_model(crop_gray)
        probs = model.predict(model_input, verbose=0)[0]
        pred_class = int(np.argmax(probs))
        confidence = float(np.max(probs))

        segments.append(
            DigitSegment(
                x1=x1,
                x2=x2,
                crop=resized_crop,
                pred_class=pred_class,
                confidence=confidence,
            )
        )

    segments.sort(key=lambda item: item.x1)
    return segments


def segments_to_reading(segments: list[DigitSegment]) -> str:
    if not segments:
        return "NO_DIGITS"

    chars: list[str] = []
    for segment in segments:
        if segment.pred_class == 10:
            chars.append("N")
        elif 0 <= segment.pred_class <= 9:
            chars.append(str(segment.pred_class))
        else:
            chars.append("?")
    return "".join(chars)


def build_digit_strip(segments: list[DigitSegment]) -> np.ndarray:
    if not segments:
        strip = np.full((70, 280, 3), 255, dtype=np.uint8)
        cv2.putText(
            strip,
            "No digit crops detected",
            (10, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 180),
            2,
            cv2.LINE_AA,
        )
        return strip

    cell_w = 46
    cell_h = 74
    strip = np.full((cell_h, cell_w * len(segments), 3), 255, dtype=np.uint8)

    for idx, seg in enumerate(segments):
        resized = cv2.resize(seg.crop, (28, 48), interpolation=cv2.INTER_NEAREST)
        x0 = idx * cell_w + 9
        y0 = 6
        strip[y0 : y0 + 48, x0 : x0 + 28] = resized

        label = "N" if seg.pred_class == 10 else str(seg.pred_class)
        conf_txt = f"{seg.confidence:.2f}"
        cv2.putText(
            strip,
            f"{label}:{conf_txt}",
            (idx * cell_w + 3, 68),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

    return strip


def save_debug_output(
    roi_bgr: np.ndarray,
    binary: np.ndarray,
    boundaries: list[tuple[int, int]],
    segments: list[DigitSegment],
    reading: str,
    frame_idx: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = roi_bgr.copy()
    for seg in segments:
        cv2.rectangle(annotated, (seg.x1, 0), (seg.x2, annotated.shape[0] - 1), (0, 220, 0), 2)
        cls_label = "N" if seg.pred_class == 10 else str(seg.pred_class)
        cv2.putText(
            annotated,
            f"{cls_label}:{seg.confidence:.2f}",
            (max(0, seg.x1 - 2), 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 80, 255),
            1,
            cv2.LINE_AA,
        )

    if not boundaries:
        cv2.putText(
            annotated,
            "No boundaries found",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 200),
            2,
            cv2.LINE_AA,
        )

    binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    digit_strip = build_digit_strip(segments)

    w = max(annotated.shape[1], binary_vis.shape[1], digit_strip.shape[1])

    def pad_to_width(img: np.ndarray, target_w: int) -> np.ndarray:
        pad = target_w - img.shape[1]
        if pad <= 0:
            return img
        return cv2.copyMakeBorder(img, 0, 0, 0, pad, cv2.BORDER_CONSTANT, value=(255, 255, 255))

    panel_top = pad_to_width(annotated, w)
    panel_mid = pad_to_width(binary_vis, w)
    panel_bottom = pad_to_width(digit_strip, w)

    panel = np.vstack([panel_top, panel_mid, panel_bottom])
    cv2.putText(
        panel,
        f"Reading: {reading}",
        (10, panel.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 220),
        2,
        cv2.LINE_AA,
    )

    safe_reading = "".join(c if c.isalnum() else "_" for c in reading)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_path = output_dir / f"{frame_idx:05d}_{safe_reading}_{timestamp}.jpg"
    cv2.imwrite(str(out_path), panel)
    return out_path


def process_roi(
    roi_bgr: np.ndarray,
    model: tf.keras.Model,
    save_debug: bool,
    debug_dir: Path,
    frame_idx: int,
) -> tuple[str, int, Optional[Path]]:
    gray, binary = preprocess_roi(roi_bgr)
    boundaries = find_digit_boundaries(binary)
    segments = classify_segments(gray, boundaries, model)
    reading = segments_to_reading(segments)

    debug_path = None
    if save_debug:
        debug_path = save_debug_output(
            roi_bgr=roi_bgr,
            binary=binary,
            boundaries=boundaries,
            segments=segments,
            reading=reading,
            frame_idx=frame_idx,
            output_dir=debug_dir,
        )

    return reading, len(segments), debug_path


def handle_connection(
    conn: socket.socket,
    addr: tuple[str, int],
    model: tf.keras.Model,
    save_debug: bool,
    debug_dir: Path,
) -> None:
    print(f"[Machine B] Client connected from {addr[0]}:{addr[1]}")
    frame_idx = 0

    while True:
        payload = recv_message(conn)
        if payload is None:
            print("[Machine B] Client disconnected.")
            break

        frame_idx += 1
        print(f"\n[Machine B] ---- Frame #{frame_idx} ----")

        if len(payload) == 0:
            reading = "EMPTY_PAYLOAD"
            send_message(conn, reading.encode("utf-8"))
            print("[Machine B] Received empty payload.")
            continue

        arr = np.frombuffer(payload, dtype=np.uint8)
        roi = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if roi is None:
            reading = "DECODE_ERROR"
            send_message(conn, reading.encode("utf-8"))
            print("[Machine B] Failed to decode JPEG bytes.")
            continue

        reading, digit_count, debug_path = process_roi(
            roi_bgr=roi,
            model=model,
            save_debug=save_debug,
            debug_dir=debug_dir,
            frame_idx=frame_idx,
        )

        send_message(conn, reading.encode("utf-8"))
        print(
            "[Machine B] ROI shape="
            f"{roi.shape[1]}x{roi.shape[0]} | digits={digit_count} | reading={reading}"
        )
        if debug_path is not None:
            print(f"[Machine B] Debug image saved: {debug_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Machine B server: receives ROI images, segments digits, classifies, and returns reading."
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host/IP to bind the socket server.")
    parser.add_argument("--port", type=int, default=9999, help="Port to bind the socket server.")
    parser.add_argument(
        "--model-path",
        default="models/cnn_model.keras",
        help="Path to Keras digit classifier model file.",
    )
    parser.add_argument(
        "--debug-dir",
        default="machine_b/debug_outputs",
        help="Folder to store segmentation and crop debug images.",
    )
    parser.add_argument(
        "--no-debug-save",
        action="store_true",
        help="Disable writing debug images to disk.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_dir = Path(args.debug_dir)

    model_path = resolve_model_path(args.model_path)
    print(f"[Machine B] Loading CNN model from: {model_path}")
    model = tf.keras.models.load_model(model_path)

    input_shape = getattr(model, "input_shape", None)
    output_shape = getattr(model, "output_shape", None)
    print(f"[Machine B] Model loaded. input_shape={input_shape}, output_shape={output_shape}")

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((args.host, args.port))
        server.listen(4)
        print(f"[Machine B] Listening on {args.host}:{args.port} ...")

        try:
            while True:
                conn, addr = server.accept()
                with conn:
                    handle_connection(
                        conn=conn,
                        addr=addr,
                        model=model,
                        save_debug=not args.no_debug_save,
                        debug_dir=debug_dir,
                    )
        except KeyboardInterrupt:
            print("\n[Machine B] Stopped by user.")


if __name__ == "__main__":
    main()
