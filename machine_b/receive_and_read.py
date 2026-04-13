from __future__ import annotations

"""
receive_and_read.py  —  Machine B

All meters have exactly 5 digits.
Split ROI into 5 equal strips → run CNN on each crop. That's it.
"""

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


N_DIGITS = 5   # ← all meters have 5 digits


@dataclass
class DigitSegment:
    x1: int
    x2: int
    crop: np.ndarray
    pred_class: int
    confidence: float


def resolve_model_path(requested: str) -> Path:
    candidates = [Path(requested), Path("models/cnn_model.keras"), Path("cnn_model.keras")]
    seen: set[str] = set()
    for c in candidates:
        key = str(c.resolve() if c.exists() else c)
        if key in seen:
            continue
        seen.add(key)
        if c.exists():
            return c.resolve()
    raise FileNotFoundError(
        "cnn_model.keras not found. Checked:\n" + "\n".join(f"  {c}" for c in candidates)
    )


def recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def recv_message(sock: socket.socket) -> Optional[bytes]:
    hdr = recvall(sock, 4)
    if hdr is None:
        return None
    size = struct.unpack("!I", hdr)[0]
    return b"" if size == 0 else recvall(sock, size)


def send_message(sock: socket.socket, data: bytes) -> None:
    sock.sendall(struct.pack("!I", len(data)) + data)


# ═══════════════════════════════════════════════════════════════════════════════
#  SPLIT  —  divide ROI width into N_DIGITS equal strips
# ═══════════════════════════════════════════════════════════════════════════════

def split_roi(roi_bgr: np.ndarray) -> list[tuple[int, int]]:
    """Return N_DIGITS (x1, x2) strips of equal width across the ROI."""
    h, w = roi_bgr.shape[:2]
    sw   = w / N_DIGITS
    margin = max(1, int(sw * 0.025))   # tiny overlap so digit edges aren't clipped
    return [
        (max(0,     int(i * sw)       - margin),
         min(w - 1, int((i + 1) * sw) + margin))
        for i in range(N_DIGITS)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
#  CNN PREPARATION  —  must match training pipeline exactly
#  BGR → RGB → resize(20, 32) → raw float32  (NO /255 — BatchNorm handles it)
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_crop(bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if bgr.ndim == 2:
        bgr = cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    if bgr.shape[0] == 0 or bgr.shape[1] == 0:
        bgr = np.full((32, 20, 3), 128, dtype=np.uint8)
    resized = cv2.resize(bgr, (20, 32), interpolation=cv2.INTER_AREA)
    rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return rgb.astype(np.float32)[None, ...], resized   # (1, 32, 20, 3)


# ═══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE: split → CNN → reading
# ═══════════════════════════════════════════════════════════════════════════════

def extract_and_read_digits(
    roi_bgr: np.ndarray,
    model: tf.keras.Model,
) -> tuple[str, list[DigitSegment]]:
    strips   = split_roi(roi_bgr)
    segments: list[DigitSegment] = []

    for x1, x2 in strips:
        crop = roi_bgr[:, x1:x2]
        if crop.size == 0:
            continue
        inp, display = prepare_crop(crop)
        probs        = model.predict(inp, verbose=0)[0]
        segments.append(DigitSegment(
            x1=x1, x2=x2, crop=display,
            pred_class=int(np.argmax(probs)),
            confidence=float(np.max(probs)),
        ))

    if not segments:
        return "NO_DIGITS", []

    reading = "".join(
        "N"  if s.pred_class == 10 else
        str(s.pred_class) if 0 <= s.pred_class <= 9 else "?"
        for s in segments
    )
    return reading, segments


# ═══════════════════════════════════════════════════════════════════════════════
#  DEBUG OUTPUT
# ═══════════════════════════════════════════════════════════════════════════════

def build_digit_strip(segments: list[DigitSegment]) -> np.ndarray:
    if not segments:
        strip = np.full((70, 280, 3), 255, dtype=np.uint8)
        cv2.putText(strip, "No digits", (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 180), 2, cv2.LINE_AA)
        return strip
    cw, ch = 52, 80
    strip  = np.full((ch, cw * len(segments), 3), 255, dtype=np.uint8)
    for i, seg in enumerate(segments):
        thumb = cv2.resize(seg.crop, (32, 56), interpolation=cv2.INTER_NEAREST)
        strip[4:60, i * cw + 10 : i * cw + 42] = thumb
        lbl = "N" if seg.pred_class == 10 else str(seg.pred_class)
        cv2.putText(strip, f"{lbl}:{seg.confidence:.2f}",
                    (i * cw + 2, 74), cv2.FONT_HERSHEY_SIMPLEX,
                    0.36, (20, 20, 20), 1, cv2.LINE_AA)
    return strip


def save_debug_output(
    roi_bgr: np.ndarray,
    segments: list[DigitSegment],
    reading: str,
    frame_idx: int,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated = roi_bgr.copy()
    for seg in segments:
        cv2.rectangle(annotated, (seg.x1, 0),
                      (seg.x2, roi_bgr.shape[0] - 1), (0, 220, 0), 2)
        lbl = "N" if seg.pred_class == 10 else str(seg.pred_class)
        cv2.putText(annotated, f"{lbl}:{seg.confidence:.2f}",
                    (max(0, seg.x1), 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 60, 255), 1, cv2.LINE_AA)
    strip = build_digit_strip(segments)
    W = max(annotated.shape[1], strip.shape[1])

    def pad(img, tw):
        p = tw - img.shape[1]
        return img if p <= 0 else cv2.copyMakeBorder(
            img, 0, 0, 0, p, cv2.BORDER_CONSTANT, value=255)

    panel = np.vstack([pad(annotated, W), pad(strip, W)])
    cv2.putText(panel, f"Reading: {reading}",
                (10, panel.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, (0, 0, 200), 2, cv2.LINE_AA)
    safe = "".join(c if c.isalnum() else "_" for c in reading)
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"{frame_idx:05d}_{safe}_{ts}.jpg"
    cv2.imwrite(str(path), panel)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
#  SERVER
# ═══════════════════════════════════════════════════════════════════════════════

def process_roi(
    roi_bgr: np.ndarray,
    model: tf.keras.Model,
    save_debug: bool,
    debug_dir: Path,
    frame_idx: int,
) -> tuple[str, int, Optional[Path]]:
    reading, segments = extract_and_read_digits(roi_bgr, model)
    debug_path = None
    if save_debug:
        debug_path = save_debug_output(
            roi_bgr, segments, reading, frame_idx, debug_dir)
    return reading, len(segments), debug_path


def handle_connection(conn, addr, model, save_debug, debug_dir):
    print(f"[Machine B] Connected: {addr[0]}:{addr[1]}")
    frame_idx = 0
    while True:
        payload = recv_message(conn)
        if payload is None:
            print("[Machine B] Client disconnected.")
            break
        frame_idx += 1
        print(f"\n[Machine B] ── Frame #{frame_idx} ──")
        if len(payload) == 0:
            send_message(conn, b"EMPTY_PAYLOAD")
            continue
        roi = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
        if roi is None:
            send_message(conn, b"DECODE_ERROR")
            continue
        reading, n, dp = process_roi(roi, model, save_debug, debug_dir, frame_idx)
        send_message(conn, reading.encode("utf-8"))
        print(f"[Machine B] {roi.shape[1]}x{roi.shape[0]} | digits={n} | reading={reading}")
        if dp:
            print(f"[Machine B] Debug: {dp}")


def parse_args():
    p = argparse.ArgumentParser(description="Machine B: 5-digit split + CNN reading")
    p.add_argument("--host",          default="127.0.0.1")
    p.add_argument("--port",          type=int, default=9999)
    p.add_argument("--model-path",    default="models/cnn_model.keras")
    p.add_argument("--debug-dir",     default="machine_b/debug_outputs")
    p.add_argument("--no-debug-save", action="store_true")
    return p.parse_args()


def main():
    args       = parse_args()
    model_path = resolve_model_path(args.model_path)
    print(f"[Machine B] Model  : {model_path}")
    print(f"[Machine B] Mode   : fixed {N_DIGITS}-digit equal split → CNN")
    model = tf.keras.models.load_model(model_path)
    print(f"[Machine B] input  : {model.input_shape}")    # (None, 32, 20, 3)
    print(f"[Machine B] output : {model.output_shape}")   # (None, 11)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((args.host, args.port))
        srv.listen(4)
        print(f"[Machine B] Listening on {args.host}:{args.port}")
        try:
            while True:
                conn, addr = srv.accept()
                with conn:
                    handle_connection(conn, addr, model,
                                      not args.no_debug_save,
                                      Path(args.debug_dir))
        except KeyboardInterrupt:
            print("\n[Machine B] Stopped.")


if __name__ == "__main__":
    main()
