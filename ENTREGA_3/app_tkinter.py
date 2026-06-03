from __future__ import annotations

import html
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import BOTH, END, LEFT, RIGHT, W, filedialog, messagebox, ttk
from tkinter import BooleanVar, Frame, Label, StringVar, Tk, Toplevel
from tkinter.scrolledtext import ScrolledText
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.transforms import InterpolationMode

try:
    import winsound
except ImportError:  # pragma: no cover - winsound solo existe en Windows.
    winsound = None


PROJECT_ROOT = Path(__file__).resolve().parent
YOLO_MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"
CNN_CHECKPOINT_PATH = PROJECT_ROOT / "outputs" / "models" / "best_model.pt"
APP_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "app"

NEGATIVE_LABEL = "normal"
POSITIVE_LABEL = "hurto_simulado"
LABEL_TO_INDEX = {NEGATIVE_LABEL: 0, POSITIVE_LABEL: 1}
INDEX_TO_LABEL = {value: key for key, value in LABEL_TO_INDEX.items()}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SENSITIVITY_LEVELS = {
    "Alta": {
        "threshold": 0.45,
        "description": "Detecta mas eventos, puede generar mas alertas.",
    },
    "Media": {
        "threshold": 0.60,
        "description": "Equilibrado, recomendado.",
    },
    "Baja": {
        "threshold": 0.80,
        "description": "Solo marca los casos mas evidentes.",
    },
}


@dataclass(frozen=True)
class AppSettings:
    alert_threshold: float = 0.60
    early_exit_threshold: float = 0.90
    debug_mode: bool = False
    device_name: str = "auto"
    tracker_name: str = "botsort.yaml"
    person_class_id: int = 0
    conf_threshold: float = 0.25
    iou_threshold: float = 0.50
    min_track_length_frames: int = 8
    person_window_seconds: float = 3.0
    person_window_stride_seconds: float = 1.5
    frames_per_person_window: int = 9
    min_valid_frames_per_window: int = 7
    bbox_padding_ratio: float = 0.10
    grid_rows: int = 3
    grid_cols: int = 3
    crop_width: int = 96
    crop_height: int = 96
    cnn_image_size: int = 224
    cnn_dropout: float = 0.30


LogCallback = Callable[[str], None]
ProgressCallback = Callable[[float, str, str], None]


def emit_progress(
    progress: ProgressCallback | None,
    value: float,
    stage: str,
    detail: str = "",
) -> None:
    if progress is None:
        return

    bounded_value = max(0.0, min(1.0, float(value)))
    progress(bounded_value, stage, detail)


def emit_stage_progress(
    progress: ProgressCallback | None,
    stage_start: float,
    stage_end: float,
    stage_value: float,
    stage: str,
    detail: str = "",
) -> None:
    bounded_stage_value = max(0.0, min(1.0, float(stage_value)))
    absolute_value = stage_start + ((stage_end - stage_start) * bounded_stage_value)
    emit_progress(progress, absolute_value, stage, detail)


def emit_debug(debug: LogCallback | None, message: str) -> None:
    if debug is not None:
        debug(message)


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_name)


def ensure_yolo_weights(weights_path: Path, log: Callable[[str], None]) -> None:
    if weights_path.exists():
        log(f"Peso YOLO encontrado en disco: {weights_path}")
        return

    log(f"No existe {weights_path}. Se descargará yolo11n.pt desde Ultralytics.")
    attempt_download_asset(str(weights_path), release="latest")
    if not weights_path.exists():
        raise FileNotFoundError(f"No fue posible descargar {weights_path}")


def load_yolo_model(weights_path: Path, log: Callable[[str], None]) -> YOLO:
    ensure_yolo_weights(weights_path, log)
    model = YOLO(str(weights_path))
    log(f"YOLO cargado. Clases disponibles: {len(model.names)}. Clase 0: {model.names[0]}")
    return model


def load_classifier(
    checkpoint_path: Path,
    device: torch.device,
    dropout: float,
    image_size: int,
) -> tuple[torch.nn.Module, transforms.Compose, dict[int, str]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = resnet50(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 2),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    checkpoint_index_to_class = checkpoint.get("index_to_class", INDEX_TO_LABEL)
    index_to_class = {int(key): str(value) for key, value in checkpoint_index_to_class.items()}
    return model, transform, index_to_class


def build_run_dir(video_path: Path) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = APP_OUTPUT_DIR / f"{video_path.stem}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def track_people_in_video(
    video_path: Path,
    yolo_model: YOLO,
    settings: AppSettings,
    log: LogCallback,
    progress: ProgressCallback | None = None,
    debug: LogCallback | None = None,
) -> tuple[pd.DataFrame, float, int, tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()

    if fps <= 0:
        fps = 30.0

    emit_debug(
        debug,
        (
            f"Tracking: fps={fps:.2f}, frames={frame_count}, tamano={frame_width}x{frame_height}, "
            f"tracker={settings.tracker_name}, conf={settings.conf_threshold:.2f}, iou={settings.iou_threshold:.2f}"
        ),
    )

    tracker_name = settings.tracker_name
    if tracker_name in {"botsort", "bytetrack"}:
        tracker_name = f"{tracker_name}.yaml"

    track_kwargs = {
        "source": str(video_path),
        "imgsz": 480,
        "stream": True,
        "persist": True,
        "tracker": tracker_name,
        "classes": [settings.person_class_id],
        "conf": settings.conf_threshold,
        "iou": settings.iou_threshold,
        "verbose": False,
    }

    log("Iniciando tracking de personas.")
    results = yolo_model.track(**track_kwargs)
    raw_rows: list[dict[str, float | int | str]] = []
    progress_step = max(1, frame_count // 40) if frame_count > 0 else 1

    for frame_idx, result in enumerate(results):
        if result.boxes is None or len(result.boxes) == 0 or result.boxes.id is None:
            if frame_count > 0 and ((frame_idx + 1) % progress_step == 0 or frame_idx + 1 == frame_count):
                emit_stage_progress(
                    progress,
                    0.0,
                    1.0,
                    (frame_idx + 1) / frame_count,
                    "Tracking personas",
                    f"Frames procesados: {frame_idx + 1}/{frame_count}",
                )
            continue

        boxes = result.boxes
        timestamp_sec = frame_idx / fps

        for detection_idx in range(len(boxes)):
            class_id = int(boxes.cls[detection_idx].item())
            if class_id != settings.person_class_id:
                continue

            track_id = int(boxes.id[detection_idx].item())
            confidence = float(boxes.conf[detection_idx].item())
            x1, y1, x2, y2 = boxes.xyxy[detection_idx].tolist()
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            center_x = x1 + (bbox_width / 2.0)
            center_y = y1 + (bbox_height / 2.0)

            raw_rows.append(
                {
                    "frame_idx": int(frame_idx),
                    "timestamp_sec": round(float(timestamp_sec), 6),
                    "track_id": int(track_id),
                    "class_id": int(class_id),
                    "confidence": round(confidence, 6),
                    "x1": round(float(x1), 6),
                    "y1": round(float(y1), 6),
                    "x2": round(float(x2), 6),
                    "y2": round(float(y2), 6),
                    "bbox_width": round(float(bbox_width), 6),
                    "bbox_height": round(float(bbox_height), 6),
                    "center_x": round(float(center_x), 6),
                    "center_y": round(float(center_y), 6),
                }
            )

        if frame_count > 0 and ((frame_idx + 1) % progress_step == 0 or frame_idx + 1 == frame_count):
            emit_stage_progress(
                progress,
                0.0,
                1.0,
                (frame_idx + 1) / frame_count,
                "Tracking personas",
                f"Frames procesados: {frame_idx + 1}/{frame_count}",
            )
            emit_debug(
                debug,
                f"Tracking parcial: frame {frame_idx + 1}/{frame_count}, detecciones acumuladas={len(raw_rows)}",
            )

    raw_columns = [
        "frame_idx",
        "timestamp_sec",
        "track_id",
        "class_id",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "bbox_width",
        "bbox_height",
        "center_x",
        "center_y",
    ]
    raw_df = pd.DataFrame(raw_rows, columns=raw_columns) if raw_rows else pd.DataFrame(columns=raw_columns)

    if raw_df.empty:
        clean_df = pd.DataFrame(columns=raw_columns)
    else:
        valid_mask = (
            raw_df["track_id"].notna()
            & raw_df["x1"].notna()
            & raw_df["y1"].notna()
            & raw_df["x2"].notna()
            & raw_df["y2"].notna()
            & raw_df["bbox_width"].notna()
            & raw_df["bbox_height"].notna()
            & (raw_df["x2"] > raw_df["x1"])
            & (raw_df["y2"] > raw_df["y1"])
            & (raw_df["bbox_width"] > 0)
            & (raw_df["bbox_height"] > 0)
        )
        clean_df = raw_df.loc[valid_mask, raw_columns].copy()

    log(f"Tracking completado. Detecciones limpias: {len(clean_df)}")
    emit_debug(
        debug,
        f"Tracking finalizado: detecciones brutas={len(raw_df)}, detecciones limpias={len(clean_df)}",
    )
    return clean_df, fps, frame_count, (frame_width, frame_height)


def build_track_windows(track_df: pd.DataFrame, settings: AppSettings) -> list[dict[str, object]]:
    track_df = (
        track_df.sort_values(["frame_idx", "timestamp_sec"])
        .drop_duplicates(subset=["frame_idx"], keep="first")
        .reset_index(drop=True)
    )

    track_start_sec = float(track_df["timestamp_sec"].min())
    track_end_sec = float(track_df["timestamp_sec"].max())
    track_duration_sec = track_end_sec - track_start_sec

    if track_duration_sec <= settings.person_window_seconds:
        time_windows = [(round(track_start_sec, 6), round(track_end_sec, 6))]
    else:
        max_start_sec = track_end_sec - settings.person_window_seconds
        start_values = np.arange(track_start_sec, max_start_sec + 1e-9, settings.person_window_stride_seconds)
        time_windows = [
            (
                round(float(start_sec), 6),
                round(float(start_sec + settings.person_window_seconds), 6),
            )
            for start_sec in start_values
        ]

    windows: list[dict[str, object]] = []
    for window_number, (start_sec, end_sec) in enumerate(time_windows):
        window_track_df = track_df[
            (track_df["timestamp_sec"] >= start_sec)
            & (track_df["timestamp_sec"] <= end_sec)
        ].copy()

        if window_track_df.empty:
            sampled_rows_df = window_track_df.copy()
        elif len(window_track_df) <= settings.frames_per_person_window:
            sampled_rows_df = (
                window_track_df.sort_values("frame_idx")
                .drop_duplicates(subset=["frame_idx"], keep="first")
                .copy()
            )
        else:
            target_timestamps = np.linspace(start_sec, end_sec, num=settings.frames_per_person_window)
            available_df = window_track_df.sort_values("timestamp_sec").copy()
            selected_indices: list[int] = []

            for target_timestamp in target_timestamps:
                remaining_df = available_df.loc[~available_df.index.isin(selected_indices)].copy()
                if remaining_df.empty:
                    break

                remaining_df["distance_to_target"] = (remaining_df["timestamp_sec"] - target_timestamp).abs()
                best_row_index = remaining_df.sort_values(["distance_to_target", "frame_idx"]).index[0]
                selected_indices.append(int(best_row_index))

            sampled_rows_df = available_df.loc[selected_indices].sort_values("frame_idx").copy()

        if sampled_rows_df.empty:
            start_frame = int(track_df["frame_idx"].min())
            end_frame = int(track_df["frame_idx"].max())
            sampled_frame_indices: list[int] = []
            sampled_timestamps: list[float] = []
        else:
            start_frame = int(window_track_df["frame_idx"].min())
            end_frame = int(window_track_df["frame_idx"].max())
            sampled_frame_indices = sampled_rows_df["frame_idx"].astype(int).tolist()
            sampled_timestamps = sampled_rows_df["timestamp_sec"].round(6).astype(float).tolist()

        windows.append(
            {
                "track_id": int(track_df["track_id"].iloc[0]),
                "window_id": f"w{window_number:04d}",
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
                "start_sec": round(float(start_sec), 6),
                "end_sec": round(float(end_sec), 6),
                "sampled_rows_df": sampled_rows_df,
                "sampled_frame_indices": sampled_frame_indices,
                "sampled_timestamps": sampled_timestamps,
                "valid_sample": bool(len(sampled_rows_df) >= settings.min_valid_frames_per_window),
            }
        )

    return windows


def crop_person_from_frame(
    frame: np.ndarray,
    track_row: pd.Series | dict[str, object],
    settings: AppSettings,
) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    x1 = float(track_row["x1"])
    y1 = float(track_row["y1"])
    x2 = float(track_row["x2"])
    y2 = float(track_row["y2"])

    bbox_width = x2 - x1
    bbox_height = y2 - y1
    pad_x = bbox_width * settings.bbox_padding_ratio
    pad_y = bbox_height * settings.bbox_padding_ratio

    padded_x1 = max(0, int(round(x1 - pad_x)))
    padded_y1 = max(0, int(round(y1 - pad_y)))
    padded_x2 = min(frame_width, int(round(x2 + pad_x)))
    padded_y2 = min(frame_height, int(round(y2 + pad_y)))

    if padded_x2 <= padded_x1 or padded_y2 <= padded_y1:
        return None

    crop = frame[padded_y1:padded_y2, padded_x1:padded_x2]
    if crop is None or crop.size == 0 or crop.shape[0] <= 1 or crop.shape[1] <= 1:
        return None

    return crop


def build_temporal_mosaic(crops: list[np.ndarray], settings: AppSettings) -> np.ndarray:
    total_cells = settings.grid_rows * settings.grid_cols
    resized_crops: list[np.ndarray] = []

    for tile_index in range(total_cells):
        if tile_index < len(crops):
            resized_crop = cv2.resize(
                crops[tile_index],
                (settings.crop_width, settings.crop_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized_crop = np.zeros((settings.crop_height, settings.crop_width, 3), dtype=np.uint8)
        resized_crops.append(resized_crop)

    mosaic_rows: list[np.ndarray] = []
    for row_index in range(settings.grid_rows):
        start_index = row_index * settings.grid_cols
        end_index = start_index + settings.grid_cols
        mosaic_rows.append(np.concatenate(resized_crops[start_index:end_index], axis=1))

    return np.concatenate(mosaic_rows, axis=0)


def select_window_entries(
    buffer_entries: deque[dict[str, object]],
    start_sec: float,
    end_sec: float,
    settings: AppSettings,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    window_entries = [
        entry
        for entry in buffer_entries
        if start_sec <= float(entry["timestamp_sec"]) <= end_sec
    ]

    if len(window_entries) <= settings.frames_per_person_window:
        sampled_entries = sorted(window_entries, key=lambda entry: int(entry["frame_idx"]))
        return window_entries, sampled_entries

    available_entries = sorted(window_entries, key=lambda entry: float(entry["timestamp_sec"]))
    selected_indices: list[int] = []
    target_timestamps = np.linspace(start_sec, end_sec, num=settings.frames_per_person_window)

    for target_timestamp in target_timestamps:
        remaining_indices = [
            index for index in range(len(available_entries)) if index not in selected_indices
        ]
        if not remaining_indices:
            break

        best_index = min(
            remaining_indices,
            key=lambda index: (
                abs(float(available_entries[index]["timestamp_sec"]) - float(target_timestamp)),
                int(available_entries[index]["frame_idx"]),
            ),
        )
        selected_indices.append(int(best_index))

    sampled_entries = sorted(
        [available_entries[index] for index in selected_indices],
        key=lambda entry: int(entry["frame_idx"]),
    )
    return window_entries, sampled_entries


def classify_window_entries(
    sample_id: str,
    track_id: int,
    window_id: str,
    start_sec: float,
    end_sec: float,
    window_entries: list[dict[str, object]],
    sampled_entries: list[dict[str, object]],
    classifier: torch.nn.Module,
    eval_transform: transforms.Compose,
    index_to_class: dict[int, str],
    device: torch.device,
    settings: AppSettings,
    run_dir: Path,
    best_positive_probability: float,
    debug: LogCallback | None = None,
) -> tuple[dict[str, object], float, Path | None, bool]:
    crops = [
        entry["crop"]
        for entry in sampled_entries
        if isinstance(entry.get("crop"), np.ndarray)
    ]
    used_frame_indices = [
        int(entry["frame_idx"])
        for entry in sampled_entries
        if isinstance(entry.get("crop"), np.ndarray)
    ]
    used_timestamps = [
        round(float(entry["timestamp_sec"]), 6)
        for entry in sampled_entries
        if isinstance(entry.get("crop"), np.ndarray)
    ]

    if window_entries:
        start_frame = int(min(int(entry["frame_idx"]) for entry in window_entries))
        end_frame = int(max(int(entry["frame_idx"]) for entry in window_entries))
    else:
        start_frame = -1
        end_frame = -1

    valid_sample = len(crops) >= settings.min_valid_frames_per_window
    positive_probability = 0.0
    negative_probability = 0.0
    predicted_label = NEGATIVE_LABEL
    best_evidence_path: Path | None = None
    high_confidence_exit = False

    if valid_sample:
        mosaic = build_temporal_mosaic(crops, settings)
        mosaic_rgb = cv2.cvtColor(mosaic, cv2.COLOR_BGR2RGB)
        mosaic_pil = Image.fromarray(mosaic_rgb)
        image_tensor = eval_transform(mosaic_pil).unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = classifier(image_tensor)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]

        negative_probability = float(probabilities[LABEL_TO_INDEX[NEGATIVE_LABEL]])
        positive_probability = float(probabilities[LABEL_TO_INDEX[POSITIVE_LABEL]])
        predicted_index = int(np.argmax(probabilities))
        predicted_label = index_to_class.get(predicted_index, INDEX_TO_LABEL[predicted_index])

        if positive_probability >= settings.alert_threshold and positive_probability > best_positive_probability:
            best_positive_probability = positive_probability
            best_evidence_path = run_dir / "best_evidence.png"
            cv2.imwrite(str(best_evidence_path), mosaic)
            emit_debug(
                debug,
                (
                    f"Nueva mejor evidencia: {sample_id}, prob_hurto={positive_probability:.4f}, "
                    f"umbral={settings.alert_threshold:.2f}"
                ),
            )

        high_confidence_exit = positive_probability >= settings.early_exit_threshold

    prediction = {
        "sample_id": sample_id,
        "track_id": int(track_id),
        "window_id": str(window_id),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "start_sec": round(float(start_sec), 6),
        "end_sec": round(float(end_sec), 6),
        "num_frames_used": int(len(crops)),
        "valid_sample": bool(valid_sample),
        "sampled_frame_indices": json.dumps(used_frame_indices),
        "sampled_timestamps": json.dumps(used_timestamps),
        "cnn_prob_normal": round(float(negative_probability), 6),
        "cnn_prob_hurto": round(float(positive_probability), 6),
        "predicted_label": str(predicted_label),
        "alert_positive": bool(positive_probability >= settings.alert_threshold),
    }
    return prediction, best_positive_probability, best_evidence_path, high_confidence_exit


def classify_video_windows(
    video_path: Path,
    clean_df: pd.DataFrame,
    classifier: torch.nn.Module,
    eval_transform: transforms.Compose,
    index_to_class: dict[int, str],
    device: torch.device,
    settings: AppSettings,
    run_dir: Path,
    log: LogCallback,
    progress: ProgressCallback | None = None,
    debug: LogCallback | None = None,
) -> tuple[pd.DataFrame, Path | None]:
    prediction_columns = [
        "sample_id",
        "track_id",
        "window_id",
        "start_frame",
        "end_frame",
        "start_sec",
        "end_sec",
        "num_frames_used",
        "valid_sample",
        "sampled_frame_indices",
        "sampled_timestamps",
        "cnn_prob_normal",
        "cnn_prob_hurto",
        "predicted_label",
        "alert_positive",
    ]

    if clean_df.empty:
        return pd.DataFrame(columns=prediction_columns), None

    valid_track_ids = (
        clean_df.groupby("track_id")
        .size()
        .reset_index(name="num_detections")
        .loc[lambda frame: frame["num_detections"] >= settings.min_track_length_frames, "track_id"]
        .astype(int)
        .tolist()
    )
    filtered_tracks_df = clean_df[clean_df["track_id"].isin(valid_track_ids)].copy()
    if filtered_tracks_df.empty:
        log("No quedaron tracks con longitud suficiente para construir ventanas.")
        return pd.DataFrame(columns=prediction_columns), None

    track_window_counts: dict[int, int] = {}
    track_start_by_id: dict[int, float] = {}
    track_end_by_id: dict[int, float] = {}
    total_windows = 0
    for track_id, track_df in filtered_tracks_df.groupby("track_id"):
        windows = build_track_windows(track_df, settings)
        int_track_id = int(track_id)
        track_window_counts[int_track_id] = len(windows)
        track_start_by_id[int_track_id] = round(float(track_df["timestamp_sec"].min()), 6)
        track_end_by_id[int_track_id] = round(float(track_df["timestamp_sec"].max()), 6)
        total_windows += len(windows)

    emit_debug(
        debug,
        (
            f"Clasificacion rolling buffer: tracks validos={len(track_window_counts)}, "
            f"ventanas planificadas={total_windows}, dispositivo={device}"
        ),
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video para clasificar ventanas: {video_path}")

    frame_lookup = {
        int(frame_idx): frame_rows.to_dict(orient="records")
        for frame_idx, frame_rows in filtered_tracks_df.groupby("frame_idx")
    }
    track_buffers: dict[int, deque[dict[str, object]]] = {
        int(track_id): deque() for track_id in track_window_counts
    }
    next_window_start_by_track = dict(track_start_by_id)
    window_counter_by_track = {int(track_id): 0 for track_id in track_window_counts}
    closed_tracks: set[int] = set()
    predictions: list[dict[str, object]] = []
    best_positive_probability = -1.0
    best_evidence_path: Path | None = None
    video_id = video_path.stem
    processed_windows = 0
    skipped_windows = 0
    frame_idx = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            for row in frame_lookup.get(frame_idx, []):
                track_id = int(row["track_id"])
                if track_id in closed_tracks:
                    continue

                crop = crop_person_from_frame(frame, row, settings)
                entry = {
                    "frame_idx": int(row["frame_idx"]),
                    "timestamp_sec": round(float(row["timestamp_sec"]), 6),
                    "crop": crop,
                }
                track_buffer = track_buffers[track_id]
                track_buffer.append(entry)

                while (
                    window_counter_by_track[track_id] < track_window_counts[track_id]
                    and float(row["timestamp_sec"])
                    >= next_window_start_by_track[track_id] + settings.person_window_seconds
                ):
                    window_start_sec = next_window_start_by_track[track_id]
                    window_end_sec = round(float(window_start_sec + settings.person_window_seconds), 6)
                    window_number = window_counter_by_track[track_id]
                    window_id = f"w{window_number:04d}"
                    sample_id = f"{video_id}__t{track_id}__{window_id}"
                    window_entries, sampled_entries = select_window_entries(
                        track_buffer,
                        window_start_sec,
                        window_end_sec,
                        settings,
                    )
                    (
                        prediction,
                        best_positive_probability,
                        new_best_evidence_path,
                        high_confidence_exit,
                    ) = classify_window_entries(
                        sample_id=sample_id,
                        track_id=track_id,
                        window_id=window_id,
                        start_sec=window_start_sec,
                        end_sec=window_end_sec,
                        window_entries=window_entries,
                        sampled_entries=sampled_entries,
                        classifier=classifier,
                        eval_transform=eval_transform,
                        index_to_class=index_to_class,
                        device=device,
                        settings=settings,
                        run_dir=run_dir,
                        best_positive_probability=best_positive_probability,
                        debug=debug,
                    )
                    if new_best_evidence_path is not None:
                        best_evidence_path = new_best_evidence_path

                    predictions.append(prediction)
                    processed_windows += 1
                    window_counter_by_track[track_id] += 1
                    next_window_start_by_track[track_id] = round(
                        float(next_window_start_by_track[track_id] + settings.person_window_stride_seconds),
                        6,
                    )

                    if total_windows > 0:
                        emit_stage_progress(
                            progress,
                            0.0,
                            1.0,
                            processed_windows / total_windows,
                            "Clasificando ventanas",
                            f"Ventanas procesadas: {processed_windows}/{total_windows}",
                        )

                    if settings.debug_mode and (
                        processed_windows == 1
                        or processed_windows == total_windows
                        or processed_windows % max(1, total_windows // 20) == 0
                    ):
                        emit_debug(
                            debug,
                            (
                                f"Ventana {sample_id}: valid_sample={prediction['valid_sample']}, "
                                f"prob_hurto={float(prediction['cnn_prob_hurto']):.4f}, "
                                f"prob_normal={float(prediction['cnn_prob_normal']):.4f}"
                            ),
                        )

                    while (
                        track_buffer
                        and float(track_buffer[0]["timestamp_sec"])
                        < next_window_start_by_track[track_id] - 1e-9
                    ):
                        track_buffer.popleft()

                    if high_confidence_exit:
                        remaining_track_windows = (
                            track_window_counts[track_id] - window_counter_by_track[track_id]
                        )
                        skipped_windows += remaining_track_windows
                        processed_windows += remaining_track_windows
                        closed_tracks.add(track_id)
                        track_buffer.clear()
                        emit_debug(
                            debug,
                            (
                                f"Early exit track {track_id}: "
                                f"prob_hurto={float(prediction['cnn_prob_hurto']):.4f}, "
                                f"umbral_alto={settings.early_exit_threshold:.2f}, "
                                f"ventanas_omitidas={remaining_track_windows}"
                            ),
                        )
                        if total_windows > 0:
                            emit_stage_progress(
                                progress,
                                0.0,
                                1.0,
                                processed_windows / total_windows,
                                "Clasificando ventanas",
                                f"Ventanas procesadas: {processed_windows}/{total_windows}",
                            )
                        break

            frame_idx += 1

        for track_id, track_buffer in track_buffers.items():
            if track_id in closed_tracks or window_counter_by_track[track_id] > 0:
                continue

            track_duration_sec = track_end_by_id[track_id] - track_start_by_id[track_id]
            if track_duration_sec > settings.person_window_seconds:
                continue

            window_id = "w0000"
            sample_id = f"{video_id}__t{track_id}__{window_id}"
            window_start_sec = track_start_by_id[track_id]
            window_end_sec = track_end_by_id[track_id]
            window_entries, sampled_entries = select_window_entries(
                track_buffer,
                window_start_sec,
                window_end_sec,
                settings,
            )
            (
                prediction,
                best_positive_probability,
                new_best_evidence_path,
                _high_confidence_exit,
            ) = classify_window_entries(
                sample_id=sample_id,
                track_id=track_id,
                window_id=window_id,
                start_sec=window_start_sec,
                end_sec=window_end_sec,
                window_entries=window_entries,
                sampled_entries=sampled_entries,
                classifier=classifier,
                eval_transform=eval_transform,
                index_to_class=index_to_class,
                device=device,
                settings=settings,
                run_dir=run_dir,
                best_positive_probability=best_positive_probability,
                debug=debug,
            )
            if new_best_evidence_path is not None:
                best_evidence_path = new_best_evidence_path
            predictions.append(prediction)
            processed_windows += 1
            window_counter_by_track[track_id] += 1
            if total_windows > 0:
                emit_stage_progress(
                    progress,
                    0.0,
                    1.0,
                    processed_windows / total_windows,
                    "Clasificando ventanas",
                    f"Ventanas procesadas: {processed_windows}/{total_windows}",
                )
    finally:
        capture.release()

    if skipped_windows > 0:
        log(f"Early exit activo: se omitieron {skipped_windows} ventanas posteriores a alertas de alta confianza.")

    predictions_df = pd.DataFrame(predictions, columns=prediction_columns)
    if not predictions_df.empty:
        predictions_df = predictions_df.sort_values(["track_id", "window_id"]).reset_index(drop=True)
    predictions_path = run_dir / "window_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    log(f"Ventanas evaluadas: {len(predictions_df)}")
    log(f"Resultados guardados en {predictions_path}")
    emit_debug(
        debug,
        f"Clasificacion finalizada: positivas={int(predictions_df['alert_positive'].sum()) if not predictions_df.empty else 0}",
    )
    return predictions_df, best_evidence_path


def render_annotated_video(
    video_path: Path,
    clean_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    fps: float,
    frame_count: int,
    frame_size: tuple[int, int],
    settings: AppSettings,
    run_dir: Path,
    log: LogCallback,
    progress: ProgressCallback | None = None,
    debug: LogCallback | None = None,
) -> Path:
    frame_width, frame_height = frame_size
    annotated_video_path = run_dir / "annotated_detection.mp4"
    writer = cv2.VideoWriter(
        str(annotated_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_width, frame_height),
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        writer.release()
        raise RuntimeError(f"No se pudo abrir el video para dibujar la salida: {video_path}")

    frame_lookup = {int(frame_idx): frame_rows.copy() for frame_idx, frame_rows in clean_df.groupby("frame_idx")}
    positive_windows_df = predictions_df[predictions_df["alert_positive"] == True].copy()
    alert_ranges_by_track: dict[int, list[dict[str, float | int]]] = {}

    for track_id, track_rows in positive_windows_df.groupby("track_id"):
        alert_ranges_by_track[int(track_id)] = track_rows[
            ["start_frame", "end_frame", "cnn_prob_hurto"]
        ].to_dict(orient="records")

    log("Construyendo video anotado.")
    frame_idx = 0
    progress_step = max(1, frame_count // 40) if frame_count > 0 else 1
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break

        frame_rows = frame_lookup.get(frame_idx)
        frame_has_alert = False

        if frame_rows is not None:
            for _, row in frame_rows.iterrows():
                x1 = int(round(float(row["x1"])))
                y1 = int(round(float(row["y1"])))
                x2 = int(round(float(row["x2"])))
                y2 = int(round(float(row["y2"])))
                track_id = int(row["track_id"])

                positive_probability = None
                for alert_window in alert_ranges_by_track.get(track_id, []):
                    if int(alert_window["start_frame"]) <= frame_idx <= int(alert_window["end_frame"]):
                        probability_value = float(alert_window["cnn_prob_hurto"])
                        if positive_probability is None or probability_value > positive_probability:
                            positive_probability = probability_value

                if positive_probability is not None:
                    frame_has_alert = True
                    color = (0, 0, 255)
                    label = "Posible hurto"
                else:
                    color = (0, 180, 0)
                    label = "Persona"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    label,
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        if frame_has_alert:
            timestamp_text = f"Tiempo: {format_seconds(frame_idx / fps)}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (0, 0, 180), -1)
            cv2.putText(
                frame,
                f"ALERTA: posible hurto detectado   {timestamp_text}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        writer.write(frame)
        frame_idx += 1
        if frame_count > 0 and (frame_idx % progress_step == 0 or frame_idx == frame_count):
            emit_stage_progress(
                progress,
                0.0,
                1.0,
                frame_idx / frame_count,
                "Renderizando salida",
                f"Frames anotados: {frame_idx}/{frame_count}",
            )
            emit_debug(debug, f"Render: frame {frame_idx}/{frame_count}")

    capture.release()
    writer.release()
    log(f"Video anotado guardado en {annotated_video_path}")
    emit_debug(debug, f"Render finalizado: {annotated_video_path}")
    return annotated_video_path


def analyze_video(
    video_path: Path,
    settings: AppSettings,
    yolo_model: YOLO,
    classifier: torch.nn.Module,
    eval_transform: transforms.Compose,
    index_to_class: dict[int, str],
    device: torch.device,
    log: LogCallback,
    progress: ProgressCallback | None = None,
    debug: LogCallback | None = None,
) -> dict[str, object]:
    run_dir = build_run_dir(video_path)

    emit_progress(progress, 0.02, "Preparando analisis", f"Creando salida para {video_path.name}")
    log(f"Video seleccionado: {video_path}")
    log(f"Salida del análisis: {run_dir}")
    log(f"Dispositivo de inferencia: {device}")
    log(f"Detector de personas: {YOLO_MODEL_PATH}")
    log(f"Clasificador de hurto entrenado: {CNN_CHECKPOINT_PATH}")
    emit_debug(
        debug,
        (
            "Modelos en uso: detector YOLO general para personas y checkpoint entrenado "
            "ResNet50 del proyecto para clasificacion normal vs hurto_simulado."
        ),
    )

    emit_progress(progress, 0.14, "Modelos listos", "Usando modelos cargados en memoria")
    emit_debug(debug, f"Clases del clasificador: {index_to_class}")
    emit_progress(progress, 0.20, "Modelos cargados", "Comenzando tracking de personas")
    log(f"Checkpoint de clasificación cargado: {CNN_CHECKPOINT_PATH}")

    clean_df, fps, frame_count, frame_size = track_people_in_video(
        video_path=video_path,
        yolo_model=yolo_model,
        settings=settings,
        log=log,
        progress=lambda value, stage, detail: emit_stage_progress(progress, 0.20, 0.55, value, stage, detail),
        debug=debug,
    )

    clean_tracks_path = run_dir / "clean_tracks.csv"
    clean_df.to_csv(clean_tracks_path, index=False)
    emit_progress(progress, 0.58, "Tracking completado", "Persistiendo tracks limpios")
    emit_debug(debug, f"Tracks limpios guardados en: {clean_tracks_path}")

    predictions_df, best_evidence_path = classify_video_windows(
        video_path=video_path,
        clean_df=clean_df,
        classifier=classifier,
        eval_transform=eval_transform,
        index_to_class=index_to_class,
        device=device,
        settings=settings,
        run_dir=run_dir,
        log=log,
        progress=lambda value, stage, detail: emit_stage_progress(progress, 0.58, 0.85, value, stage, detail),
        debug=debug,
    )

    if predictions_df.empty:
        video_alert = False
        positive_windows_df = pd.DataFrame(columns=predictions_df.columns)
    else:
        positive_windows_df = predictions_df[predictions_df["alert_positive"] == True].copy()
        video_alert = not positive_windows_df.empty

    annotated_video_path = render_annotated_video(
        video_path=video_path,
        clean_df=clean_df,
        predictions_df=predictions_df,
        fps=fps,
        frame_count=frame_count,
        frame_size=frame_size,
        settings=settings,
        run_dir=run_dir,
        log=log,
        progress=lambda value, stage, detail: emit_stage_progress(progress, 0.85, 0.98, value, stage, detail),
        debug=debug,
    )

    top_events = []
    if not positive_windows_df.empty:
        top_events = (
            positive_windows_df.sort_values("cnn_prob_hurto", ascending=False)
            .head(10)[
                [
                    "sample_id",
                    "track_id",
                    "start_frame",
                    "end_frame",
                    "start_sec",
                    "end_sec",
                    "cnn_prob_hurto",
                ]
            ]
            .to_dict(orient="records")
        )

    mosaic_evidence_path = best_evidence_path if top_events else None

    summary = {
        "video_path": str(video_path),
        "run_dir": str(run_dir),
        "device": str(device),
        "fps": round(float(fps), 6),
        "frame_count": int(frame_count),
        "num_tracks_clean": int(clean_df["track_id"].nunique()) if not clean_df.empty else 0,
        "num_clean_detections": int(len(clean_df)),
        "num_windows_evaluated": int(len(predictions_df)),
        "num_positive_windows": int(len(positive_windows_df)),
        "alert_threshold": float(settings.alert_threshold),
        "video_alert": bool(video_alert),
        "annotated_video_path": str(annotated_video_path),
        "clean_tracks_path": str(clean_tracks_path),
        "window_predictions_path": str(run_dir / "window_predictions.csv"),
        "best_evidence_path": str(best_evidence_path) if best_evidence_path is not None else "",
        "mosaic_evidence_path": str(mosaic_evidence_path) if mosaic_evidence_path is not None else "",
        "incident_evidence_path": str(mosaic_evidence_path) if mosaic_evidence_path is not None else "",
        "models_used": {
            "person_detector_path": str(YOLO_MODEL_PATH),
            "person_detector_type": "pretrained_yolo_person_detector",
            "classifier_checkpoint_path": str(CNN_CHECKPOINT_PATH),
            "classifier_type": "trained_resnet50_classifier",
        },
        "debug_mode": bool(settings.debug_mode),
        "top_events": top_events,
    }
    report_path = write_user_report(summary, run_dir)
    summary["incident_report_path"] = str(report_path)

    summary_path = run_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"Resumen guardado en {summary_path}")
    emit_progress(progress, 1.0, "Analisis completado", f"Proceso finalizado para {video_path.name}")

    return summary


def open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"No existe la ruta: {path}")

    if os.name == "nt":
        os.startfile(str(path))
        return

    subprocess.Popen(["xdg-open", str(path)])


def format_seconds(seconds: float) -> str:
    bounded_seconds = max(0, int(round(float(seconds))))
    minutes, remaining_seconds = divmod(bounded_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{remaining_seconds:02d}"
    return f"{minutes:02d}:{remaining_seconds:02d}"


def suspicion_level(probability: float) -> str:
    if probability >= 0.85:
        return "Alta"
    if probability >= 0.60:
        return "Media"
    return "Baja"


def save_annotated_evidence_frame(
    annotated_video_path: Path,
    frame_idx: int,
    output_path: Path,
) -> Path | None:
    capture = cv2.VideoCapture(str(annotated_video_path))
    if not capture.isOpened():
        return None

    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_idx)))
        ok, frame = capture.read()
        if not ok or frame is None or frame.size == 0:
            return None
        cv2.imwrite(str(output_path), frame)
        return output_path
    finally:
        capture.release()


def write_user_report(summary: dict[str, object], run_dir: Path) -> Path:
    video_alert = bool(summary.get("video_alert"))
    report_prefix = "Incidente_Hurto" if video_alert else "Revision_Sin_Alerta"
    report_path = run_dir / f"{report_prefix}_{time.strftime('%Y-%m-%d_%H-%M-%S')}.html"
    top_events = summary.get("top_events", [])
    top_event = top_events[0] if isinstance(top_events, list) and top_events else None

    if isinstance(top_event, dict):
        event_time = f"{format_seconds(float(top_event['start_sec']))} - {format_seconds(float(top_event['end_sec']))}"
        event_level = suspicion_level(float(top_event["cnn_prob_hurto"]))
        event_probability = f"{float(top_event['cnn_prob_hurto']) * 100:.1f}%"
        event_text = (
            f"Persona/track {int(top_event['track_id'])} entre {event_time}. "
            f"Nivel de sospecha: {event_level}. Probabilidad hurto: {event_probability}."
        )
    else:
        event_time = "Sin alerta"
        event_level = "N/A"
        event_probability = "N/A"
        event_text = "No se registraron eventos sospechosos sobre la sensibilidad seleccionada."

    evidence_path = str(
        summary.get("mosaic_evidence_path")
        or summary.get("best_evidence_path")
        or summary.get("incident_evidence_path")
        or ""
    )
    evidence_uri = Path(evidence_path).resolve().as_uri() if evidence_path else ""
    annotated_video_path = str(summary.get("annotated_video_path", ""))
    annotated_video_uri = Path(annotated_video_path).resolve().as_uri() if annotated_video_path else ""
    source_video_path = str(summary.get("video_path", ""))

    escaped_event_text = html.escape(event_text)
    escaped_source_video_path = html.escape(source_video_path)
    escaped_annotated_video_path = html.escape(annotated_video_path)
    escaped_evidence_path = html.escape(evidence_path)
    escaped_event_time = html.escape(event_time)
    escaped_event_level = html.escape(event_level)
    escaped_event_probability = html.escape(event_probability)
    escaped_threshold = html.escape(f"{float(summary.get('alert_threshold', 0.0)):.2f}")
    escaped_run_dir = html.escape(str(summary.get("run_dir", "")))

    save_links = []
    if annotated_video_uri:
        save_links.append(
            f'<a class="button" href="{html.escape(annotated_video_uri)}" download>Guardar video anotado</a>'
        )
    if evidence_uri:
        save_links.append(
            f'<a class="button" href="{html.escape(evidence_uri)}" download>Guardar mosaico</a>'
        )
    save_links_html = "\n    ".join(save_links) if save_links else "<span>No hay archivos de evidencia para guardar.</span>"
    evidence_html = (
        f'<h2>Evidencia</h2><p>Mosaico temporal usado por el modelo para generar la alerta.</p>'
        f'<img src="{html.escape(evidence_uri)}" alt="Mosaico de evidencia">'
        if evidence_uri
        else ""
    )

    html_content = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <title>{report_prefix}</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #111827; line-height: 1.45; }}
    h1 {{ margin-bottom: 4px; }}
    h2 {{ margin-top: 28px; }}
    .status {{ padding: 12px 16px; border-radius: 8px; color: white; background: {'#991b1b' if video_alert else '#14532d'}; }}
    .grid {{ display: grid; grid-template-columns: 190px 1fr; gap: 8px 16px; margin-top: 24px; }}
    img {{ max-width: 620px; width: 100%; image-rendering: auto; border: 1px solid #d1d5db; margin-top: 12px; }}
    a {{ color: #1d4ed8; }}
    .actions {{ display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }}
    .button {{ display: inline-block; padding: 10px 14px; border-radius: 6px; background: #111827; color: white; text-decoration: none; }}
    .note {{ color: #4b5563; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>{'Informe de incidente' if video_alert else 'Informe de revision'}</h1>
  <p>{time.strftime('%Y-%m-%d %H:%M:%S')}</p>
  <div class="status">{'Posible hurto detectado' if video_alert else 'No se detecto alerta'}</div>
  <div class="grid">
    <strong>Video revisado</strong><span>{escaped_source_video_path}</span>
    <strong>Resultado</strong><span>{escaped_event_text}</span>
    <strong>Momento</strong><span>{escaped_event_time}</span>
    <strong>Nivel</strong><span>{escaped_event_level}</span>
    <strong>Probabilidad hurto</strong><span>{escaped_event_probability}</span>
    <strong>Umbral usado</strong><span>{escaped_threshold}</span>
    <strong>Video anotado</strong><span><a href="{html.escape(annotated_video_uri)}">{escaped_annotated_video_path}</a></span>
    <strong>Mosaico</strong><span>{escaped_evidence_path}</span>
    <strong>Carpeta salida</strong><span>{escaped_run_dir}</span>
  </div>
  {evidence_html}
  <h2>Guardar archivos</h2>
  <p class="note">Opcional: usa estos enlaces para guardar el video anotado y el mosaico en una ubicacion del equipo.</p>
  <div class="actions">
    {save_links_html}
  </div>
</body>
</html>
"""
    report_path.write_text(html_content, encoding="utf-8")
    return report_path



class TheftDetectionApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("EXIMERKAR'S - Detección de Hurto")
        self.root.geometry("1180x760")
        self.root.minsize(1080, 700)

        self.video_path_var = StringVar()
        self.sensitivity_var = StringVar(value="Media")
        self.debug_var = BooleanVar(value=False)
        self.status_var = StringVar(value="Selecciona un video y ejecuta el análisis.")
        self.summary_var = StringVar(value="Sin resultados todavía.")

        self.progress_stage_var = StringVar(value="En espera")
        self.progress_detail_var = StringVar(value="Selecciona un video para comenzar.")
        self.progress_percent_var = StringVar(value="0%")
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.last_result: dict[str, object] | None = None
        self.evidence_window: Toplevel | None = None
        self.evidence_photo: ImageTk.PhotoImage | None = None
        self.advanced_visible = False
        self.alert_sound_stop_event = threading.Event()
        self.alert_sound_thread: threading.Thread | None = None
        self._model_cache_key: tuple[str, float, int] | None = None
        self._model_cache: tuple[
            YOLO,
            torch.nn.Module,
            transforms.Compose,
            dict[int, str],
            torch.device,
        ] | None = None

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(150, self._poll_worker_queue)

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=14)
        root_frame.pack(fill=BOTH, expand=True)

        controls_frame = ttk.LabelFrame(root_frame, text="Revision de video", padding=12)
        controls_frame.pack(fill="x")

        ttk.Label(controls_frame, text="Video").grid(row=0, column=0, sticky=W)
        video_entry = ttk.Entry(controls_frame, textvariable=self.video_path_var, width=88)
        video_entry.grid(row=0, column=1, padx=(8, 8), sticky="ew")

        select_button = ttk.Button(controls_frame, text="Seleccionar video", command=self._select_video)
        select_button.grid(row=0, column=2, padx=(0, 8))

        ttk.Label(controls_frame, text="Sensibilidad").grid(row=1, column=0, pady=(12, 0), sticky=W)
        sensitivity_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.sensitivity_var,
            values=list(SENSITIVITY_LEVELS.keys()),
            width=14,
            state="readonly",
        )
        sensitivity_combo.grid(row=1, column=1, pady=(12, 0), sticky=W)

        self.analyze_button = ttk.Button(controls_frame, text="Analizar", command=self._start_analysis)
        self.analyze_button.grid(row=1, column=2, padx=(0, 8), pady=(12, 0), sticky="e")

        self.open_result_button = ttk.Button(
            controls_frame,
            text="Ver evidencia",
            command=self._open_evidence_view,
            state="disabled",
        )
        self.open_result_button.grid(row=1, column=3, pady=(12, 0), sticky="e")

        controls_frame.columnconfigure(1, weight=1)
        self.root.bind("<Control-d>", self._toggle_advanced_panel)

        status_frame = ttk.Frame(root_frame, padding=(0, 12, 0, 12))
        status_frame.pack(fill="x")
        ttk.Label(status_frame, textvariable=self.status_var).pack(anchor=W)
        ttk.Label(status_frame, textvariable=self.summary_var, font=("Segoe UI", 11, "bold")).pack(anchor=W, pady=(6, 0))

        progress_frame = ttk.LabelFrame(root_frame, text="Estado del proceso", padding=10)
        progress_frame.pack(fill="x", pady=(0, 12))

        progress_header = ttk.Frame(progress_frame)
        progress_header.pack(fill="x")
        ttk.Label(progress_header, textvariable=self.progress_stage_var, font=("Segoe UI", 10, "bold")).pack(side=LEFT)
        ttk.Label(progress_header, textvariable=self.progress_percent_var).pack(side=RIGHT)

        self.progress_bar = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progress_bar.pack(fill="x", pady=(8, 6))
        ttk.Label(progress_frame, textvariable=self.progress_detail_var, wraplength=1020, justify="left").pack(anchor=W)

        self.alert_box = Frame(
            root_frame,
            bg="#1f2937",
            highlightthickness=2,
            highlightbackground="#1f2937",
            padx=14,
            pady=12,
        )
        self.alert_box.pack(fill="x", pady=(0, 12))

        self.alert_title_label = Label(
            self.alert_box,
            text="Sistema listo",
            bg="#1f2937",
            fg="#f8fafc",
            font=("Segoe UI", 14, "bold"),
            anchor="w",
        )
        self.alert_title_label.pack(fill="x", anchor=W)

        self.alert_detail_label = Label(
            self.alert_box,
            text="La alerta aparecera aqui cuando el modelo detecte una escena sospechosa.",
            bg="#1f2937",
            fg="#e5e7eb",
            justify="left",
            anchor="w",
            wraplength=1020,
        )
        self.alert_detail_label.pack(fill="x", anchor=W, pady=(6, 0))

        self.alert_actions_frame = Frame(self.alert_box, bg="#1f2937")
        self.alert_actions_frame.pack(fill="x", pady=(10, 0))
        self.stop_sound_button = ttk.Button(
            self.alert_actions_frame,
            text="Detener sonido",
            command=self._stop_alert_sound,
            state="disabled",
        )
        self.stop_sound_button.pack(anchor=W)

        timeline_frame = ttk.LabelFrame(root_frame, text="Linea de tiempo", padding=10)
        timeline_frame.pack(fill=BOTH, expand=True)

        self.events_tree = ttk.Treeview(
            timeline_frame,
            columns=("persona", "momento", "nivel"),
            show="headings",
            height=14,
        )
        self.events_tree.heading("persona", text="Persona")
        self.events_tree.heading("momento", text="Momento")
        self.events_tree.heading("nivel", text="Nivel")
        self.events_tree.column("persona", width=120, anchor="center")
        self.events_tree.column("momento", width=180, anchor="center")
        self.events_tree.column("nivel", width=140, anchor="center")
        self.events_tree.tag_configure("Alta", background="#fee2e2", foreground="#991b1b")
        self.events_tree.tag_configure("Media", background="#fef3c7", foreground="#92400e")
        self.events_tree.tag_configure("Baja", background="#dcfce7", foreground="#14532d")
        self.events_tree.pack(fill=BOTH, expand=True)

        self.advanced_frame = ttk.LabelFrame(root_frame, text="Configuracion avanzada", padding=10)
        self.debug_check = ttk.Checkbutton(
            self.advanced_frame,
            text="Mostrar registro tecnico",
            variable=self.debug_var,
        )
        self.debug_check.pack(anchor=W)
        self.log_text = ScrolledText(self.advanced_frame, height=8, wrap="word")
        self.log_text.pack(fill=BOTH, expand=True, pady=(8, 0))

        self._update_progress(0.0, "En espera", "Selecciona un video para comenzar.")
        self._set_alert_panel("idle")

    def _select_video(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Seleccionar video",
            filetypes=[
                ("Videos", "*.mp4 *.avi *.mov *.mkv"),
                ("Todos los archivos", "*.*"),
            ],
        )
        if file_path:
            self.video_path_var.set(file_path)
            selected_video_path = Path(file_path)
            self._clear_previous_result()
            self.status_var.set(f"Video cargado: {selected_video_path.name}")
            self.summary_var.set("Listo para ejecutar el analisis.")
            self._set_alert_panel(
                "idle",
                detail=f"Video seleccionado: {selected_video_path.name}. Ejecuta el analisis para validar posibles hurtos.",
            )

    def _get_models(
        self,
        settings: AppSettings,
        log: LogCallback,
    ) -> tuple[YOLO, torch.nn.Module, transforms.Compose, dict[int, str], torch.device]:
        device = resolve_device(settings.device_name)
        cache_key = (str(device), float(settings.cnn_dropout), int(settings.cnn_image_size))

        if self._model_cache is not None and self._model_cache_key == cache_key:
            log("Modelos reutilizados desde memoria.")
            return self._model_cache

        log("Cargando modelos en memoria.")
        yolo_model = load_yolo_model(YOLO_MODEL_PATH, log)
        classifier, eval_transform, index_to_class = load_classifier(
            checkpoint_path=CNN_CHECKPOINT_PATH,
            device=device,
            dropout=settings.cnn_dropout,
            image_size=settings.cnn_image_size,
        )
        self._model_cache_key = cache_key
        self._model_cache = (yolo_model, classifier, eval_transform, index_to_class, device)
        log("Modelos cargados y guardados para reutilizar durante esta sesion.")
        return self._model_cache

    def _start_analysis(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Análisis en curso", "Espera a que termine el procesamiento actual.")
            return

        video_path = Path(self.video_path_var.get().strip())
        if not video_path.exists():
            messagebox.showerror("Video no encontrado", "Selecciona un archivo de video válido.")
            return

        sensitivity_name = self.sensitivity_var.get()
        sensitivity_settings = SENSITIVITY_LEVELS.get(sensitivity_name, SENSITIVITY_LEVELS["Media"])
        alert_threshold = float(sensitivity_settings["threshold"])

        self._clear_previous_result()
        self.status_var.set("Procesando video. Esto puede tardar algunos minutos.")
        self.summary_var.set("Análisis en ejecución.")
        self.analyze_button.configure(state="disabled")
        self.open_result_button.configure(state="disabled")
        self.debug_check.configure(state="disabled")
        self._set_alert_panel(
            "running",
            detail=f"Procesando {video_path.name}. Sensibilidad: {sensitivity_name.lower()}.",
        )
        self._update_progress(0.0, "Preparando analisis", f"Inicializando procesamiento para {video_path.name}")
        self.events_tree.insert("", END, values=("Video", "En proceso", "Analizando"))

        settings = AppSettings(
            alert_threshold=alert_threshold,
            debug_mode=bool(self.debug_var.get()),
        )

        def worker() -> None:
            try:
                worker_log = lambda message: self.worker_queue.put(("log", message))
                worker_progress = lambda value, stage, detail: self.worker_queue.put(
                    (
                        "progress",
                        {
                            "value": float(value),
                            "stage": str(stage),
                            "detail": str(detail),
                        },
                    )
                )
                worker_debug = (
                    (lambda message: self.worker_queue.put(("debug", message)))
                    if settings.debug_mode
                    else None
                )
                worker_progress(0.01, "Cargando modelos", "Inicializando modelos si no estan en memoria")
                yolo_model, classifier, eval_transform, index_to_class, device = self._get_models(settings, worker_log)
                result = analyze_video(
                    video_path=video_path,
                    settings=settings,
                    yolo_model=yolo_model,
                    classifier=classifier,
                    eval_transform=eval_transform,
                    index_to_class=index_to_class,
                    device=device,
                    log=worker_log,
                    progress=worker_progress,
                    debug=worker_debug,
                )
                self.worker_queue.put(("done", result))
            except Exception as error:  # noqa: BLE001
                if settings.debug_mode:
                    self.worker_queue.put(("debug", traceback.format_exc()))
                self.worker_queue.put(("error", str(error)))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _poll_worker_queue(self) -> None:
        try:
            while True:
                event_name, payload = self.worker_queue.get_nowait()
                if event_name == "log":
                    self._append_log(str(payload))
                elif event_name == "debug":
                    self._append_log(str(payload), level="DEBUG")
                elif event_name == "progress":
                    progress_payload = payload if isinstance(payload, dict) else {}
                    self._update_progress(
                        float(progress_payload.get("value", 0.0)),
                        str(progress_payload.get("stage", "Procesando")),
                        str(progress_payload.get("detail", "")),
                    )
                elif event_name == "done":
                    self._handle_result(payload if isinstance(payload, dict) else {})
                elif event_name == "error":
                    self._handle_error(str(payload))
        except queue.Empty:
            pass

        self.root.after(150, self._poll_worker_queue)

    def _update_progress(self, value: float, stage: str, detail: str) -> None:
        bounded_value = max(0.0, min(1.0, float(value)))
        friendly_stage, friendly_detail = self._friendly_progress_text(stage, detail)
        self.progress_bar.configure(value=bounded_value * 100.0)
        self.progress_stage_var.set(friendly_stage)
        self.progress_detail_var.set(friendly_detail)
        self.progress_percent_var.set(f"{bounded_value * 100.0:5.1f}%")

    def _friendly_progress_text(self, stage: str, detail: str) -> tuple[str, str]:
        stage_lower = stage.lower()
        if "tracking" in stage_lower or "persona" in stage_lower:
            return "Buscando personas", "Identificando personas y siguiendo su movimiento."
        if "clasificando" in stage_lower or "ventana" in stage_lower:
            return "Analizando eventos", "Revisando las secuencias donde aparece cada persona."
        if "render" in stage_lower:
            return "Preparando evidencia", "Generando el video anotado para revision."
        if "modelo" in stage_lower:
            return "Preparando analisis", "Cargando los modelos de deteccion y clasificacion."
        if "error" in stage_lower:
            return "Error", detail or "No fue posible completar el analisis."
        if "complet" in stage_lower:
            return "Analisis completado", "La revision termino y los resultados estan disponibles."
        return stage or "Procesando", "Esperando para analizar."

    def _append_log(self, message: str, level: str = "INFO") -> None:
        prefix = f"[{level}] " if level else ""
        self.log_text.insert(END, f"{prefix}{message}\n")
        self.log_text.see(END)

    def _handle_result(self, result: dict[str, object]) -> None:
        self.last_result = result
        self.analyze_button.configure(state="normal")
        self.open_result_button.configure(state="normal")
        self.debug_check.configure(state="normal")
        self._update_progress(1.0, "Analisis completado", "Los resultados ya estan disponibles.")

        if bool(result.get("video_alert")):
            self.status_var.set("ALERTA: se detectó posible hurto en el video.")
        else:
            self.status_var.set("No se detectaron eventos sospechosos en el video.")

        self.summary_var.set(
            f"Personas revisadas: {result.get('num_tracks_clean', 0)} | "
            f"Eventos sospechosos: {result.get('num_positive_windows', 0)}"
        )

        for item_id in self.events_tree.get_children():
            self.events_tree.delete(item_id)

        top_events = result.get("top_events", [])
        for person_number, event in enumerate(top_events, start=1):
            if not isinstance(event, dict):
                continue
            interval_text = f"{format_seconds(float(event['start_sec']))} - {format_seconds(float(event['end_sec']))}"
            level_text = suspicion_level(float(event["cnn_prob_hurto"]))
            self.events_tree.insert(
                "",
                END,
                values=(f"Persona {person_number}", interval_text, level_text),
                tags=(level_text,),
            )
        if not top_events:
            self.events_tree.insert("", END, values=("Video", "Completo", "Sin alerta"), tags=("Baja",))

        if bool(result.get("video_alert")):
            top_event = result.get("top_events", [])[0] if result.get("top_events") else None
            if isinstance(top_event, dict):
                level_text = suspicion_level(float(top_event["cnn_prob_hurto"]))
                alert_detail = (
                    f"Persona observada entre {format_seconds(float(top_event['start_sec']))} y "
                    f"{format_seconds(float(top_event['end_sec']))}. Nivel de sospecha: {level_text.lower()}."
                )
            else:
                alert_detail = "Se detectaron eventos sospechosos en el video analizado."
            self._set_alert_panel("alert", detail=alert_detail)
            self._start_alert_sound()
        else:
            self._stop_alert_sound()
            self._set_alert_panel(
                "clear",
                detail="No se detectaron eventos sospechosos con la sensibilidad seleccionada.",
            )

    def _handle_error(self, error_message: str) -> None:
        self.analyze_button.configure(state="normal")
        self.open_result_button.configure(state="disabled")
        self.debug_check.configure(state="normal")
        self._update_progress(float(self.progress_bar["value"]) / 100.0, "Error", error_message)
        self.status_var.set("El análisis terminó con error.")
        self.summary_var.set("No se generaron resultados.")
        self._stop_alert_sound()
        self._set_alert_panel("error", detail=error_message)
        messagebox.showerror("Error durante el análisis", error_message)

    def _clear_previous_result(self) -> None:
        self.last_result = None
        self.log_text.delete("1.0", END)
        self._stop_alert_sound()
        if self.evidence_window is not None and self.evidence_window.winfo_exists():
            self.evidence_window.destroy()
        self.evidence_window = None
        self.evidence_photo = None
        self.open_result_button.configure(state="disabled")
        self._update_progress(0.0, "En espera", "Selecciona un video para comenzar.")
        for item_id in self.events_tree.get_children():
            self.events_tree.delete(item_id)

    def _open_evidence_view(self) -> None:
        if not self.last_result:
            return

        if self.evidence_window is not None and self.evidence_window.winfo_exists():
            self.evidence_window.lift()
            self.evidence_window.focus_force()
            return

        evidence_window = Toplevel(self.root)
        self.evidence_window = evidence_window
        evidence_window.title("Evidencia")
        evidence_window.geometry("560x480")
        evidence_window.minsize(420, 340)
        evidence_window.transient(self.root)
        evidence_window.protocol("WM_DELETE_WINDOW", self._close_evidence_window)

        canvas = tk.Canvas(evidence_window, highlightthickness=0)
        scrollbar = ttk.Scrollbar(evidence_window, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas, padding=16)

        inner.bind(
            "<Configure>",
            lambda _event: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=inner, anchor="nw", tags="evidence_inner")

        def on_canvas_configure(event: tk.Event) -> None:
            canvas.itemconfig("evidence_inner", width=event.width)

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.bind("<Configure>", on_canvas_configure)
        canvas.bindtags((canvas.bindtags() or ()) + ("TScrolledEvidence",))
        canvas.bind_class(
            "TScrolledEvidence",
            "<MouseWheel>",
            lambda _event: canvas.yview_scroll(int(-1 * (_event.delta / 120)), "units"),
        )

        canvas.pack(side=LEFT, fill=BOTH, expand=True)
        scrollbar.pack(side=RIGHT, fill="y")

        ttk.Label(inner, text="Evidencia", font=("Segoe UI", 16, "bold")).pack(anchor=W)
        ttk.Label(
            inner,
            text="Mosaico temporal usado por el modelo para generar la alerta.",
            wraplength=480,
            justify="left",
        ).pack(anchor=W, pady=(4, 12))

        mosaic_path = self._get_evidence_mosaic_path()
        if mosaic_path is not None and mosaic_path.exists():
            evidence_image = Image.open(mosaic_path)
            display_image = self._resize_image_for_evidence(evidence_image)
            self.evidence_photo = ImageTk.PhotoImage(display_image)
            ttk.Label(inner, image=self.evidence_photo).pack(anchor=W, pady=(0, 12))
        else:
            self.evidence_photo = None
            ttk.Label(
                inner,
                text="No hay mosaico de alerta disponible para el ultimo analisis.",
                wraplength=480,
                justify="left",
            ).pack(anchor=W, pady=(0, 12))

        details_text = ScrolledText(inner, height=8, wrap="word")
        details_text.insert("1.0", self._build_evidence_details_text())
        details_text.configure(state="disabled")
        details_text.pack(fill=BOTH, expand=True, pady=(0, 12))

        actions_frame = ttk.Frame(inner)
        actions_frame.pack(fill="x")
        ttk.Button(
            actions_frame,
            text="Guardar video y mosaico...",
            command=self._save_evidence_files,
        ).pack(side=LEFT)
        ttk.Button(
            actions_frame,
            text="Abrir video anotado",
            command=self._open_last_annotated_video,
        ).pack(side=LEFT, padx=(8, 0))
        ttk.Button(
            actions_frame,
            text="Abrir informe",
            command=self._open_run_folder,
        ).pack(side=LEFT, padx=(8, 0))
        ttk.Button(
            actions_frame,
            text="Cerrar",
            command=self._close_evidence_window,
        ).pack(side=RIGHT)

        self._center_window(evidence_window)

    def _close_evidence_window(self) -> None:
        if self.evidence_window is not None and self.evidence_window.winfo_exists():
            self.evidence_window.destroy()
        self.evidence_window = None
        self.evidence_photo = None

    def _center_window(self, window: Toplevel) -> None:
        window.update_idletasks()
        root_x = self.root.winfo_x()
        root_y = self.root.winfo_y()
        root_w = self.root.winfo_width()
        root_h = self.root.winfo_height()
        win_w = window.winfo_width()
        win_h = window.winfo_height()
        x = root_x + (root_w - win_w) // 2
        y = root_y + (root_h - win_h) // 2
        window.geometry(f"+{x}+{y}")

    def _get_evidence_mosaic_path(self) -> Path | None:
        if not self.last_result:
            return None

        path_value = (
            self.last_result.get("mosaic_evidence_path")
            or self.last_result.get("best_evidence_path")
            or self.last_result.get("incident_evidence_path")
            or ""
        )
        return Path(str(path_value)) if path_value else None

    def _get_last_annotated_video_path(self) -> Path | None:
        if not self.last_result:
            return None

        path_value = str(self.last_result.get("annotated_video_path", ""))
        return Path(path_value) if path_value else None

    def _get_last_report_path(self) -> Path | None:
        if not self.last_result:
            return None

        path_value = str(self.last_result.get("incident_report_path", ""))
        return Path(path_value) if path_value else None

    def _resize_image_for_evidence(self, image: Image.Image) -> Image.Image:
        display_image = image.copy()
        max_width = 620
        max_height = 360
        width, height = display_image.size
        if width <= 0 or height <= 0:
            return display_image

        scale = min(max_width / width, max_height / height)
        if scale >= 1.0:
            resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
            return display_image.resize(resized_size, Image.Resampling.NEAREST)

        display_image.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        return display_image

    def _open_last_annotated_video(self) -> None:
        annotated_video_path = self._get_last_annotated_video_path()
        if annotated_video_path is None or not annotated_video_path.exists():
            messagebox.showerror("Archivo no disponible", "No existe el video anotado del ultimo analisis.")
            return

        try:
            open_path(annotated_video_path)
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("No fue posible abrir el archivo", str(error))

    def _save_evidence_files(self) -> None:
        if not self.last_result:
            return

        target_dir_value = filedialog.askdirectory(title="Guardar evidencia")
        if not target_dir_value:
            return

        target_dir = Path(target_dir_value)
        copied_paths: list[Path] = []

        for source_path in [self._get_last_annotated_video_path(), self._get_evidence_mosaic_path()]:
            if source_path is None or not source_path.exists():
                continue
            destination_path = self._unique_destination(target_dir / source_path.name)
            shutil.copy2(source_path, destination_path)
            copied_paths.append(destination_path)

        details_path = self._unique_destination(target_dir / "detalles_evidencia.txt")
        details_path.write_text(self._build_evidence_details_text(), encoding="utf-8")
        copied_paths.append(details_path)

        copied_text = "\n".join(str(path) for path in copied_paths)
        messagebox.showinfo("Evidencia guardada", f"Archivos guardados:\n{copied_text}")

    def _unique_destination(self, destination_path: Path) -> Path:
        if not destination_path.exists():
            return destination_path

        stem = destination_path.stem
        suffix = destination_path.suffix
        parent = destination_path.parent
        counter = 1
        while True:
            candidate_path = parent / f"{stem}_{counter}{suffix}"
            if not candidate_path.exists():
                return candidate_path
            counter += 1

    def _build_evidence_details_text(self) -> str:
        result = self.last_result or {}
        top_events = result.get("top_events", [])
        top_event = top_events[0] if isinstance(top_events, list) and top_events else None
        mosaic_path = self._get_evidence_mosaic_path()
        annotated_video_path = self._get_last_annotated_video_path()
        report_path = self._get_last_report_path()

        lines = [
            "Informe de evidencia",
            f"Fecha: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Video original: {result.get('video_path', '')}",
            f"Video anotado: {annotated_video_path or ''}",
            f"Mosaico de alerta: {mosaic_path or ''}",
            f"Informe: {report_path or ''}",
            f"Personas revisadas: {result.get('num_tracks_clean', 0)}",
            f"Eventos sospechosos: {result.get('num_positive_windows', 0)}",
            f"Umbral usado: {float(result.get('alert_threshold', 0.0)):.2f}",
        ]

        if isinstance(top_event, dict):
            level_text = suspicion_level(float(top_event["cnn_prob_hurto"]))
            probability_text = f"{float(top_event['cnn_prob_hurto']) * 100:.1f}%"
            interval_text = (
                f"{format_seconds(float(top_event['start_sec']))} - "
                f"{format_seconds(float(top_event['end_sec']))}"
            )
            lines.extend(
                [
                    "",
                    "Detalle del evento principal",
                    f"Track/persona: {int(top_event['track_id'])}",
                    f"Momento: {interval_text}",
                    f"Nivel de sospecha: {level_text}",
                    f"Probabilidad hurto: {probability_text}",
                    f"Ventana: {top_event.get('sample_id', '')}",
                ]
            )
        else:
            lines.extend(["", "Detalle del evento principal", "No se detecto alerta."])

        return "\n".join(lines)

    def _open_annotated_video(self) -> None:
        if not self.last_result:
            return

        annotated_video_path = Path(str(self.last_result.get("annotated_video_path", "")))
        if not annotated_video_path.exists():
            messagebox.showerror("Archivo no disponible", "No existe el video anotado del último análisis.")
            return

        try:
            open_path(annotated_video_path)
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("No fue posible abrir el archivo", str(error))

    def _open_run_folder(self) -> None:
        if not self.last_result:
            return

        report_path_value = str(self.last_result.get("incident_report_path", ""))
        report_path = Path(report_path_value) if report_path_value else None
        if report_path is not None and report_path.exists():
            try:
                open_path(report_path)
            except Exception as error:  # noqa: BLE001
                messagebox.showerror("No fue posible abrir el informe", str(error))
            return

        run_dir = Path(str(self.last_result.get("run_dir", "")))
        if not run_dir.exists():
            messagebox.showerror("Carpeta no disponible", "No existe la carpeta del ultimo analisis.")
            return

        try:
            open_path(run_dir)
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("No fue posible abrir la carpeta", str(error))

    def _toggle_advanced_panel(self, _event: object | None = None) -> None:
        if self.advanced_visible:
            self.advanced_frame.pack_forget()
            self.advanced_visible = False
            self.debug_var.set(False)
            return

        self.advanced_frame.pack(fill=BOTH, expand=False, pady=(12, 0))
        self.advanced_visible = True

    def _set_alert_panel(self, state: str, detail: str | None = None) -> None:
        palette = {
            "idle": {
                "bg": "#1f2937",
                "fg": "#f8fafc",
                "detail_fg": "#e5e7eb",
                "title": "Sistema listo",
                "detail": "La alerta aparecera aqui cuando se detecte una escena sospechosa.",
            },
            "running": {
                "bg": "#92400e",
                "fg": "#fff7ed",
                "detail_fg": "#ffedd5",
                "title": "Analizando video",
                "detail": "El modelo esta procesando el archivo seleccionado.",
            },
            "clear": {
                "bg": "#14532d",
                "fg": "#ecfdf5",
                "detail_fg": "#d1fae5",
                "title": "Sin alerta",
                "detail": "No se detectaron eventos sospechosos con la sensibilidad configurada.",
            },
            "alert": {
                "bg": "#991b1b",
                "fg": "#fef2f2",
                "detail_fg": "#fecaca",
                "title": "ALERTA DE HURTO",
                "detail": "Se detecto una escena sospechosa.",
            },
            "error": {
                "bg": "#7c2d12",
                "fg": "#fff7ed",
                "detail_fg": "#fed7aa",
                "title": "Error en el analisis",
                "detail": "No fue posible completar el procesamiento.",
            },
        }
        selected_palette = palette.get(state, palette["idle"])
        alert_detail = detail or selected_palette["detail"]

        self.alert_box.configure(
            bg=selected_palette["bg"],
            highlightbackground=selected_palette["bg"],
            highlightcolor=selected_palette["bg"],
        )
        self.alert_title_label.configure(
            text=selected_palette["title"],
            bg=selected_palette["bg"],
            fg=selected_palette["fg"],
        )
        self.alert_detail_label.configure(
            text=alert_detail,
            bg=selected_palette["bg"],
            fg=selected_palette["detail_fg"],
        )

    def _start_alert_sound(self) -> None:
        if winsound is None:
            return

        self._stop_alert_sound()
        self.alert_sound_stop_event.clear()
        self.stop_sound_button.configure(state="normal")

        def alert_loop() -> None:
            start_time = time.monotonic()
            duration_s = 7.0
            while time.monotonic() - start_time < duration_s:
                if self.alert_sound_stop_event.is_set():
                    break
                winsound.Beep(1000, 300)
                if self.alert_sound_stop_event.wait(0.4):
                    break

            self.root.after(0, self._stop_alert_sound)

        self.alert_sound_thread = threading.Thread(target=alert_loop, daemon=True)
        self.alert_sound_thread.start()

    def _stop_alert_sound(self) -> None:
        self.alert_sound_stop_event.set()
        self.alert_sound_thread = None
        self.stop_sound_button.configure(state="disabled")

    def _on_close(self) -> None:
        self._stop_alert_sound()
        if self.evidence_window is not None and self.evidence_window.winfo_exists():
            self.evidence_window.destroy()
        self.root.destroy()


def main() -> None:
    root = Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = TheftDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
