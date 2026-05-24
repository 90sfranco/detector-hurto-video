from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageTk
from tkinter import BOTH, END, LEFT, RIGHT, W, filedialog, messagebox, ttk
from tkinter import BooleanVar, Frame, Label, StringVar, Tk
from tkinter.scrolledtext import ScrolledText
from ultralytics import YOLO
from ultralytics.utils.downloads import attempt_download_asset
from torchvision import transforms
from torchvision.models import resnet50
from torchvision.transforms import InterpolationMode


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


@dataclass(frozen=True)
class AppSettings:
    alert_threshold: float = 0.60
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
    max_frame_cache_size: int = 64


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


def read_frame_by_index(
    capture: cv2.VideoCapture,
    frame_idx: int,
    frame_cache: OrderedDict[int, np.ndarray],
    max_cache_size: int,
) -> np.ndarray | None:
    if frame_idx in frame_cache:
        frame = frame_cache.pop(frame_idx)
        frame_cache[frame_idx] = frame
        return frame

    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = capture.read()
    if not ok or frame is None:
        return None

    frame_cache[frame_idx] = frame
    if len(frame_cache) > max_cache_size:
        frame_cache.popitem(last=False)
    return frame


def recover_window_crops(
    capture: cv2.VideoCapture,
    sampled_rows_df: pd.DataFrame,
    settings: AppSettings,
    frame_cache: OrderedDict[int, np.ndarray],
) -> tuple[list[np.ndarray], list[int], list[float]]:
    crops: list[np.ndarray] = []
    used_frame_indices: list[int] = []
    used_timestamps: list[float] = []

    for _, track_row in sampled_rows_df.iterrows():
        frame_idx = int(track_row["frame_idx"])
        frame = read_frame_by_index(capture, frame_idx, frame_cache, settings.max_frame_cache_size)
        if frame is None:
            continue

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
            continue

        crop = frame[padded_y1:padded_y2, padded_x1:padded_x2]
        if crop is None or crop.size == 0 or crop.shape[0] <= 1 or crop.shape[1] <= 1:
            continue

        crops.append(crop)
        used_frame_indices.append(frame_idx)
        used_timestamps.append(round(float(track_row["timestamp_sec"]), 6))

    return crops, used_frame_indices, used_timestamps


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

    track_windows: list[tuple[int, list[dict[str, object]]]] = []
    total_windows = 0
    for track_id, track_df in filtered_tracks_df.groupby("track_id"):
        windows = build_track_windows(track_df, settings)
        track_windows.append((int(track_id), windows))
        total_windows += len(windows)

    emit_debug(
        debug,
        f"Clasificacion: tracks validos={len(track_windows)}, ventanas totales={total_windows}, dispositivo={device}",
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video para clasificar ventanas: {video_path}")

    frame_cache: OrderedDict[int, np.ndarray] = OrderedDict()
    predictions: list[dict[str, object]] = []
    best_positive_probability = -1.0
    best_evidence_path: Path | None = None
    video_id = video_path.stem
    processed_windows = 0

    try:
        for track_id, windows in track_windows:
            emit_debug(debug, f"Clasificando track {track_id} con {len(windows)} ventanas.")
            for window_spec in windows:
                sampled_rows_df = window_spec["sampled_rows_df"]
                sample_id = f"{video_id}__t{int(track_id)}__{window_spec['window_id']}"

                crops, used_frame_indices, used_timestamps = recover_window_crops(
                    capture=capture,
                    sampled_rows_df=sampled_rows_df,
                    settings=settings,
                    frame_cache=frame_cache,
                )

                valid_sample = len(crops) >= settings.min_valid_frames_per_window
                positive_probability = 0.0
                negative_probability = 0.0
                predicted_label = NEGATIVE_LABEL

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

                predictions.append(
                    {
                        "sample_id": sample_id,
                        "track_id": int(track_id),
                        "window_id": str(window_spec["window_id"]),
                        "start_frame": int(window_spec["start_frame"]),
                        "end_frame": int(window_spec["end_frame"]),
                        "start_sec": round(float(window_spec["start_sec"]), 6),
                        "end_sec": round(float(window_spec["end_sec"]), 6),
                        "num_frames_used": int(len(crops)),
                        "valid_sample": bool(valid_sample),
                        "sampled_frame_indices": json.dumps(used_frame_indices),
                        "sampled_timestamps": json.dumps(used_timestamps),
                        "cnn_prob_normal": round(float(negative_probability), 6),
                        "cnn_prob_hurto": round(float(positive_probability), 6),
                        "predicted_label": str(predicted_label),
                        "alert_positive": bool(positive_probability >= settings.alert_threshold),
                    }
                )
                processed_windows += 1
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
                            f"Ventana {sample_id}: valid_sample={valid_sample}, "
                            f"prob_hurto={positive_probability:.4f}, prob_normal={negative_probability:.4f}"
                        ),
                    )
    finally:
        capture.release()

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
                    label = f"Posible hurto {positive_probability:.2f}"
                else:
                    color = (0, 180, 0)
                    label = f"Persona {track_id}"

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
            cv2.rectangle(frame, (0, 0), (frame.shape[1], 45), (0, 0, 180), -1)
            cv2.putText(
                frame,
                "ALERTA: posible hurto detectado",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
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
    log: LogCallback,
    progress: ProgressCallback | None = None,
    debug: LogCallback | None = None,
) -> dict[str, object]:
    run_dir = build_run_dir(video_path)
    device = resolve_device(settings.device_name)

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

    emit_progress(progress, 0.08, "Cargando modelos", "Inicializando detector y clasificador")
    yolo_model = load_yolo_model(YOLO_MODEL_PATH, log)
    emit_progress(progress, 0.14, "Cargando modelos", "Detector YOLO listo")
    classifier, eval_transform, index_to_class = load_classifier(
        checkpoint_path=CNN_CHECKPOINT_PATH,
        device=device,
        dropout=settings.cnn_dropout,
        image_size=settings.cnn_image_size,
    )
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
                    "start_sec",
                    "end_sec",
                    "cnn_prob_hurto",
                ]
            ]
            .to_dict(orient="records")
        )

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
        "models_used": {
            "person_detector_path": str(YOLO_MODEL_PATH),
            "person_detector_type": "pretrained_yolo_person_detector",
            "classifier_checkpoint_path": str(CNN_CHECKPOINT_PATH),
            "classifier_type": "trained_resnet50_classifier",
        },
        "debug_mode": bool(settings.debug_mode),
        "top_events": top_events,
    }

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


def load_video_preview_image(video_path: Path) -> Image.Image:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"No se pudo abrir el video para generar la vista previa: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    candidate_indices = [0]
    if frame_count > 1:
        candidate_indices.append(min(frame_count - 1, max(0, frame_count // 3)))
    if frame_count > 2:
        candidate_indices.append(min(frame_count - 1, max(0, frame_count // 2)))

    selected_frame: np.ndarray | None = None
    try:
        for frame_idx in candidate_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok or frame is None or frame.size == 0:
                continue

            selected_frame = frame
            if float(frame.mean()) > 5.0:
                break
    finally:
        capture.release()

    if selected_frame is None:
        raise RuntimeError(f"No fue posible leer frames validos de {video_path}")

    frame_rgb = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


class TheftDetectionApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("EXIMERKAR'S - Detección de Hurto")
        self.root.geometry("1180x760")
        self.root.minsize(1080, 700)

        self.video_path_var = StringVar()
        self.threshold_var = StringVar(value="0.60")
        self.debug_var = BooleanVar(value=True)
        self.preview_caption_var = StringVar(value="Vista previa")
        self.preview_detail_var = StringVar(
            value="Selecciona un video para revisar la escena antes de ejecutar el analisis."
        )
        self.status_var = StringVar(value="Selecciona un video y ejecuta el análisis.")
        self.summary_var = StringVar(value="Sin resultados todavía.")

        self.progress_stage_var = StringVar(value="En espera")
        self.progress_detail_var = StringVar(value="Selecciona un video para comenzar.")
        self.progress_percent_var = StringVar(value="0%")
        self.worker_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker_thread: threading.Thread | None = None
        self.last_result: dict[str, object] | None = None
        self.preview_photo: ImageTk.PhotoImage | None = None

        self._build_ui()
        self.root.after(150, self._poll_worker_queue)

    def _build_ui(self) -> None:
        root_frame = ttk.Frame(self.root, padding=14)
        root_frame.pack(fill=BOTH, expand=True)

        controls_frame = ttk.LabelFrame(root_frame, text="Entrada", padding=12)
        controls_frame.pack(fill="x")

        ttk.Label(controls_frame, text="Video").grid(row=0, column=0, sticky=W)
        video_entry = ttk.Entry(controls_frame, textvariable=self.video_path_var, width=88)
        video_entry.grid(row=0, column=1, padx=(8, 8), sticky="ew")

        select_button = ttk.Button(controls_frame, text="Seleccionar video", command=self._select_video)
        select_button.grid(row=0, column=2, padx=(0, 8))

        ttk.Label(controls_frame, text="Umbral hurto").grid(row=1, column=0, pady=(12, 0), sticky=W)
        threshold_entry = ttk.Entry(controls_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.grid(row=1, column=1, pady=(12, 0), sticky=W)

        self.analyze_button = ttk.Button(controls_frame, text="Analizar video", command=self._start_analysis)
        self.analyze_button.grid(row=1, column=2, padx=(0, 8), pady=(12, 0), sticky="e")

        self.open_result_button = ttk.Button(
            controls_frame,
            text="Abrir video anotado",
            command=self._open_annotated_video,
            state="disabled",
        )
        self.open_result_button.grid(row=1, column=3, pady=(12, 0), sticky="e")

        self.open_folder_button = ttk.Button(
            controls_frame,
            text="Abrir carpeta",
            command=self._open_run_folder,
            state="disabled",
        )
        self.open_folder_button.grid(row=1, column=4, pady=(12, 0), sticky="e")

        self.debug_check = ttk.Checkbutton(
            controls_frame,
            text="Modo debug",
            variable=self.debug_var,
        )
        self.debug_check.grid(row=1, column=5, padx=(12, 0), pady=(12, 0), sticky="w")

        controls_frame.columnconfigure(1, weight=1)

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

        content_frame = ttk.Frame(root_frame)
        content_frame.pack(fill=BOTH, expand=True)

        left_frame = ttk.LabelFrame(content_frame, text="Eventos detectados", padding=10)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True, padx=(0, 8))

        self.events_tree = ttk.Treeview(
            left_frame,
            columns=("track_id", "intervalo", "probabilidad"),
            show="headings",
            height=14,
        )
        self.events_tree.heading("track_id", text="Track")
        self.events_tree.heading("intervalo", text="Intervalo (s)")
        self.events_tree.heading("probabilidad", text="Prob. hurto")
        self.events_tree.column("track_id", width=80, anchor="center")
        self.events_tree.column("intervalo", width=180, anchor="center")
        self.events_tree.column("probabilidad", width=120, anchor="center")
        self.events_tree.pack(fill=BOTH, expand=True)

        logs_frame = ttk.LabelFrame(left_frame, text="Registro", padding=8)
        logs_frame.pack(fill=BOTH, expand=True, pady=(10, 0))

        self.log_text = ScrolledText(logs_frame, height=15, wrap="word")
        self.log_text.pack(fill=BOTH, expand=True)

        right_frame = ttk.LabelFrame(content_frame, text="Vista y evidencia", padding=10)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        ttk.Label(right_frame, textvariable=self.preview_caption_var, font=("Segoe UI", 11, "bold")).pack(anchor=W)

        self.preview_label = ttk.Label(
            right_frame,
            text="La mejor evidencia positiva aparecerá aquí.\nSi no hay alerta, se mostrará el resumen en texto.",
            anchor="center",
            justify="center",
        )
        self.preview_label.pack(fill=BOTH, expand=True, pady=(10, 0))
        ttk.Label(
            right_frame,
            textvariable=self.preview_detail_var,
            wraplength=460,
            justify="left",
        ).pack(fill="x", pady=(10, 0))
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
            self._show_video_preview(
                selected_video_path,
                caption="Vista previa del video seleccionado",
                detail=f"Archivo listo: {selected_video_path}",
            )

    def _start_analysis(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            messagebox.showinfo("Análisis en curso", "Espera a que termine el procesamiento actual.")
            return

        video_path = Path(self.video_path_var.get().strip())
        if not video_path.exists():
            messagebox.showerror("Video no encontrado", "Selecciona un archivo de video válido.")
            return

        try:
            alert_threshold = float(self.threshold_var.get().strip())
        except ValueError:
            messagebox.showerror("Umbral inválido", "El umbral debe ser un número decimal.")
            return

        if not 0.0 <= alert_threshold <= 1.0:
            messagebox.showerror("Umbral inválido", "El umbral debe estar entre 0 y 1.")
            return

        self._clear_previous_result()
        self.status_var.set("Procesando video. Esto puede tardar algunos minutos.")
        self.summary_var.set("Análisis en ejecución.")
        self.analyze_button.configure(state="disabled")
        self.open_result_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.debug_check.configure(state="disabled")
        self._set_alert_panel(
            "running",
            detail=f"Procesando {video_path.name}. El modelo esta revisando personas, tracks y ventanas temporales.",
        )
        self.preview_detail_var.set("Analisis en curso. La evidencia o la vista previa se actualizara al finalizar.")
        self._update_progress(0.0, "Preparando analisis", f"Inicializando procesamiento para {video_path.name}")

        settings = AppSettings(
            alert_threshold=alert_threshold,
            debug_mode=bool(self.debug_var.get()),
        )

        def worker() -> None:
            try:
                result = analyze_video(
                    video_path=video_path,
                    settings=settings,
                    log=lambda message: self.worker_queue.put(("log", message)),
                    progress=lambda value, stage, detail: self.worker_queue.put(
                        (
                            "progress",
                            {
                                "value": float(value),
                                "stage": str(stage),
                                "detail": str(detail),
                            },
                        )
                    ),
                    debug=(
                        (lambda message: self.worker_queue.put(("debug", message)))
                        if settings.debug_mode
                        else None
                    ),
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
        self.progress_bar.configure(value=bounded_value * 100.0)
        self.progress_stage_var.set(stage)
        self.progress_detail_var.set(detail or "Procesando...")
        self.progress_percent_var.set(f"{bounded_value * 100.0:5.1f}%")

    def _append_log(self, message: str, level: str = "INFO") -> None:
        prefix = f"[{level}] " if level else ""
        self.log_text.insert(END, f"{prefix}{message}\n")
        self.log_text.see(END)

    def _handle_result(self, result: dict[str, object]) -> None:
        self.last_result = result
        self.analyze_button.configure(state="normal")
        self.open_result_button.configure(state="normal")
        self.open_folder_button.configure(state="normal")
        self.debug_check.configure(state="normal")
        self._update_progress(1.0, "Analisis completado", "Los resultados ya estan disponibles.")

        if bool(result.get("video_alert")):
            self.status_var.set("ALERTA: se detectó posible hurto en el video.")
        else:
            self.status_var.set("No se detectaron ventanas positivas de hurto con el umbral actual.")

        self.summary_var.set(
            "Tracks limpios: "
            f"{result.get('num_tracks_clean', 0)} | "
            "Ventanas evaluadas: "
            f"{result.get('num_windows_evaluated', 0)} | "
            "Ventanas positivas: "
            f"{result.get('num_positive_windows', 0)}"
        )

        for item_id in self.events_tree.get_children():
            self.events_tree.delete(item_id)

        for event in result.get("top_events", []):
            if not isinstance(event, dict):
                continue
            interval_text = f"{float(event['start_sec']):.2f} - {float(event['end_sec']):.2f}"
            probability_text = f"{float(event['cnn_prob_hurto']):.2f}"
            self.events_tree.insert(
                "",
                END,
                values=(int(event["track_id"]), interval_text, probability_text),
            )

        best_evidence_path = Path(str(result.get("best_evidence_path", ""))) if result.get("best_evidence_path") else None
        if best_evidence_path and best_evidence_path.exists():
            image = Image.open(best_evidence_path)
            image.thumbnail((500, 380))
            self.preview_photo = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.preview_photo, text="")
        else:
            message = (
                "No hubo evidencia positiva para mostrar.\n"
                f"El resumen quedó guardado en:\n{result.get('run_dir', '')}"
            )
            self.preview_label.configure(image="", text=message)
            self.preview_photo = None

        if bool(result.get("video_alert")):
            top_event = result.get("top_events", [])[0] if result.get("top_events") else None
            if isinstance(top_event, dict):
                alert_detail = (
                    f"Track {int(top_event['track_id'])} entre {float(top_event['start_sec']):.2f}s y "
                    f"{float(top_event['end_sec']):.2f}s con probabilidad {float(top_event['cnn_prob_hurto']):.2f}."
                )
            else:
                alert_detail = "Se detectaron ventanas positivas en el video analizado."
            self._set_alert_panel("alert", detail=alert_detail)
        else:
            self._set_alert_panel(
                "clear",
                detail=(
                    f"No se detectaron eventos sobre el umbral {float(result.get('alert_threshold', 0.0)):.2f}. "
                    "Revisa el video anotado si quieres validar el seguimiento."
                ),
            )

        if best_evidence_path and best_evidence_path.exists():
            self._set_preview_image(
                image=Image.open(best_evidence_path),
                caption="Mejor evidencia de posible hurto",
                detail=f"Evidencia guardada en: {best_evidence_path}",
            )
        else:
            analyzed_video_path = Path(str(result.get("video_path", ""))) if result.get("video_path") else None
            if analyzed_video_path and analyzed_video_path.exists():
                self._show_video_preview(
                    analyzed_video_path,
                    caption="Vista previa del video analizado",
                    detail=f"No hubo evidencia positiva. Resultados guardados en: {result.get('run_dir', '')}",
                )
            else:
                self._set_preview_text(
                    caption="Resumen del analisis",
                    message=f"No hubo evidencia visual disponible.\nResultados guardados en: {result.get('run_dir', '')}",
                    detail="Abre la carpeta de salida para revisar el resumen y el video anotado.",
                )

    def _handle_error(self, error_message: str) -> None:
        self.analyze_button.configure(state="normal")
        self.open_result_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.debug_check.configure(state="normal")
        self._update_progress(float(self.progress_bar["value"]) / 100.0, "Error", error_message)
        self.status_var.set("El análisis terminó con error.")
        self.summary_var.set("No se generaron resultados.")
        self._set_alert_panel("error", detail=error_message)
        messagebox.showerror("Error durante el análisis", error_message)

    def _clear_previous_result(self) -> None:
        self.last_result = None
        self.log_text.delete("1.0", END)
        self.preview_label.configure(image="", text="La mejor evidencia positiva aparecerá aquí.\nSi no hay alerta, se mostrará el resumen en texto.")
        self.preview_photo = None
        self.open_result_button.configure(state="disabled")
        self.open_folder_button.configure(state="disabled")
        self.preview_caption_var.set("Vista previa")
        self.preview_detail_var.set("Selecciona un video para comenzar.")
        self._update_progress(0.0, "En espera", "Selecciona un video para comenzar.")
        for item_id in self.events_tree.get_children():
            self.events_tree.delete(item_id)

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

        run_dir = Path(str(self.last_result.get("run_dir", "")))
        if not run_dir.exists():
            messagebox.showerror("Carpeta no disponible", "No existe la carpeta del ultimo analisis.")
            return

        try:
            open_path(run_dir)
        except Exception as error:  # noqa: BLE001
            messagebox.showerror("No fue posible abrir la carpeta", str(error))

    def _set_alert_panel(self, state: str, detail: str | None = None) -> None:
        palette = {
            "idle": {
                "bg": "#1f2937",
                "fg": "#f8fafc",
                "detail_fg": "#e5e7eb",
                "title": "Sistema listo",
                "detail": "La alerta aparecera aqui cuando el modelo detecte una escena sospechosa.",
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
                "detail": "No se detectaron ventanas positivas con el umbral configurado.",
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

    def _set_preview_image(self, image: Image.Image, caption: str, detail: str) -> None:
        preview_image = image.copy()
        preview_image.thumbnail((500, 380))
        self.preview_photo = ImageTk.PhotoImage(preview_image)
        self.preview_caption_var.set(caption)
        self.preview_detail_var.set(detail)
        self.preview_label.configure(image=self.preview_photo, text="")

    def _set_preview_text(self, caption: str, message: str, detail: str) -> None:
        self.preview_photo = None
        self.preview_caption_var.set(caption)
        self.preview_detail_var.set(detail)
        self.preview_label.configure(image="", text=message)

    def _show_video_preview(self, video_path: Path, caption: str, detail: str) -> None:
        try:
            image = load_video_preview_image(video_path)
            self._set_preview_image(image=image, caption=caption, detail=detail)
        except Exception as error:  # noqa: BLE001
            self._set_preview_text(
                caption=caption,
                message=f"No fue posible generar la vista previa.\n{video_path}",
                detail=str(error),
            )


def main() -> None:
    root = Tk()
    style = ttk.Style(root)
    if "vista" in style.theme_names():
        style.theme_use("vista")
    app = TheftDetectionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
