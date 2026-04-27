import os
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


@dataclass
class Detection:
    bbox: tuple          # (x1, y1, x2, y2)
    confidence: float
    label: str
    model_name: str
    color: tuple = field(default=(0, 255, 0))


class YOLOModel:
    """Wraps a single YOLO model (.pt via ultralytics or .onnx via OpenCV)."""

    def __init__(self, model_path: str):
        self.path = Path(model_path)
        self.name = self.path.stem
        self.ext = self.path.suffix.lower()
        self._model = None
        self.enabled = True
        self.conf_threshold = 0.4
        self.task = self._guess_task()
        self._load()

    def _guess_task(self) -> str:
        name_lower = self.name.lower()
        if any(k in name_lower for k in ("gender",)):
            return "gender"
        if any(k in name_lower for k in ("expr", "emotion", "affect", "face")):
            return "expression"
        return "detection"

    def _load(self):
        if self.ext == ".pt":
            try:
                from ultralytics import YOLO
                self._model = YOLO(str(self.path))
                self._backend = "ultralytics"
            except ImportError:
                raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")
        elif self.ext == ".onnx":
            self._model = cv2.dnn.readNetFromONNX(str(self.path))
            self._backend = "onnx"
            self._read_onnx_meta()
        else:
            raise ValueError(f"Unsupported model format: {self.ext}")

    def _read_onnx_meta(self):
        """Try to read class names from a sibling .txt file (one class per line)."""
        txt = self.path.with_suffix(".txt")
        if txt.exists():
            self.class_names = [l.strip() for l in txt.read_text().splitlines() if l.strip()]
        else:
            # Fallback defaults by task
            if self.task == "gender":
                self.class_names = ["Male", "Female"]
            else:
                self.class_names = [
                    "angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"
                ]

    def infer(self, frame: np.ndarray) -> list[Detection]:
        if not self.enabled or self._model is None:
            return []
        if self._backend == "ultralytics":
            return self._infer_ultralytics(frame)
        return self._infer_onnx(frame)

    def _infer_ultralytics(self, frame: np.ndarray) -> list[Detection]:
        results = self._model(frame, conf=self.conf_threshold, verbose=False)
        detections = []
        for r in results:
            names = r.names
            
            # Object Detection models
            if getattr(r, 'boxes', None) is not None and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    label = names.get(cls_id, str(cls_id))
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=conf,
                        label=label,
                        model_name=self.name,
                        color=self._label_color(label),
                    ))
            
            # Classification models
            elif getattr(r, 'probs', None) is not None:
                top1_idx = r.probs.top1
                conf = float(r.probs.top1conf)
                if conf >= self.conf_threshold:
                    label = names.get(top1_idx, str(top1_idx))
                    h, w = frame.shape[:2]
                    detections.append(Detection(
                        bbox=(0, 0, w, h),
                        confidence=conf,
                        label=label,
                        model_name=self.name,
                        color=self._label_color(label),
                    ))
                    
        return detections

    def _infer_onnx(self, frame: np.ndarray) -> list[Detection]:
        """
        Generic YOLOv5/v8 ONNX inference.
        Expects output shape (1, num_boxes, 5+num_classes) or (1, 5+num_classes, num_boxes).
        """
        h, w = frame.shape[:2]
        input_size = 640
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255.0, (input_size, input_size),
            swapRB=True, crop=False
        )
        self._model.setInput(blob)
        outputs = self._model.forward()

        # Handle (1, num_boxes, cols) or (1, cols, num_boxes)
        out = outputs[0] if isinstance(outputs, (list, tuple)) else outputs
        if out.ndim == 3:
            out = out[0]  # -> (num_boxes, cols) or (cols, num_boxes)
        if out.shape[0] < out.shape[1]:
            out = out.T   # normalise to (num_boxes, cols)

        scale_x = w / input_size
        scale_y = h / input_size
        detections = []

        for row in out:
            objectness = float(row[4])
            if objectness < self.conf_threshold:
                continue
            class_scores = row[5:]
            cls_id = int(np.argmax(class_scores))
            conf = float(class_scores[cls_id]) * objectness
            if conf < self.conf_threshold:
                continue
            cx, cy, bw, bh = row[0], row[1], row[2], row[3]
            x1 = int((cx - bw / 2) * scale_x)
            y1 = int((cy - bh / 2) * scale_y)
            x2 = int((cx + bw / 2) * scale_x)
            y2 = int((cy + bh / 2) * scale_y)
            label = self.class_names[cls_id] if cls_id < len(self.class_names) else str(cls_id)
            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                label=label,
                model_name=self.name,
                color=self._label_color(label),
            ))
        return detections

    def _label_color(self, label: str) -> tuple:
        palette = {
            # Expressions
            "happy":    (50, 205, 50),
            "sad":      (70, 130, 180),
            "angry":    (220, 50, 50),
            "surprise": (255, 165, 0),
            "fear":     (180, 90, 180),
            "disgust":  (0, 180, 120),
            "neutral":  (160, 160, 160),
            # Gender
            "male":     (100, 149, 237),
            "female":   (255, 105, 180),
        }
        return palette.get(label.lower(), (0, 220, 100))


class ModelManager:
    """Scans /models, loads all .pt and .onnx files, runs them on frames."""

    def __init__(self):
        self.models: list[YOLOModel] = []
        self._scan()

    def _scan(self):
        if not MODELS_DIR.exists():
            MODELS_DIR.mkdir(parents=True, exist_ok=True)
            return
        for ext in ("*.pt", "*.onnx"):
            for p in sorted(MODELS_DIR.glob(ext)):
                try:
                    m = YOLOModel(str(p))
                    self.models.append(m)
                    print(f"[ModelManager] Loaded: {p.name} ({m._backend})")
                except Exception as e:
                    print(f"[ModelManager] Failed to load {p.name}: {e}")

    def reload(self):
        self.models.clear()
        self._scan()

    def run_all(self, frame: np.ndarray) -> list[Detection]:
        detections = []
        
        # Pisahkan model berdasarkan kemampuannya
        # Asumsi sementara: model gender adalah classification, sisanya detection
        cls_models = [m for m in self.models if m.enabled and m.task == "gender"]
        det_models = [m for m in self.models if m.enabled and m.task != "gender"]
        
        face_detections = []
        
        # 1. Jalankan model deteksi terlebih dahulu
        for m in det_models:
            res = m.infer(frame)
            detections.extend(res)
            # Kumpulkan hasil deteksi untuk di-crop nanti
            face_detections.extend(res)
            
        # 2. Jalankan model klasifikasi pada hasil crop wajah
        if face_detections and cls_models:
            for face in face_detections:
                x1, y1, x2, y2 = face.bbox
                h, w = frame.shape[:2]
                
                # Pastikan koordinat ada di dalam batas frame
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                
                if x2 <= x1 or y2 <= y1:
                    continue
                    
                # Crop bagian wajah
                face_crop = frame[y1:y2, x1:x2]
                
                for m in cls_models:
                    cls_res = m.infer(face_crop)
                    for cr in cls_res:
                        # Ubah bbox hasil klasifikasi (yg awalnya selebar crop) 
                        # menjadi bbox wajah di frame asli
                        cr.bbox = face.bbox
                        detections.append(cr)
                        
        # Jika tidak ada wajah yang terdeteksi, jalankan klasifikasi di seluruh frame
        elif not face_detections and cls_models:
            for m in cls_models:
                detections.extend(m.infer(frame))
                
        return detections

    @property
    def loaded(self) -> bool:
        return len(self.models) > 0
