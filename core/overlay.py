import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class OverlayConfig:
    show_bbox: bool = True
    show_label: bool = True
    show_confidence: bool = True
    show_model_name: bool = False
    box_thickness: int = 2
    font_scale: float = 0.65
    font_thickness: int = 1


def draw_detections(frame: np.ndarray, detections: list, cfg: OverlayConfig) -> np.ndarray:
    out = frame.copy()
    
    # Group detections by bounding box
    from collections import defaultdict
    grouped_dets = defaultdict(list)
    for det in detections:
        grouped_dets[det.bbox].append(det)

    for bbox, dets in grouped_dets.items():
        x1, y1, x2, y2 = bbox
        # Use the first detection's color for the bounding box and text background
        color = dets[0].color

        # Clamp to frame
        h, w = out.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if cfg.show_bbox:
            # Filled corner brackets instead of full rectangle - more elegant
            _draw_corner_box(out, x1, y1, x2, y2, color, cfg.box_thickness)

        if cfg.show_label:
            group_texts = []
            for det in dets:
                parts = [det.label]
                if cfg.show_confidence:
                    parts.append(f"{det.confidence:.0%}")
                if cfg.show_model_name:
                    parts.append(f"[{det.model_name}]")
                group_texts.append(" ".join(parts))
                
            text = " | ".join(group_texts)

            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, cfg.font_scale, cfg.font_thickness
            )
            pad = 4
            lx = x1
            ly = max(y1 - th - pad * 2, 0)

            # Pill background
            cv2.rectangle(out, (lx, ly), (lx + tw + pad * 2, ly + th + pad * 2), color, -1)
            cv2.rectangle(out, (lx, ly), (lx + tw + pad * 2, ly + th + pad * 2), (0, 0, 0), 1)
            cv2.putText(
                out, text,
                (lx + pad, ly + th + pad),
                cv2.FONT_HERSHEY_SIMPLEX,
                cfg.font_scale,
                _text_color(color),
                cfg.font_thickness,
                cv2.LINE_AA,
            )
    return out


def _draw_corner_box(img, x1, y1, x2, y2, color, thickness):
    """Draws corner-bracket style bounding box."""
    lw = max(8, min(30, (x2 - x1) // 6, (y2 - y1) // 6))
    t = thickness

    # Top-left
    cv2.line(img, (x1, y1), (x1 + lw, y1), color, t)
    cv2.line(img, (x1, y1), (x1, y1 + lw), color, t)
    # Top-right
    cv2.line(img, (x2, y1), (x2 - lw, y1), color, t)
    cv2.line(img, (x2, y1), (x2, y1 + lw), color, t)
    # Bottom-left
    cv2.line(img, (x1, y2), (x1 + lw, y2), color, t)
    cv2.line(img, (x1, y2), (x1, y2 - lw), color, t)
    # Bottom-right
    cv2.line(img, (x2, y2), (x2 - lw, y2), color, t)
    cv2.line(img, (x2, y2), (x2, y2 - lw), color, t)


def _text_color(bg: tuple) -> tuple:
    """Returns black or white text based on background luminance."""
    r, g, b = bg[2], bg[1], bg[0]   # OpenCV is BGR
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)
