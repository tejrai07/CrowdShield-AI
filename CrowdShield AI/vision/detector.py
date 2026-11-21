"""Person detector abstraction over Ultralytics YOLO.

The rest of the project must never import ultralytics/torch directly — this
module is the single boundary. Everything downstream (tracking, zones,
features) consumes only the lightweight `Detection` objects defined here.

Bounding box convention (documented once, used everywhere downstream):
    - Format: (x1, y1, x2, y2) — top-left / bottom-right corners.
    - Coordinate origin: (0, 0) at the top-left of the frame, x increases
      right, y increases down (standard image/array convention, matches
      OpenCV).
    - Units: PIXEL coordinates in the source frame's native resolution
      (NOT normalized 0-1). Any normalization needed by a downstream
      consumer is that consumer's responsibility.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

# COCO class id for "person". We only ever keep this class — filtering
# happens as early as possible (inside detect()) so nothing downstream
# needs to know about the other 79 COCO classes.
PERSON_CLASS_ID = 0


@dataclass(frozen=True, slots=True)
class Detection:
    """A single detected person in one frame.

    Lightweight and framework-agnostic on purpose — this is the only
    object type that crosses the vision/ boundary.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = PERSON_CLASS_ID

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


class PersonDetector:
    """Loads a YOLO model once and reuses it for repeated frame inference.

    Usage:
        detector = PersonDetector(model="yolov8n.pt", confidence=0.4,
                                   device="auto", image_size=640)
        detections = detector.detect(frame)   # frame: np.ndarray (H, W, 3) BGR
    """

    def __init__(
        self,
        model: str = "yolov8n.pt",
        confidence: float = 0.4,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        image_size: int = 640,
    ) -> None:
        # Imported lazily, inside __init__, not at module top-level. This
        # keeps `import vision.detector` cheap for any code that only needs
        # the `Detection` dataclass (e.g. unit tests) without forcing a
        # torch/ultralytics import just to reference the type.
        from ultralytics import YOLO

        self.confidence = confidence
        self.image_size = image_size
        self.model_name = model
        self.resolved_device = self._resolve_device(device)

        # Model loaded exactly once here. detect() must never re-construct it.
        self._model = YOLO(model)
        self._model.to(self.resolved_device)

    @staticmethod
    def _resolve_device(device: Literal["cpu", "cuda", "auto"]) -> str:
        """Resolve 'auto' to an actual device, never silently claim GPU."""
        if device in ("cpu", "cuda"):
            return device
        # auto
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Run inference on a single BGR frame and return person detections only.

        Filtering to PERSON_CLASS_ID happens here — nothing downstream ever
        sees non-person detections.
        """
        results = self._model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self.resolved_device,
            classes=[PERSON_CLASS_ID],  # ask the model to only return people
            verbose=False,
        )

        detections: list[Detection] = []
        if not results:
            return detections

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return detections

        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        cls_ids = boxes.cls.cpu().numpy()

        for (x1, y1, x2, y2), conf, cls_id in zip(xyxy, confs, cls_ids):
            cls_id_int = int(cls_id)
            if cls_id_int != PERSON_CLASS_ID:
                # Defensive filter — `classes=[PERSON_CLASS_ID]` above should
                # already guarantee this, but we never trust framework
                # internals silently in a safety-relevant pipeline.
                continue
            detections.append(
                Detection(
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                    confidence=float(conf),
                    class_id=cls_id_int,
                )
            )

        return detections
