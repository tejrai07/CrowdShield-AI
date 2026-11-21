"""Unit tests for vision/detector.py.

These are fast, mock-based unit tests — they do NOT download or run a real
YOLO model, so they run as part of the standard smoke-fast suite. Real
model/video integration testing lives in test_detector_integration.py and
is opt-in (skipped unless explicitly requested), since it requires network
access to fetch model weights and takes several seconds.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from vision.detector import PERSON_CLASS_ID, Detection, PersonDetector


def test_detection_is_frozen_and_lightweight():
    det = Detection(x1=10.0, y1=20.0, x2=50.0, y2=100.0, confidence=0.87)
    assert det.class_id == PERSON_CLASS_ID
    assert det.width == 40.0
    assert det.height == 80.0
    assert det.center == (30.0, 60.0)
    with pytest.raises(Exception):
        det.x1 = 999.0  # frozen dataclass must reject mutation


def test_resolve_device_explicit_cpu():
    assert PersonDetector._resolve_device("cpu") == "cpu"


def test_resolve_device_explicit_cuda():
    assert PersonDetector._resolve_device("cuda") == "cuda"


def test_resolve_device_auto_falls_back_to_cpu_when_no_gpu():
    with patch("torch.cuda.is_available", return_value=False):
        assert PersonDetector._resolve_device("auto") == "cpu"


def test_resolve_device_auto_prefers_cuda_when_available():
    with patch("torch.cuda.is_available", return_value=True):
        assert PersonDetector._resolve_device("auto") == "cuda"


def _make_mock_yolo_result(xyxy, confs, cls_ids):
    """Build a mock ultralytics Results object with the minimal surface
    PersonDetector.detect() actually reads: results[0].boxes.{xyxy,conf,cls}."""
    boxes = MagicMock()
    boxes.__len__ = lambda self: len(xyxy)
    boxes.xyxy = MagicMock()
    boxes.xyxy.cpu.return_value.numpy.return_value = np.array(xyxy, dtype=float)
    boxes.conf = MagicMock()
    boxes.conf.cpu.return_value.numpy.return_value = np.array(confs, dtype=float)
    boxes.cls = MagicMock()
    boxes.cls.cpu.return_value.numpy.return_value = np.array(cls_ids, dtype=float)

    result = MagicMock()
    result.boxes = boxes
    return [result]


def test_detect_filters_to_person_class_only():
    """Given detections containing multiple classes, only person detections
    remain — the core B1 filtering requirement."""
    # class 0 = person, class 2 = car (COCO). Model is asked to only return
    # class 0 via `classes=[PERSON_CLASS_ID]`, but detect() also defensively
    # filters — this test simulates the model returning a stray car anyway.
    mock_results = _make_mock_yolo_result(
        xyxy=[[10, 10, 50, 100], [200, 200, 260, 260]],
        confs=[0.9, 0.8],
        cls_ids=[0, 2],  # one person, one car
    )

    with patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_results
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model="fake.pt", device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

    assert len(detections) == 1
    assert detections[0].class_id == PERSON_CLASS_ID
    assert detections[0].confidence == pytest.approx(0.9)


def test_detect_bounding_box_representation_is_valid():
    mock_results = _make_mock_yolo_result(
        xyxy=[[15.0, 25.0, 95.0, 205.0]],
        confs=[0.73],
        cls_ids=[0],
    )

    with patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_results
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model="fake.pt", device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

    assert len(detections) == 1
    det = detections[0]
    assert det.x1 < det.x2, "x1 must be top-left, x2 bottom-right"
    assert det.y1 < det.y2, "y1 must be top-left, y2 bottom-right"
    assert 0.0 <= det.confidence <= 1.0


def test_detect_returns_empty_list_when_no_boxes():
    mock_results = [MagicMock(boxes=None)]

    with patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_results
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model="fake.pt", device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = detector.detect(frame)

    assert detections == []


def test_detector_passes_configured_settings_to_predict():
    """Verify detection.confidence / image_size / device configuration is
    actually respected and passed through to the underlying model call."""
    mock_results = _make_mock_yolo_result(xyxy=[], confs=[], cls_ids=[])

    with patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_results
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(
            model="fake.pt", confidence=0.63, device="cpu", image_size=512
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame)

        _, kwargs = mock_model.predict.call_args
        assert kwargs["conf"] == 0.63
        assert kwargs["imgsz"] == 512
        assert kwargs["device"] == "cpu"
        assert kwargs["classes"] == [PERSON_CLASS_ID]


def test_model_constructed_exactly_once():
    """detect() must never re-construct the model — it must be loaded once
    in __init__ and reused across calls."""
    mock_results = _make_mock_yolo_result(xyxy=[], confs=[], cls_ids=[])

    with patch("ultralytics.YOLO") as mock_yolo_cls:
        mock_model = MagicMock()
        mock_model.predict.return_value = mock_results
        mock_yolo_cls.return_value = mock_model

        detector = PersonDetector(model="fake.pt", device="cpu")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detector.detect(frame)
        detector.detect(frame)
        detector.detect(frame)

        assert mock_yolo_cls.call_count == 1, "YOLO() constructor must be called exactly once"
        assert mock_model.predict.call_count == 3
