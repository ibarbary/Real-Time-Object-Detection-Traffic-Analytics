
import torch
from ultralytics import YOLO

from traffic_ai.detector import VEHICLE_CLASSES


class VehicleTracker:
    """Detect and track vehicles across sequential video frames."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        device: str | None = None,
        tracker_config: str = "bytetrack.yaml",
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        self.tracker_config = tracker_config

    def track(self, frame) -> list[dict]:
        """Return vehicle detections and IDs for one video frame."""
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence_threshold,
            classes=list(VEHICLE_CLASSES),
            device=self.device,
            verbose=False,
        )

        detections = []
        boxes = results[0].boxes

        if boxes is None:
            return detections

        for index, box in enumerate(boxes):
            class_id = int(box.cls.item())
            track_id = (
                int(boxes.id[index].item())
                if boxes.id is not None
                else None
            )
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "track_id": track_id,
                    "class_id": class_id,
                    "class_name": VEHICLE_CLASSES[class_id],
                    "confidence": float(box.conf.item()),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "center": [
                        (float(x1) + float(x2)) / 2,
                        (float(y1) + float(y2)) / 2,
                    ],
                }
            )

        return detections
