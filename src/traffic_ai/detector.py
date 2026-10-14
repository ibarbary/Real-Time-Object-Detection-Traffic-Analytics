

import torch
from ultralytics import YOLO

# COCO class IDs for common vehicles.
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


class VehicleDetector:
    """Detect vehicles in images or video frames using a YOLO model."""

    def __init__(
        self,
        model_name: str = "yolo11n.pt",
        confidence_threshold: float = 0.35,
        device: str | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.device = device or ("0" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_name)

    def detect(self, frame):
        """Return vehicle detections for one OpenCV image frame."""
        results = self.model.predict(
            source=frame,
            conf=self.confidence_threshold,
            classes=list(VEHICLE_CLASSES),
            device=self.device,
            verbose=False,
        )

        detections = []

        for box in results[0].boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": VEHICLE_CLASSES[class_id],
                    "confidence": float(box.conf.item()),
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                }
            )

        return detections
