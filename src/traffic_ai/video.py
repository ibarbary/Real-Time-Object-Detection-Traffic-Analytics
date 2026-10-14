
import csv
from pathlib import Path

import cv2

from traffic_ai.analytics import TrafficAnalytics
from traffic_ai.tracker import VehicleTracker


def process_video(
    input_path: str | Path,
    output_path: str | Path,
    tracker: VehicleTracker,
    frame_stride: int = 1,
    line_position: float = 0.60,
) -> dict:
    """Track vehicles, annotate video, and export crossing events."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    if frame_stride < 1:
        raise ValueError("frame_stride must be >= 1.")
    if not 0.0 < line_position < 1.0:
        raise ValueError("line_position must be between 0 and 1.")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError("Video has invalid FPS or dimensions.")

    line_y = int(height * line_position)
    analytics = TrafficAnalytics(line_y=line_y)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_path}")

    frame_index = 0
    processed_frames = 0

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % frame_stride == 0:
                detections = tracker.track(frame)
                analytics.update(detections, frame_index)
                processed_frames += 1

                for detection in detections:
                    x1, y1, x2, y2 = map(int, detection["bbox"])
                    track_id = detection["track_id"]
                    label = (
                        f'{detection["class_name"]} '
                        f'ID:{track_id if track_id is not None else "-"}'
                    )

                    cv2.rectangle(
                        frame, (x1, y1), (x2, y2), (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

            # Draw the counting line and current crossing total.
            cv2.line(frame, (0, line_y), (width, line_y), (0, 0, 255), 2)
            cv2.putText(
                frame,
                f"Vehicles crossed: {len(analytics.events)}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

            writer.write(frame)
            frame_index += 1
    finally:
        capture.release()
        writer.release()

    report = analytics.summary()
    report.update(
        {
            "input_path": str(input_path),
            "output_path": str(output_path),
            "input_frames": frame_index,
            "processed_frames": processed_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "counting_line_y": line_y,
        }
    )

    csv_path = output_path.with_name(f"{output_path.stem}_events.csv")
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        fieldnames = [
            "track_id", "class_name", "direction", "frame_index"
        ]
        writer_csv = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer_csv.writeheader()
        writer_csv.writerows(report["events"])

    report["csv_path"] = str(csv_path)
    return report
