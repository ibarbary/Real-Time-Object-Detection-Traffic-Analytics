
import json

from traffic_ai.tracker import VehicleTracker
from traffic_ai.video import process_video


def main() -> None:
    tracker = VehicleTracker(
        model_name="yolo11n.pt",
        confidence_threshold=0.35,
    )

    report = process_video(
        input_path="data/raw/traffic.mp4",
        output_path="outputs/traffic_annotated.mp4",
        tracker=tracker,
        frame_stride=1,
        line_position=0.60,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
