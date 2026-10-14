
import cv2

from traffic_ai.detector import VehicleDetector


def main() -> None:
    image_path = "bus.jpg"
    frame = cv2.imread(image_path)

    if frame is None:
        raise FileNotFoundError(
            f"Could not load {image_path}. Put a test image in the project root."
        )

    detector = VehicleDetector()
    detections = detector.detect(frame)

    for detection in detections:
        print(detection)

    annotated = frame.copy()

    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["bbox"])
        label = (
            f'{detection["class_name"]} '
            f'{detection["confidence"]:.2f}'
        )

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path = "outputs/detections.jpg"
    cv2.imwrite(output_path, annotated)
    print(f"Saved annotated image to {output_path}")


if __name__ == "__main__":
    main()
