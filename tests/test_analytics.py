
from traffic_ai.analytics import TrafficAnalytics


def make_detection(track_id, center_y, class_name="car"):
    return {
        "track_id": track_id,
        "class_name": class_name,
        "center": [100, center_y],
    }


def test_vehicle_crossing_is_counted_once():
    analytics = TrafficAnalytics(line_y=100, deadband=5)

    analytics.update([make_detection(1, 80)], frame_index=0)
    analytics.update([make_detection(1, 120)], frame_index=1)
    analytics.update([make_detection(1, 140)], frame_index=2)

    report = analytics.summary()
    assert report["total_vehicles_crossed"] == 1
    assert report["counts_by_class"]["car"] == 1


def test_crossing_direction_is_recorded():
    analytics = TrafficAnalytics(line_y=100, deadband=5)

    analytics.update([make_detection(2, 120, "truck")], frame_index=0)
    analytics.update([make_detection(2, 80, "truck")], frame_index=1)

    event = analytics.summary()["events"][0]
    assert event["direction"] == "up"
    assert event["class_name"] == "truck"


def test_untracked_detections_are_not_counted():
    analytics = TrafficAnalytics(line_y=100)

    analytics.update([make_detection(None, 80)], frame_index=0)
    analytics.update([make_detection(None, 120)], frame_index=1)

    assert analytics.summary()["total_vehicles_crossed"] == 0

def test_crossing_is_detected_across_multiple_frames():
    analytics = TrafficAnalytics(line_y=100, deadband=5)

    for frame_index, center_y in enumerate([80, 90, 96, 100, 104, 106, 110]):
        analytics.update(
            [make_detection(7, center_y, "car")],
            frame_index=frame_index,
        )

    report = analytics.summary()
    assert report["total_vehicles_crossed"] == 1
    assert report["events"][0]["direction"] == "down"

