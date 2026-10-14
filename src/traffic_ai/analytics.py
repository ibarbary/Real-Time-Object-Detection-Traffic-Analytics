
from collections import Counter


class TrafficAnalytics:
    """Count unique tracked vehicles crossing a horizontal line."""

    def __init__(self, line_y: int, deadband: int = 5) -> None:
        if line_y < 0:
            raise ValueError("line_y must be >= 0.")
        if deadband < 0:
            raise ValueError("deadband must be >= 0.")

        self.line_y = line_y
        self.deadband = deadband
        self.previous_y: dict[int, float] = {}
        self.last_side: dict[int, str] = {}
        self.counted_ids: set[int] = set()
        self.counts_by_class: Counter = Counter()
        self.events: list[dict] = []
        self.tracked_ids_seen: set[int] = set()
        self.crossing_candidates = 0

    def update(self, detections: list[dict], frame_index: int) -> None:
        """Update positions and record crossings of the counting line."""
        for detection in detections:
            track_id = detection["track_id"]
            if track_id is None:
                continue

            track_id = int(track_id)
            center_y = float(detection["center"][1])
            self.tracked_ids_seen.add(track_id)

            # Only update a side after the vehicle clears the deadband.
            if center_y < self.line_y - self.deadband:
                current_side = "above"
            elif center_y > self.line_y + self.deadband:
                current_side = "below"
            else:
                current_side = None

            previous_side = self.last_side.get(track_id)

            if current_side is not None:
                if (
                    previous_side is not None
                    and current_side != previous_side
                    and track_id not in self.counted_ids
                ):
                    self.crossing_candidates += 1
                    vehicle_class = detection["class_name"]
                    direction = (
                        "down"
                        if previous_side == "above"
                        else "up"
                    )

                    self.counted_ids.add(track_id)
                    self.counts_by_class[vehicle_class] += 1
                    self.events.append({
                        "track_id": track_id,
                        "class_name": vehicle_class,
                        "direction": direction,
                        "frame_index": frame_index,
                    })

                self.last_side[track_id] = current_side

            self.previous_y[track_id] = center_y

    def summary(self) -> dict:
        """Return crossing counts and useful diagnostic information."""
        return {
            "total_vehicles_crossed": len(self.events),
            "counts_by_class": dict(self.counts_by_class),
            "events": list(self.events),
            "tracked_ids_seen": len(self.tracked_ids_seen),
            "crossing_candidates": self.crossing_candidates,
        }
