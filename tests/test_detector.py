
import pytest

from traffic_ai.detector import VehicleDetector


def test_invalid_confidence_threshold_does_not_load_model(monkeypatch):
    def unexpected_model_load(*args, **kwargs):
        pytest.fail("The model should not load for an invalid threshold.")

    monkeypatch.setattr("traffic_ai.detector.YOLO", unexpected_model_load)

    with pytest.raises(ValueError, match="confidence_threshold"):
        VehicleDetector(confidence_threshold=1.5)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_threshold_outside_range_is_rejected(monkeypatch, threshold):
    def unexpected_model_load(*args, **kwargs):
        pytest.fail("The model should not load for an invalid threshold.")

    monkeypatch.setattr("traffic_ai.detector.YOLO", unexpected_model_load)

    with pytest.raises(ValueError):
        VehicleDetector(confidence_threshold=threshold)
