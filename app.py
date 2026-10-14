
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from traffic_ai.tracker import VehicleTracker
from traffic_ai.video import process_video

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_browser_video(source_path: Path) -> Path:
    """Convert an annotated video to browser-compatible H.264 MP4."""
    source_path = Path(source_path)
    browser_path = source_path.with_name(
        f"{source_path.stem}_browser.mp4"
    )

    if (
        not browser_path.is_file()
        or browser_path.stat().st_mtime < source_path.stat().st_mtime
    ):
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i", str(source_path),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an",
                str(browser_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

    return browser_path


st.set_page_config(
    page_title="Traffic Analytics AI",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 Traffic Analytics AI")
st.caption(
    "Vehicle detection · Multi-object tracking · Traffic flow analytics"
)

with st.sidebar:
    st.header("Analysis settings")

    confidence = st.slider(
        "Detection confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Higher values reject more uncertain detections.",
    )

    line_position = st.slider(
        "Counting line position",
        min_value=0.10,
        max_value=0.90,
        value=0.60,
        step=0.05,
        help="Vertical position of the counting line as a fraction of video height.",
    )

    frame_stride = st.select_slider(
        "Frames to skip",
        options=[1, 2, 3, 5],
        value=1,
        help=(
            "1 processes every frame. Larger values run faster but may "
            "miss crossings and weaken tracking."
        ),
    )

st.info(
    "Runs locally on your computer. Uploaded videos are processed "
    "by YOLO and ByteTrack."
)

uploaded_file = st.file_uploader(
    "Upload a traffic video",
    type=["mp4", "avi", "mov", "mkv"],
)

if uploaded_file is not None:
    st.video(uploaded_file)

    if st.button("Run traffic analysis", type="primary"):
        run_id = uuid4().hex[:8]

        input_path = (
            RAW_DIR
            / f"upload_{run_id}{Path(uploaded_file.name).suffix}"
        )
        output_path = OUTPUT_DIR / f"traffic_{run_id}_annotated.mp4"

        # Clear previous results so they cannot be mistaken for this run.
        for key in (
            "traffic_report",
            "traffic_output_path",
            "traffic_csv_path",
            "traffic_browser_path",
        ):
            st.session_state.pop(key, None)

        try:
            input_path.write_bytes(uploaded_file.getvalue())

            with st.spinner(
                "Detecting and tracking vehicles... "
                "This may take a while on longer videos."
            ):
                start_time = time.perf_counter()

                tracker = VehicleTracker(
                    model_name="yolo11n.pt",
                    confidence_threshold=confidence,
                )

                report = process_video(
                    input_path=input_path,
                    output_path=output_path,
                    tracker=tracker,
                    frame_stride=frame_stride,
                    line_position=line_position,
                )

                report["processing_seconds"] = round(
                    time.perf_counter() - start_time, 2
                )

            # Save the correct paths for this analysis.
            st.session_state["traffic_report"] = report
            st.session_state["traffic_output_path"] = str(output_path)
            st.session_state["traffic_csv_path"] = report["csv_path"]

        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

        finally:
            input_path.unlink(missing_ok=True)


report = st.session_state.get("traffic_report")
output_path_str = st.session_state.get("traffic_output_path")
csv_path_str = st.session_state.get("traffic_csv_path")

if report and output_path_str and Path(output_path_str).is_file():
    st.divider()
    st.header("Analysis results")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Vehicles counted",
        report["total_vehicles_crossed"],
    )
    col2.metric(
        "Frames processed",
        report["processed_frames"],
    )
    col3.metric(
        "Video FPS",
        f'{report["fps"]:.1f}',
    )
    col4.metric(
        "Processing time",
        f'{report["processing_seconds"]:.1f} s',
    )

    debug1, debug2 = st.columns(2)

    debug1.metric(
        "Unique tracked IDs",
        report.get("tracked_ids_seen", "N/A"),
    )
    debug2.metric(
        "Crossing candidates",
        report.get("crossing_candidates", "N/A"),
    )

    st.subheader("Annotated video")

    output_file = Path(output_path_str)

    try:
        with st.spinner("Preparing video for browser playback..."):
            browser_file = get_browser_video(output_file)

        video_bytes = browser_file.read_bytes()

        st.caption(
            f"Output: {browser_file.name} · "
            f"{len(video_bytes) / (1024 * 1024):.1f} MB"
        )

        st.video(video_bytes, format="video/mp4")

        st.download_button(
            "Download annotated video",
            data=video_bytes,
            file_name=browser_file.name,
            mime="video/mp4",
            key=f"download_video_{browser_file.stem}",
        )

    except subprocess.CalledProcessError as exc:
        st.error("Could not convert the video for browser playback.")
        st.code(
            exc.stderr[-2000:]
            if exc.stderr
            else str(exc)
        )

    except Exception as exc:
        st.error(f"Could not display the annotated video: {exc}")

    left, right = st.columns(2)

    with left:
        st.subheader("Crossings by vehicle class")

        counts = report["counts_by_class"]

        if counts:
            counts_df = pd.DataFrame(
                [
                    {"Vehicle class": vehicle_class, "Crossings": count}
                    for vehicle_class, count in counts.items()
                ]
            )

            st.bar_chart(counts_df.set_index("Vehicle class"))
            st.dataframe(
                counts_df,
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.write("No tracked vehicles crossed the counting line.")

    with right:
        st.subheader("Crossing events")

        events_df = pd.DataFrame(
            report["events"],
            columns=[
                "track_id",
                "class_name",
                "direction",
                "frame_index",
            ],
        )

        st.dataframe(
            events_df,
            hide_index=True,
            use_container_width=True,
        )

        if csv_path_str and Path(csv_path_str).is_file():
            st.download_button(
                "Download crossing events (CSV)",
                data=Path(csv_path_str).read_bytes(),
                file_name="traffic_events.csv",
                mime="text/csv",
                key="download_crossing_events",
            )

    st.caption(
        "Counts represent unique tracked IDs crossing the configured line. "
        "Tracking errors, occlusion, and skipped frames can affect results."
    )
