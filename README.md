# Real-Time Object Detection & Traffic Analytics

A computer vision project for vehicle detection, multi-object tracking, and traffic analytics using **YOLO11, ByteTrack, OpenCV, and Streamlit**.

The application processes traffic videos, tracks vehicles across frames, counts vehicles crossing a configurable virtual line, and generates annotated videos and downloadable analytics reports.

## Features

- **Vehicle detection:** Detects cars, motorcycles, buses, and trucks using YOLO11.
- **Multi-object tracking:** Uses ByteTrack to associate detections across video frames.
- **Traffic counting:** Counts each tracked vehicle once when it crosses a configurable counting line.
- **Vehicle statistics:** Reports unique tracked IDs observed and vehicles counted by class.
- **Direction analysis:** Records whether a vehicle crosses the line upward or downward.
- **Interactive dashboard:** Adjusts detection confidence, counting-line position, and frame-processing stride.
- **Annotated video:** Displays vehicle bounding boxes, tracking IDs, and the counting line.
- **Exportable results:** Downloads annotated video and CSV crossing events.

## How It Works

1. Upload a traffic video through the Streamlit dashboard.
2. YOLO11 detects vehicles in the selected frames.
3. ByteTrack assigns tracking IDs to detected vehicles.
4. The analytics module detects when tracked vehicles cross the virtual counting line.
5. The application displays summary statistics and generates downloadable results.

## Tech Stack

- **Language:** Python
- **Object detection:** Ultralytics YOLO11
- **Object tracking:** ByteTrack
- **Computer vision:** OpenCV
- **Dashboard:** Streamlit
- **Data analysis and export:** pandas, CSV
- **Testing and code quality:** pytest, Ruff

## Project Structure

```text
traffic-analytics-ai/
├── app.py
├── run_video.py
├── pyproject.toml
├── README.md
├── src/
│   └── traffic_ai/
│       ├── detector.py
│       ├── tracker.py
│       ├── analytics.py
│       └── video.py
├── tests/
│   ├── test_detector.py
│   └── test_analytics.py
├── configs/
├── data/
├── outputs/
└── notebooks/
```

## Getting Started

### Prerequisites

- Python 3.11
- Git
- FFmpeg available on your system PATH for browser-compatible video conversion
- A compatible PyTorch installation; GPU acceleration is optional

### Installation

Clone the repository:

```bash
git clone https://github.com/ibarbary/Real-Time-Object-Detection-Traffic-Analytics.git
cd Real-Time-Object-Detection-Traffic-Analytics
```

Create and activate a virtual environment.

**Windows PowerShell:**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the project dependencies according to the project's `pyproject.toml` configuration. Ensure that PyTorch, Ultralytics, OpenCV, Streamlit, pandas, and FFmpeg are available.

### Run the Dashboard

```bash
python -m streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open it in your browser, upload a traffic video, configure the analysis settings, and select **Run traffic analysis**.

## Understanding the Metrics

- **Vehicles counted:** Unique tracked IDs that crossed the virtual counting line during the video.
- **Unique vehicles observed:** Distinct tracking IDs encountered anywhere in the video.
- **Counts by class:** Crossing counts grouped by vehicle category.
- **Direction:** The direction in which each counted vehicle crossed the line.

Tracking IDs are estimates of vehicle identity. Occlusion, missed detections, and tracking ID switches can affect the accuracy of the counts.

## Testing

Run the automated test suite:

```bash
python -m pytest -v
```

Check code quality with Ruff:

```bash
python -m ruff check src tests app.py run_video.py
```