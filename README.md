# ⚠️ Modified Fork

This fork includes a modified `main.py` to fix hallucinations and add video support.

**Changes made:**
- **Added URL Support:** Integrated `yt-dlp` to download audio from YouTube/Video links.
- **UI Update:** Added Tabs for "Upload File" vs "Paste URL".
- **Controls:** Added Language Dropdown, Task Selector, and Initial Prompt.
- **Dependencies:** Added `yt-dlp` to `pyproject.toml`.

Original README follows below:
---
# `mlx-whisper` Web UI

Fast STT (Speach-to-Text) Web UI with mlx-whisper. The model is Whisper Large-3-Turbo.

## Prerequisites

- Apple Silicon Mac
- Python `>=3.12`
- uv `>=0.4.0`

## Usage

Install the packages.

```bash
uv sync
```

Run the app.

```bash
uv run main.py
```

Open http://127.0.0.1:7860 with your browser and you can view the app.

![screenshot](./assets/screenshot-1.png)

## Reference

- [mlx-whisper · PyPI](https://pypi.org/project/mlx-whisper/)
- [gradio · PyPI](https://pypi.org/project/gradio/)
- [openai/whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://github.com/openai/whisper)
