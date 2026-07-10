"""
Video Enhancer - Backend (FastAPI version)
--------------------------------------------
FastAPI service that takes an uploaded video + enhancement options and runs it
through an FFmpeg pipeline: speed -> crop -> denoise -> color correct -> upscale -> title overlay.

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 5000

Docs (auto-generated, free with FastAPI):
    http://localhost:5000/docs

Requires FFmpeg to be installed and available on PATH.
Check with: ffmpeg -version
"""

import os
import platform
import subprocess
import uuid
import shutil
import json
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import static_ffmpeg
# Downloads ffmpeg/ffprobe binaries on first run (cached after that) and
# prepends them to this process's PATH — no manual install or system PATH
# editing needed.
static_ffmpeg.add_paths()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

app = FastAPI(title="Reel — Local Video Enhancer API")

# allow the React frontend (opened as a local file / different origin) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def get_video_info(path: str) -> dict:
    """Use ffprobe to get width/height/duration of the input video."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-show_entries", "format=duration",
        "-of", "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffprobe not found. Install FFmpeg and make sure its 'bin' folder is on your system PATH, then restart the terminal.",
        )
    try:
        data = json.loads(result.stdout)
        width = data["streams"][0]["width"]
        height = data["streams"][0]["height"]
        duration = float(data["format"]["duration"])
        return {"width": width, "height": height, "duration": duration}
    except Exception:
        return {"width": None, "height": None, "duration": None}


def default_fontfile() -> Optional[str]:
    """
    Best-effort guess at a system font file, used only as a fallback when the
    caller doesn't supply title.fontfile. Returns None if nothing obvious is found.
    """
    system = platform.system()
    candidates = []
    if system == "Windows":
        candidates = [
            "C:/Windows/Fonts/arial.ttf",
            "C:/Windows/Fonts/segoeui.ttf",
        ]
    elif system == "Darwin":
        candidates = [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:  # Linux and others
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def escape_drawtext(text: str) -> str:
    """Escape special characters so ffmpeg's drawtext filter doesn't choke on them."""
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "\\%")
    return text


def build_audio_speed_filters(factor: float) -> list:
    """
    FFmpeg's atempo filter is restricted to a range of 0.5 to 2.0.
    This function breaks down larger or smaller speed adjustments into multiple
    chained filters to maintain consistent audio pitch.
    """
    filters = []
    current = factor
    while current > 2.0:
        filters.append("atempo=2.0")
        current /= 2.0
    while current < 0.5:
        filters.append("atempo=0.5")
        current /= 0.5
    if current != 1.0:
        filters.append(f"atempo={current:.2f}")
    return filters


def build_filter_chain(options: dict) -> tuple[Optional[str], Optional[str]]:
    """
    Build the FFmpeg filter strings.
    Returns a tuple of: (video_filter_chain_string, audio_filter_chain_string)
    """
    video_filters = []
    audio_filters = []

    # --- Speed Control ---
    speed = options.get("speed")
    if speed and speed.get("enabled"):
        factor = float(speed.get("factor", 1.0))
        if factor != 1.0 and factor > 0:
            # Video: setpts scales timestamps inversely (e.g., 2x speed means half the presentation time duration)
            video_filters.append(f"setpts=PTS/{factor}")
            # Audio: chain pitch-preserving atempo filters
            audio_filters.extend(build_audio_speed_filters(factor))

    crop = options.get("crop")
    if crop and crop.get("enabled"):
        w, h, x, y = crop["width"], crop["height"], crop["x"], crop["y"]
        video_filters.append(f"crop={w}:{h}:{x}:{y}")

    denoise = options.get("denoise")
    if denoise and denoise.get("enabled"):
        strength = denoise.get("strength", "medium")
        presets = {
            "light": "hqdn3d=2:1:3:2",
            "medium": "hqdn3d=4:3:6:4",
            "strong": "hqdn3d=8:6:10:8",
        }
        video_filters.append(presets.get(strength, presets["medium"]))

    color = options.get("color")
    if color and color.get("enabled"):
        brightness = color.get("brightness", 0)
        contrast = color.get("contrast", 1.0)
        saturation = color.get("saturation", 1.0)
        video_filters.append(f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}")

    resolution = options.get("resolution")
    if resolution and resolution.get("enabled"):
        target = resolution.get("target", "1080p")
        target_map = {
            "720p": (1280, 720),
            "1080p": (1920, 1080),
            "1440p": (2560, 1440),
            "4k": (3840, 2160),
            "short": (1080, 1920),
        }
        w, h = target_map.get(target, (1920, 1080))
        fill = resolution.get("fill", False)
        if fill:
            video_filters.append(
                f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,"
                f"crop={w}:{h}"
            )
        else:
            video_filters.append(
                f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,"
                f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
            )

    title = options.get("title")
    if title and title.get("enabled") and title.get("text"):
        text = escape_drawtext(title["text"])
        fontsize = title.get("fontsize", 48)
        fontcolor = title.get("fontcolor", "white")
        box = title.get("box", True)
        boxcolor = title.get("boxcolor", "black@0.5")
        top_margin = title.get("top_margin", 30)

        drawtext = (
            f"drawtext=text='{text}':"
            f"fontsize={fontsize}:fontcolor={fontcolor}:"
            f"x=(w-text_w)/2:y={top_margin}"
        )
        if box:
            drawtext += f":box=1:boxcolor={boxcolor}:boxborderw=10"

        fontfile = title.get("fontfile") or default_fontfile()
        if fontfile:
            drawtext += f":fontfile='{fontfile}'"

        video_filters.append(drawtext)

    vf_str = ",".join(video_filters) if video_filters else None
    af_str = ",".join(audio_filters) if audio_filters else None
    return vf_str, af_str


def build_trim_args(options: dict) -> list:
    """Return ffmpeg args for trimming (-ss start -t duration), placed before -i for speed."""
    trim = options.get("trim")
    args = []
    if trim and trim.get("enabled"):
        start = trim.get("start", 0)
        duration = trim.get("duration")
        args += ["-ss", str(start)]
        if duration:
            args += ["-t", str(duration)]
    return args


@app.get("/api/health")
def health():
    return {"status": "ok", "ffmpeg_available": ffmpeg_available()}


@app.post("/api/probe")
async def probe(video: UploadFile = File(...)):
    if not video.filename or not allowed_file(video.filename):
        raise HTTPException(status_code=400, detail="Invalid or missing file")

    job_id = str(uuid.uuid4())
    ext = video.filename.rsplit(".", 1)[1].lower()
    saved_name = f"{job_id}.{ext}"
    input_path = os.path.join(UPLOAD_DIR, saved_name)

    with open(input_path, "wb") as f:
        shutil.copyfileobj(video.file, f)

    info = get_video_info(input_path)
    return {"job_id": job_id, "filename": saved_name, **info}


@app.post("/api/enhance")
async def enhance(filename: str = Form(...), options: str = Form("{}")):
    if not ffmpeg_available():
        raise HTTPException(status_code=500, detail="FFmpeg not found on server. Please install FFmpeg.")

    try:
        opts = json.loads(options)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid options JSON")

    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found. Please re-upload.")

    job_id = str(uuid.uuid4())
    output_filename = f"{job_id}_enhanced.mp4"
    output_path = os.path.join(OUTPUT_DIR, output_filename)

    trim_args = build_trim_args(opts)
    vf_chain, af_chain = build_filter_chain(opts)

    cmd = ["ffmpeg", "-y"] + trim_args + ["-i", input_path]
    if vf_chain:
        cmd += ["-vf", vf_chain]
    if af_chain:
        cmd += ["-af", af_chain]
        
    cmd += [
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="ffmpeg not found. Install FFmpeg and make sure its 'bin' folder is on your system PATH, then restart the terminal.",
        )

    if result.returncode != 0:
        raise HTTPException(
            status_code=500,
            detail={"error": "FFmpeg processing failed", "details": result.stderr[-1500:]},
        )

    return {"job_id": job_id, "status": "done", "download": f"/api/download/{output_filename}"}


@app.get("/api/download/{filename}")
def download(filename: str):
    safe_name = os.path.basename(filename)
    file_path = os.path.join(OUTPUT_DIR, safe_name)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path, media_type="video/mp4", filename=safe_name)


if __name__ == "__main__":
    import uvicorn
    if not ffmpeg_available():
        print("WARNING: ffmpeg was not found on PATH. Install it before processing videos.")
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True)