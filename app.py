"""
Video Enhancer - Backend (FastAPI production version)
--------------------------------------------
FastAPI service that takes an uploaded video + enhancement options and runs it
through an FFmpeg pipeline: speed -> flip -> zoom -> crop -> denoise -> filters 
-> color correct -> upscale -> borders -> animated subtitles -> title overlay.
Supports exporting the entire timeline as 30-second individual sequential chunks.

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 5000

Docs (auto-generated, free with FastAPI):
    http://localhost:5000/docs
"""

import os
import platform
import subprocess
import uuid
import shutil
import json
import zipfile
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

import static_ffmpeg
static_ffmpeg.add_paths()

try:
    from faster_whisper import WhisperModel
except ImportError:
    WhisperModel = None

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv", "webm"}

RESOLUTION_MAP = {
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "4k": (3840, 2160),
    "short": (1080, 1920),
}

app = FastAPI(title="Reel — Local Video Enhancer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


# ---------------------------------------------------------------------------
# Speech-to-text (auto subtitle generation) via faster-whisper
# ---------------------------------------------------------------------------

_WHISPER_MODEL_CACHE = {}
VALID_WHISPER_SIZES = {"tiny", "base", "small", "medium"}


def get_whisper_model(model_size: str):
    if WhisperModel is None:
        raise HTTPException(
            status_code=500,
            detail="Auto-subtitles require the 'faster-whisper' package. Install it with: pip install faster-whisper",
        )
    if model_size not in VALID_WHISPER_SIZES:
        model_size = "base"
    if model_size not in _WHISPER_MODEL_CACHE:
        _WHISPER_MODEL_CACHE[model_size] = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _WHISPER_MODEL_CACHE[model_size]


def transcribe_audio(path: str, model_size: str = "base", language: Optional[str] = None) -> tuple:
    model = get_whisper_model(model_size)
    segments, info = model.transcribe(
        path,
        beam_size=5,
        language=language or None,
        vad_filter=True,
    )
    entries = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        entries.append({"start": round(seg.start, 2), "end": round(seg.end, 2), "text": text})
    detected_language = language or (getattr(info, "language", None) or "en")
    return entries, detected_language


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# YouTube import via yt-dlp — downloads straight into UPLOAD_DIR so the
# result can flow through the exact same probe/enhance pipeline as a manual
# file upload. Only use this on content you have the rights to use.
# ---------------------------------------------------------------------------

def download_youtube_video(url: str, job_id: str) -> str:
    if yt_dlp is None:
        raise HTTPException(
            status_code=500,
            detail="YouTube import requires the 'yt-dlp' package. Install it with: pip install yt-dlp",
        )

    output_template = os.path.join(UPLOAD_DIR, f"{job_id}.%(ext)s")
    ydl_opts = {
        # Cap at 1080p — plenty for the enhancement pipeline and keeps
        # downloads/processing fast. Falls back gracefully if mp4 streams
        # aren't available for a given video.
        "format": "bestvideo[ext=mp4][height<=1080]+bestaudio[ext=m4a]/best[ext=mp4][height<=1080]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "restrictfilenames": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            raw_path = ydl.prepare_filename(info)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not download that YouTube video: {e}")

    # merge_output_format=mp4 means the final file should be job_id.mp4 even
    # if prepare_filename() reported a pre-merge extension.
    mp4_path = os.path.join(UPLOAD_DIR, f"{job_id}.mp4")
    if os.path.exists(mp4_path):
        return mp4_path
    if os.path.exists(raw_path):
        return raw_path
    raise HTTPException(status_code=500, detail="Download finished but the output file could not be located.")


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
        raise HTTPException(status_code=500, detail="ffprobe not found on PATH.")
        
    try:
        data = json.loads(result.stdout)
        width = data["streams"][0]["width"]
        height = data["streams"][0]["height"]
        duration = float(data["format"]["duration"])
        return {"width": width, "height": height, "duration": duration}
    except Exception:
        return {"width": None, "height": None, "duration": None}


def get_audio_sample_rate(path: str) -> Optional[int]:
    """Probe the input's audio sample rate (Hz) so pitch-shifting math lands
    on the correct rate instead of assuming a fixed 44100/48000."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate",
        "-of", "json",
        path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return int(data["streams"][0]["sample_rate"])
    except Exception:
        return None


def default_fontfile() -> Optional[str]:
    system = platform.system()
    candidates = []
    if system == "Windows":
        candidates = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/Calibri.ttf"]
    elif system == "Darwin":
        candidates = ["/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc"]
    else:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf"
        ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def default_fontname() -> str:
    system = platform.system()
    if system == "Windows":
        return "Arial"
    if system == "Darwin":
        return "Helvetica"
    return "DejaVu Sans"


def escape_drawtext(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace(":", "\\:")
    text = text.replace("'", "\\'")
    text = text.replace("%", "\\%")
    return text


def escape_filter_path(path: str) -> str:
    """Safely normalizes windows drive parameters and structures for FFmpeg filtergraphs."""
    p = path.replace("\\", "/")
    p = p.replace(":", "\\:")
    p = p.replace("'", "\\'")
    return p


def build_audio_speed_filters(factor: float) -> list:
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


VOICE_PRESETS = {
    # pitch_ratio: >1.0 raises pitch, <1.0 lowers it. Tempo is compensated
    # automatically so playback speed is unaffected by the pitch shift.
    "none": 1.0,
    "deep": 0.8,
    "high": 1.3,
    "robot": 0.75,  # paired with extra ring-mod-ish flavor below
}


def build_voice_filters(voice: dict, sample_rate: int) -> list:
    """Pitch-shift the audio track using the classic asetrate+atempo+aresample
    trick: change the sample rate to bend pitch, then correct tempo back to
    normal with atempo, then resample back to the original rate so the
    output encodes cleanly."""
    preset = voice.get("preset", "custom")
    if preset in VOICE_PRESETS and preset != "custom":
        ratio = VOICE_PRESETS[preset]
    else:
        ratio = float(voice.get("pitch_ratio", 1.0))

    if ratio <= 0 or abs(ratio - 1.0) < 0.001:
        return []

    sr = sample_rate or 44100
    new_rate = max(4000, int(round(sr * ratio)))
    filters = [f"asetrate={new_rate}"]
    # atempo compensates the speed-up/slow-down asetrate introduces so only
    # pitch changes, not duration. atempo only accepts 0.5–2.0 per stage.
    tempo_factor = 1.0 / ratio
    filters.extend(build_audio_speed_filters(tempo_factor))
    filters.append(f"aresample={sr}")

    if preset == "robot":
        # Cheap, dependency-free "robotic" flavor: a fast, shallow tremolo on
        # top of the lowered pitch reads as mechanical/vocoder-ish.
        filters.append("vibrato=f=10:d=0.6")

    return filters


# ---------------------------------------------------------------------------
# Animated subtitle (ASS) generation
# ---------------------------------------------------------------------------

def seconds_to_ass_time(sec: float) -> str:
    if sec < 0:
        sec = 0
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


def escape_ass_text(text: str) -> str:
    text = text.replace("\\", "\\\\")
    text = text.replace("{", "\\{").replace("}", "\\}")
    text = text.replace("\r\n", "\\N").replace("\n", "\\N")
    return text


def hex_to_ass_color(hex_color: str, alpha: str = "00") -> str:
    hex_color = (hex_color or "#FFFFFF").lstrip("#")
    if len(hex_color) != 6:
        hex_color = "FFFFFF"
    r, g, b = hex_color[0:2], hex_color[2:4], hex_color[4:6]
    return f"&H{alpha}{b}{g}{r}".upper()


def build_subtitle_animation_tags(animation: str, video_w: int, video_h: int, margin_v: int):
    cx = video_w // 2
    y_final = video_h - margin_v

    if animation == "fade":
        return "{\\fad(280,220)}", False
    if animation == "slide":
        y_start = min(video_h, y_final + 60)
        return f"{{\\move({cx},{y_start},{cx},{y_final},0,260)\\fad(120,180)\\an2}}", True
    if animation == "pop":
        return "{\\fscx60\\fscy60\\t(0,180,\\fscx100\\fscy100)\\fad(0,150)}", False
    return "", False


def build_subtitle_ass(entries, out_path, video_w, video_h, style_opts, animation):
    fontsize = int(style_opts.get("fontsize", max(28, video_h // 22)))
    fontcolor_hex = style_opts.get("fontcolor", "#FFFFFF")
    outline_hex = style_opts.get("outlinecolor", "#000000")
    margin_v = int(style_opts.get("margin_v", max(40, video_h // 14)))

    primary = hex_to_ass_color(fontcolor_hex)
    outline = hex_to_ass_color(outline_hex)
    back = "&H64000000"
    
    # FIX: Point Windows directly to the hardcoded file path of the system font.
    # Windows FFmpeg handles .ass structures infinitely better when tracking fontfile locations.
    fontname = "C:\\Windows\\Fonts\\arial.ttf"
    if not os.path.exists(fontname):
        fontname = style_opts.get("fontname") or default_fontname()

    override_prefix, _ = build_subtitle_animation_tags(animation, video_w, video_h, margin_v)

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {video_w}
PlayResY: {video_h}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{fontname},{fontsize},{primary},&H000000FF,{outline},{back},-1,0,0,0,100,100,0,0,1,2,1,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = [header]
    for entry in entries:
        start = float(entry.get("start", 0))
        end = float(entry.get("end", start + 2))
        if end <= start:
            end = start + 1.5
        text = escape_ass_text(str(entry.get("text", "")).strip())
        if not text:
            continue
        lines.append(
            f"Dialogue: 0,{seconds_to_ass_time(start)},{seconds_to_ass_time(end)},Default,,0,0,0,,{override_prefix}{text}\n"
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def build_ticker_filter(ticker: dict) -> str:
    text = escape_drawtext(ticker.get("text", ""))
    fontsize = ticker.get("fontsize", 42)
    fontcolor = ticker.get("fontcolor", "black")
    speed = float(ticker.get("speed", 120))
    if speed <= 0:
        speed = 1
    bar_height = int(ticker.get("bar_height", 90))
    bg_color = ticker.get("bg_color", "white")

    drawbox = f"drawbox=x=0:y=0:w=iw:h={bar_height}:color={bg_color}:t=fill"
    x_expr = f"w-mod(t*{speed}\\,w+text_w)"
    y_expr = f"({bar_height}-text_h)/2"

    drawtext = f"drawtext=text='{text}':fontsize={fontsize}:fontcolor={fontcolor}:x={x_expr}:y={y_expr}"
    fontfile = ticker.get("fontfile") or default_fontfile()
    if fontfile:
        drawtext += f":fontfile='{escape_filter_path(fontfile)}'"

    return f"{drawbox},{drawtext}"


def build_filter_chain(options: dict, work_dir: str, source_dims: dict) -> tuple[Optional[str], Optional[str]]:
    video_filters = []
    audio_filters = []

    # 1. Speed Adjustments
    speed = options.get("speed")
    if speed and speed.get("enabled"):
        factor = float(speed.get("factor", 1.0))
        if factor != 1.0 and factor > 0:
            video_filters.append(f"setpts=PTS/{factor}")
            audio_filters.extend(build_audio_speed_filters(factor))

    # 1b. Voice Changer (pitch shift, independent of playback speed)
    voice = options.get("voice")
    if voice and voice.get("enabled"):
        sample_rate = source_dims.get("sample_rate") or 44100
        audio_filters.extend(build_voice_filters(voice, sample_rate))

    # 2. Mirror Flips
    transform = options.get("transform", {})
    if transform.get("enabled"):
        if transform.get("hflip"):
            video_filters.append("hflip")
        if transform.get("vflip"):
            video_filters.append("vflip")

    # 3. Scale Zoom
    if transform.get("enabled") and float(transform.get("zoom", 1.0)) > 1.0:
        z_factor = float(transform["zoom"])
        video_filters.append(f"scale={z_factor}*iw:-1,crop=iw/{z_factor}:ih/{z_factor}")

    # 4. Standard Frame Cropping
    crop = options.get("crop")
    if crop and crop.get("enabled"):
        w, h, x, y = crop["width"], crop["height"], crop["x"], crop["y"]
        video_filters.append(f"crop={w}:{h}:{x}:{y}")

    # 5. Denoise Logic
    denoise = options.get("denoise")
    if denoise and denoise.get("enabled"):
        strength = denoise.get("strength", "medium")
        presets = {
            "light": "hqdn3d=2:1:3:2",
            "medium": "hqdn3d=4:3:6:4",
            "strong": "hqdn3d=8:6:10:8",
        }
        video_filters.append(presets.get(strength, presets["medium"]))

    # 6. Aesthetic Preset Filters
    visual_filter = options.get("filter", {})
    if visual_filter.get("enabled"):
        preset = visual_filter.get("preset", "none").lower()
        if preset == "vintage":
            video_filters.append("curves=vintage")
        elif preset == "bw" or preset == "monochrome":
            video_filters.append("hue=s=0")
        elif preset == "cyberpunk":
            video_filters.append("curves=r='0/0 0.5/0.3 1/1':g='0/0 0.5/0.4 1/0.9':b='0/0 0.5/0.7 1/1'")
        elif preset == "cinematic":
            video_filters.append("eq=contrast=1.15:saturation=1.1,curves=all='0/0 0.3/0.2 0.7/0.8 1/1'")

    # 7. Base Level Color Correction
    color = options.get("color")
    if color and color.get("enabled"):
        brightness = color.get("brightness", 0)
        contrast = color.get("contrast", 1.0)
        saturation = color.get("saturation", 1.0)
        video_filters.append(f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}")

    out_w = crop.get("width") if crop and crop.get("enabled") else source_dims.get("width") or 1080
    out_h = crop.get("height") if crop and crop.get("enabled") else source_dims.get("height") or 1920

    # 8. Frame Target Resolution Up-scaling
    resolution = options.get("resolution")
    if resolution and resolution.get("enabled"):
        target = resolution.get("target", "1080p")
        w, h = RESOLUTION_MAP.get(target, (1920, 1080))
        fill = resolution.get("fill", True)
        if fill:
            video_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=increase:flags=lanczos,crop={w}:{h}")
        else:
            video_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags=lanczos,pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")
        out_w, out_h = w, h

    # 9. Physical Frame Borders (Solid / Blurred background padding)
    borders = options.get("borders", {})
    if borders.get("enabled"):
        b_type = borders.get("type", "solid").lower()
        thickness = int(borders.get("thickness", 20))
        b_color = borders.get("color", "black")

        if b_type == "solid":
            video_filters.append(f"scale=iw-{2*thickness}:ih-{2*thickness},pad=iw+{2*thickness}:ih+{2*thickness}:{thickness}:{thickness}:{b_color}")
        elif b_type == "blurred":
            video_filters.append(f"split=2[bg][fg];[bg]boxblur=20:5,scale={out_w}:{out_h}[bg_scaled];[fg]scale={out_w-2*thickness}:{out_h-2*thickness}[fg_scaled];[bg_scaled][fg_scaled]overlay={thickness}:{thickness}")

    # 10. News Ticker Overlay
    ticker = options.get("ticker")
    if ticker and ticker.get("enabled") and ticker.get("text"):
        video_filters.append(build_ticker_filter(ticker))

    # 11. Subtitle Muxing Graphics Layer
    subtitles = options.get("subtitles")
    if subtitles and subtitles.get("enabled") and subtitles.get("entries"):
        ass_path = os.path.join(work_dir, "subtitles.ass")
        style_opts = {
            "fontsize": subtitles.get("fontsize"),
            "fontcolor": subtitles.get("fontcolor", "#FFFFFF"),
            "outlinecolor": subtitles.get("outlinecolor", "#000000"),
            "margin_v": subtitles.get("margin_v"),
            "fontname": subtitles.get("fontname"),
        }
        style_opts = {k: v for k, v in style_opts.items() if v is not None}
        animation = subtitles.get("animation", "fade")
        build_subtitle_ass(subtitles["entries"], ass_path, out_w, out_h, style_opts, animation)
        
        # Fixed path escaping syntax wrapper specifically for the FFmpeg subtitles filter on Windows systems
        escaped_ass_path = escape_filter_path(ass_path)
        video_filters.append(f"subtitles=filename='{escaped_ass_path}'")

    # 12. Static Title Banner Overlay
    title = options.get("title")
    if title and title.get("enabled") and title.get("text"):
        text = escape_drawtext(title["text"])
        fontsize = title.get("fontsize", 48)
        fontcolor = title.get("fontcolor", "white")
        box = title.get("box", True)
        boxcolor = title.get("boxcolor", "black@0.5")
        top_margin = title.get("top_margin", 30)

        drawtext = f"drawtext=text='{text}':fontsize={fontsize}:fontcolor={fontcolor}:x=(w-text_w)/2:y={top_margin}"
        if box:
            drawtext += f":box=1:boxcolor={boxcolor}:boxborderw=10"
        fontfile = title.get("fontfile") or default_fontfile()
        if fontfile:
            drawtext += f":fontfile='{escape_filter_path(fontfile)}'"
        video_filters.append(drawtext)

    vf_str = ",".join(video_filters) if video_filters else None
    af_str = ",".join(audio_filters) if audio_filters else None
    return vf_str, af_str


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


@app.post("/api/import_youtube")
def import_youtube(url: str = Form(...)):
    """Downloads a YouTube video straight into the uploads folder and returns
    the same shape as /api/probe, so the frontend can treat it identically
    to a manually-uploaded file from this point on."""
    url = (url or "").strip()
    if not url:
        raise HTTPException(status_code=400, detail="A YouTube URL is required.")

    job_id = str(uuid.uuid4())
    downloaded_path = download_youtube_video(url, job_id)
    saved_name = os.path.basename(downloaded_path)
    info = get_video_info(downloaded_path)
    return {"job_id": job_id, "filename": saved_name, **info}


@app.get("/api/preview/{filename}")
def preview_source(filename: str):
    """Streams the original uploaded/imported file so the frontend can show
    a before-processing preview player."""
    file_path = os.path.join(UPLOAD_DIR, os.path.basename(filename))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File missing")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@app.post("/api/transcribe")
def transcribe(filename: str = Form(...), model_size: str = Form("base"), language: str = Form("")):
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found.")

    try:
        entries, detected_language = transcribe_audio(input_path, model_size=model_size, language=language.strip() or None)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")

    return {"entries": entries, "cue_count": len(entries), "language": detected_language}


@app.post("/api/enhance")
async def enhance(filename: str = Form(...), options: str = Form("{}")):
    if not ffmpeg_available():
        raise HTTPException(status_code=500, detail="FFmpeg not found on server.")

    try:
        opts = json.loads(options)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid options JSON")

    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="Uploaded file not found.")

    job_id = str(uuid.uuid4())
    job_temp_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_temp_dir, exist_ok=True)

    trim_opt = opts.get("trim", {})

    if trim_opt.get("enabled"):
        clip_start = float(trim_opt.get("start", 0))
        subs = opts.get("subtitles")
        if subs and subs.get("enabled") and subs.get("entries") and clip_start:
            shifted = []
            for e in subs["entries"]:
                s = float(e.get("start", 0)) - clip_start
                en = float(e.get("end", s + 2)) - clip_start
                if en <= 0:
                    continue
                shifted.append({**e, "start": max(0, s), "end": en})
            subs["entries"] = shifted

    source_dims = get_video_info(input_path)
    source_dims["sample_rate"] = get_audio_sample_rate(input_path)
    vf_chain, af_chain = build_filter_chain(opts, job_temp_dir, source_dims)

    # --- CHUNKED MULTI-EXPORT SYSTEM ---
    if not trim_opt.get("enabled"):
        chunk_size = int(trim_opt.get("chunk_length", 30))
        job_output_dir = os.path.join(OUTPUT_DIR, job_id)
        os.makedirs(job_output_dir, exist_ok=True)

        output_template = os.path.join(job_output_dir, "chunk_%03d.mp4")

        cmd = ["ffmpeg", "-y", "-i", input_path]
        if vf_chain: cmd += ["-vf", vf_chain]
        if af_chain: cmd += ["-af", af_chain]

        cmd += [
            "-c:v", "libx264", "-preset", "medium", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            "-f", "segment",
            "-segment_time", str(chunk_size),
            "-reset_timestamps", "1",
            "-force_key_frames", f"expr:gte(t,n_forced*{chunk_size})",
            output_template
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail={"error": "Transcription track compilation failed", "details": result.stderr[-1500:]})

        generated_chunks = sorted(os.listdir(job_output_dir))
        chunks = [
            {
                "name": name,
                "download": f"/api/download_chunk/{job_id}/{name}",
            }
            for name in generated_chunks
        ]
        return {
            "job_id": job_id,
            "status": "done",
            "is_segmented": True,
            "chunk_count": len(generated_chunks),
            "chunk_length": chunk_size,
            "directory_location": job_output_dir,
            "chunks": chunks,
            "download_all_zip": f"/api/download_all_chunks/{job_id}",
            "download": f"/api/download_chunk/{job_id}/{generated_chunks[0]}" if generated_chunks else None
        }

    # --- STANDARD SINGLE FILE EXPORT ---
    else:
        output_filename = f"{job_id}_enhanced.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)

        start = trim_opt.get("start", 0)
        duration = trim_opt.get("duration", 30)

        cmd = ["ffmpeg", "-y", "-ss", str(start), "-t", str(duration), "-i", input_path]
        if vf_chain: cmd += ["-vf", vf_chain]
        if af_chain: cmd += ["-af", af_chain]
        cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "20", "-c:a", "aac", "-b:a", "192k", output_path]

        result = subprocess.run(cmd, capture_output=True, text=True)
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        if result.returncode != 0:
            raise HTTPException(status_code=500, detail={"error": "Transcription track compilation failed", "details": result.stderr[-1500:]})

        return {"job_id": job_id, "status": "done", "is_segmented": False, "download": f"/api/download/{output_filename}"}


@app.get("/api/download_all_chunks/{job_id}")
def download_all_chunks(job_id: str):
    job_output_dir = os.path.join(OUTPUT_DIR, os.path.basename(job_id))
    if not os.path.isdir(job_output_dir):
        raise HTTPException(status_code=404, detail="No chunks found for this job.")

    chunk_files = sorted(f for f in os.listdir(job_output_dir) if f.lower().endswith(".mp4"))
    if not chunk_files:
        raise HTTPException(status_code=404, detail="No chunks found for this job.")

    zip_path = os.path.join(OUTPUT_DIR, f"{job_id}_chunks.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name in chunk_files:
            zf.write(os.path.join(job_output_dir, name), arcname=name)

    return FileResponse(zip_path, media_type="application/zip", filename=f"reel_chunks_{job_id[:8]}.zip")


@app.get("/api/download/{filename}")
def download(filename: str):
    file_path = os.path.join(OUTPUT_DIR, os.path.basename(filename))
    if not os.path.exists(file_path): 
        raise HTTPException(status_code=404, detail="File missing")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)


@app.get("/api/download_chunk/{job_id}/{chunk_name}")
def download_chunk(job_id: str, chunk_name: str):
    file_path = os.path.join(OUTPUT_DIR, os.path.basename(job_id), os.path.basename(chunk_name))
    if not os.path.exists(file_path): 
        raise HTTPException(status_code=404, detail="Chunk item missing")
    return FileResponse(file_path, media_type="video/mp4", filename=chunk_name)