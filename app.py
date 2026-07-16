"""
Video Enhancer - Backend (FastAPI production version)
--------------------------------------------
FastAPI service that takes an uploaded video + enhancement options and runs it
through an FFmpeg pipeline: speed -> flip -> zoom -> crop -> denoise -> filters 
-> color correct -> upscale -> borders -> animated subtitles -> title overlay.
Supports exporting the entire timeline as 30-second individual sequential chunks.

Also supports a multi-clip timeline workflow: import up to 4 YouTube videos in
parallel, cut multiple clips out of them, arrange the clips in any order,
and merge the arranged sequence into one final exported video.

Run:
    pip install -r requirements.txt
    uvicorn app:app --reload --port 5000

Docs (auto-generated, free with FastAPI):
    http://localhost:5000/docs
"""

import os
import platform
import re
import subprocess
import uuid
import shutil
import json
import zipfile
import threading
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
CLIPS_DIR = os.path.join(BASE_DIR, "clips")
MERGED_DIR = os.path.join(BASE_DIR, "merged")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(CLIPS_DIR, exist_ok=True)
os.makedirs(MERGED_DIR, exist_ok=True)

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

YOUTUBE_URL_RE = re.compile(
    r"^(https?://)?(www\.)?(m\.)?(youtube\.com/(watch\?v=|shorts/|embed/)|youtu\.be/)[\w-]{6,}",
    re.IGNORECASE,
)


def is_valid_youtube_url(url: str) -> bool:
    return bool(YOUTUBE_URL_RE.match((url or "").strip()))


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


# ---------------------------------------------------------------------------
# Encoding performance: auto-detect a GPU encoder if one's available (NVENC /
# QuickSync / VideoToolbox) and map a simple "render speed" choice onto the
# right preset for whichever encoder ends up being used. Long videos are
# dominated by encode time, so this is the single biggest speed lever.
# ---------------------------------------------------------------------------

_HW_ENCODER_CACHE = "unchecked"


def detect_hardware_encoder() -> Optional[str]:
    global _HW_ENCODER_CACHE
    if _HW_ENCODER_CACHE != "unchecked":
        return _HW_ENCODER_CACHE
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        )
        out = result.stdout
        if "h264_nvenc" in out:
            _HW_ENCODER_CACHE = "nvenc"
        elif "h264_qsv" in out:
            _HW_ENCODER_CACHE = "qsv"
        elif "h264_videotoolbox" in out:
            _HW_ENCODER_CACHE = "videotoolbox"
        else:
            _HW_ENCODER_CACHE = None
    except Exception:
        _HW_ENCODER_CACHE = None
    return _HW_ENCODER_CACHE


def get_encode_args(perf: dict) -> list:
    """Returns the -c:v ... video-encoder args for the requested speed/quality
    tradeoff, using a GPU encoder automatically if one's available and not
    explicitly disabled."""
    render_speed = (perf or {}).get("render_speed", "balanced")
    use_hw = (perf or {}).get("hardware_accel", True)
    hw_encoder = detect_hardware_encoder() if use_hw else None

    if hw_encoder == "nvenc":
        nvenc_preset = {"fast": "p1", "balanced": "p4", "quality": "p6"}.get(render_speed, "p4")
        return ["-c:v", "h264_nvenc", "-preset", nvenc_preset, "-cq", "20"]
    if hw_encoder == "qsv":
        qsv_preset = {"fast": "veryfast", "balanced": "fast", "quality": "medium"}.get(render_speed, "fast")
        return ["-c:v", "h264_qsv", "-preset", qsv_preset, "-global_quality", "20"]
    if hw_encoder == "videotoolbox":
        return ["-c:v", "h264_videotoolbox", "-q:v", "60"]

    # CPU fallback (also used if hardware encoding fails at runtime)
    x264_preset = {"fast": "veryfast", "balanced": "fast", "quality": "medium"}.get(render_speed, "fast")
    return ["-c:v", "libx264", "-preset", x264_preset, "-crf", "20"]


# ---------------------------------------------------------------------------
# Job progress tracking — an in-memory store the frontend polls while ffmpeg
# runs, fed by parsing ffmpeg's own `-progress pipe:1` machine-readable
# output (out_time_ms) against an estimated target output duration. This same
# store/poll pattern is reused for the enhance pipeline, YouTube imports,
# clip extraction, and clip merging below.
# ---------------------------------------------------------------------------

PROGRESS_LOCK = threading.Lock()
PROGRESS_STORE = {}


def update_progress(job_id: str, **kwargs):
    with PROGRESS_LOCK:
        PROGRESS_STORE.setdefault(job_id, {}).update(kwargs)


def estimate_output_duration(opts: dict, source_duration: Optional[float]) -> float:
    """Best-effort guess at the final output's duration, used only to turn
    ffmpeg's out_time_ms into a percentage — doesn't need to be exact."""
    source_duration = source_duration or 60.0
    speed_factor = 1.0
    speed_opt = opts.get("speed")
    if speed_opt and speed_opt.get("enabled"):
        f = float(speed_opt.get("factor", 1.0))
        if f > 0:
            speed_factor = f

    trim_opt = opts.get("trim", {})
    base_duration = trim_opt.get("duration", source_duration) if trim_opt.get("enabled") else source_duration
    if not base_duration or base_duration <= 0:
        base_duration = source_duration
    return max(0.5, base_duration / speed_factor)


def run_ffmpeg_with_progress(cmd: list, job_id: str, target_duration: float) -> tuple:
    """Runs ffmpeg with -progress pipe:1 already baked into cmd, streaming
    stdout for progress key=value lines while draining stderr on a separate
    thread (needed for the eventual error message without deadlocking on a
    full pipe buffer). Returns (returncode, stderr_text)."""
    stderr_lines = []

    def drain_stderr(pipe):
        for line in iter(pipe.readline, ""):
            stderr_lines.append(line)

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
    err_thread = threading.Thread(target=drain_stderr, args=(proc.stderr,), daemon=True)
    err_thread.start()

    out_time_sec = 0.0
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("out_time_ms="):
            try:
                out_time_sec = int(line.split("=", 1)[1]) / 1_000_000
            except ValueError:
                pass
        elif line.startswith("out_time="):
            try:
                h, m, s = line.split("=", 1)[1].split(":")
                out_time_sec = int(h) * 3600 + int(m) * 60 + float(s)
            except Exception:
                pass
        elif line.startswith("progress="):
            state = line.split("=", 1)[1]
            percent = (out_time_sec / target_duration) * 100 if target_duration else 0
            percent = max(0, min(percent, 99 if state == "continue" else 100))
            update_progress(job_id, status="processing", percent=round(percent, 1))

    proc.wait()
    err_thread.join(timeout=2)
    return proc.returncode, "".join(stderr_lines)


def run_ffmpeg_with_fallback(input_args: list, vf_chain: Optional[str], af_chain: Optional[str],
                              build_tail, perf_opts: dict, job_id: str, target_duration: float):
    """Runs ffmpeg with the chosen (possibly hardware) encoder; if that fails
    (e.g. an encoder that *looked* available but the driver isn't actually
    working), automatically retries once on CPU (libx264) instead of just
    erroring out."""
    encode_args = get_encode_args(perf_opts)
    cmd = list(input_args)
    if vf_chain: cmd += ["-vf", vf_chain]
    if af_chain: cmd += ["-af", af_chain]
    cmd += build_tail(encode_args)

    update_progress(job_id, status="processing", percent=0)
    returncode, stderr_text = run_ffmpeg_with_progress(cmd, job_id, target_duration)

    if returncode != 0 and encode_args[1] != "libx264":
        # Hardware encoder looked available but failed at runtime — retry on CPU.
        update_progress(job_id, status="processing", percent=0)
        cpu_args = get_encode_args({**(perf_opts or {}), "hardware_accel": False})
        cmd = list(input_args)
        if vf_chain: cmd += ["-vf", vf_chain]
        if af_chain: cmd += ["-af", af_chain]
        cmd += build_tail(cpu_args)
        returncode, stderr_text = run_ffmpeg_with_progress(cmd, job_id, target_duration)

    class _Result:
        pass
    result = _Result()
    result.returncode = returncode
    result.stderr = stderr_text
    return result


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
        # lanczos looks best but is the slowest scaler; bicubic is a good
        # speed/quality tradeoff and is what "fast"/"balanced" use.
        render_speed = (options.get("performance") or {}).get("render_speed", "balanced")
        scale_flags = "lanczos" if render_speed == "quality" else "bicubic"
        if fill:
            video_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=increase:flags={scale_flags},crop={w}:{h}")
        else:
            video_filters.append(f"scale={w}:{h}:force_original_aspect_ratio=decrease:flags={scale_flags},pad={w}:{h}:(ow-iw)/2:(oh-ih)/2")
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
    if not is_valid_youtube_url(url):
        raise HTTPException(status_code=400, detail=f"'{url}' doesn't look like a valid YouTube URL.")

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


def process_enhance_job(job_id: str, opts: dict, input_path: str):
    """Runs the full enhance pipeline (previously the body of /api/enhance)
    on a background thread, writing status/percent/result into
    PROGRESS_STORE as it goes so the frontend can poll for updates."""
    job_temp_dir = os.path.join(TEMP_DIR, job_id)
    os.makedirs(job_temp_dir, exist_ok=True)

    try:
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
        target_duration = estimate_output_duration(opts, source_dims.get("duration"))
        perf_opts = opts.get("performance", {})

        # --- CHUNKED MULTI-EXPORT SYSTEM ---
        if not trim_opt.get("enabled"):
            chunk_size = int(trim_opt.get("chunk_length", 30))
            job_output_dir = os.path.join(OUTPUT_DIR, job_id)
            os.makedirs(job_output_dir, exist_ok=True)

            output_template = os.path.join(job_output_dir, "chunk_%03d.mp4")

            def build_chunk_tail(encode_args):
                return encode_args + [
                    "-c:a", "aac", "-b:a", "192k",
                    "-f", "segment",
                    "-segment_time", str(chunk_size),
                    "-reset_timestamps", "1",
                    "-force_key_frames", f"expr:gte(t,n_forced*{chunk_size})",
                    output_template
                ]

            result = run_ffmpeg_with_fallback(
                ["ffmpeg", "-y", "-progress", "pipe:1", "-nostats", "-i", input_path],
                vf_chain, af_chain, build_chunk_tail, perf_opts, job_id, target_duration
            )
            shutil.rmtree(job_temp_dir, ignore_errors=True)
            if result.returncode != 0:
                update_progress(job_id, status="error", detail=result.stderr[-1500:])
                return

            generated_chunks = sorted(os.listdir(job_output_dir))
            chunks = [
                {"name": name, "download": f"/api/download_chunk/{job_id}/{name}"}
                for name in generated_chunks
            ]
            final_result = {
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
            update_progress(job_id, status="done", percent=100, result=final_result)

        # --- STANDARD SINGLE FILE EXPORT ---
        else:
            output_filename = f"{job_id}_enhanced.mp4"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            start = trim_opt.get("start", 0)
            duration = trim_opt.get("duration", 30)

            def build_single_tail(encode_args):
                return encode_args + ["-c:a", "aac", "-b:a", "192k", output_path]

            result = run_ffmpeg_with_fallback(
                ["ffmpeg", "-y", "-progress", "pipe:1", "-nostats", "-ss", str(start), "-t", str(duration), "-i", input_path],
                vf_chain, af_chain, build_single_tail, perf_opts, job_id, target_duration
            )
            shutil.rmtree(job_temp_dir, ignore_errors=True)
            if result.returncode != 0:
                update_progress(job_id, status="error", detail=result.stderr[-1500:])
                return

            final_result = {
                "job_id": job_id, "status": "done", "is_segmented": False,
                "download": f"/api/download/{output_filename}"
            }
            update_progress(job_id, status="done", percent=100, result=final_result)

    except Exception as e:
        shutil.rmtree(job_temp_dir, ignore_errors=True)
        update_progress(job_id, status="error", detail=str(e))


@app.post("/api/enhance")
def enhance(filename: str = Form(...), options: str = Form("{}")):
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
    update_progress(job_id, status="queued", percent=0)

    thread = threading.Thread(target=process_enhance_job, args=(job_id, opts, input_path), daemon=True)
    thread.start()

    return {"job_id": job_id, "status": "started"}


@app.get("/api/progress/{job_id}")
def get_progress(job_id: str):
    with PROGRESS_LOCK:
        data = PROGRESS_STORE.get(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Unknown job id.")
    return data


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


# =============================================================================
# MULTI-CLIP TIMELINE WORKFLOW
# -----------------------------------------------------------------------------
# Import up to 4 YouTube videos in parallel, cut multiple clips out of any of
# them, arrange the resulting clips in any order, and merge that ordered
# sequence into one final exported video. Every long-running step (import /
# extraction / merge) is dispatched onto its own background thread and
# reports progress through the same PROGRESS_STORE / /api/progress/{id}
# polling contract already used by the enhance pipeline above, so the
# frontend can track import, trimming, and merging with one shared mechanism.
# =============================================================================

MAX_BATCH_IMPORTS = 4

CLIP_LOCK = threading.Lock()
CLIP_STORE = {}  # clip_id -> clip metadata


def process_youtube_import_job(job_id: str, url: str):
    """Background job body for a single URL within a batch import. Writes
    into the shared PROGRESS_STORE under `job_id` exactly like every other
    job type, so the existing polling endpoint/pattern works unchanged."""
    url = (url or "").strip()
    update_progress(job_id, status="processing", percent=5, stage="Validating URL", url=url)

    if not url:
        update_progress(job_id, status="error", detail="Empty YouTube URL.", url=url)
        return
    if not is_valid_youtube_url(url):
        update_progress(job_id, status="error", detail=f"'{url}' doesn't look like a valid YouTube URL.", url=url)
        return

    try:
        update_progress(job_id, status="processing", percent=15, stage="Downloading", url=url)
        path = download_youtube_video(url, job_id)
        update_progress(job_id, status="processing", percent=90, stage="Reading video info", url=url)
        info = get_video_info(path)
        if not info.get("duration"):
            update_progress(job_id, status="error", detail="Downloaded, but the video appears to be unreadable/corrupt.", url=url)
            return
        result = {"filename": os.path.basename(path), "source_url": url, **info}
        update_progress(job_id, status="done", percent=100, result=result, url=url)
    except HTTPException as e:
        update_progress(job_id, status="error", detail=e.detail, url=url)
    except Exception as e:
        update_progress(job_id, status="error", detail=f"Unexpected error importing this video: {e}", url=url)


@app.post("/api/import_youtube_batch")
def import_youtube_batch(urls: str = Form(...)):
    """Kicks off up to MAX_BATCH_IMPORTS parallel YouTube downloads. Returns
    immediately with one job_id per URL; poll each with /api/progress/{id}."""
    try:
        url_list = json.loads(urls)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid urls JSON — expected a JSON array of strings.")

    if not isinstance(url_list, list) or not url_list:
        raise HTTPException(status_code=400, detail="Provide a non-empty list of YouTube URLs.")
    if len(url_list) > MAX_BATCH_IMPORTS:
        raise HTTPException(status_code=400, detail=f"Import up to {MAX_BATCH_IMPORTS} videos at a time.")

    jobs = []
    for url in url_list:
        job_id = str(uuid.uuid4())
        update_progress(job_id, status="queued", percent=0, url=url)
        thread = threading.Thread(target=process_youtube_import_job, args=(job_id, url), daemon=True)
        thread.start()
        jobs.append({"job_id": job_id, "url": url})

    return {"jobs": jobs}


def process_clip_extraction_job(job_id: str, filename: str, start: float, duration: float, label: str):
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        update_progress(job_id, status="error", detail="Source video not found — it may not have finished importing.")
        return

    clip_id = job_id
    output_path = os.path.join(CLIPS_DIR, f"{clip_id}.mp4")
    cmd = [
        "ffmpeg", "-y", "-progress", "pipe:1", "-nostats",
        "-ss", str(max(0.0, start)), "-i", input_path, "-t", str(duration),
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]

    update_progress(job_id, status="processing", percent=0)
    returncode, stderr_text = run_ffmpeg_with_progress(cmd, job_id, max(duration, 0.5))

    if returncode != 0:
        update_progress(job_id, status="error", detail=stderr_text[-1500:] or "Clip extraction failed.")
        return

    info = get_video_info(output_path)
    if not info.get("duration"):
        update_progress(job_id, status="error", detail="Extraction finished but the clip came out unreadable — try a different start time.")
        return

    clip_meta = {
        "clip_id": clip_id,
        "source_filename": filename,
        "label": label or filename,
        "start": start,
        "duration": duration,
        "download": f"/api/download_clip/{clip_id}",
        **info,
    }
    with CLIP_LOCK:
        CLIP_STORE[clip_id] = {**clip_meta, "path": output_path}
    update_progress(job_id, status="done", percent=100, result=clip_meta)


@app.post("/api/extract_clip")
def extract_clip(filename: str = Form(...), start: float = Form(...), duration: float = Form(...), label: str = Form("")):
    """Starts one clip extraction job. The frontend fires several of these at
    once (one per requested clip) to extract clips in parallel.

    Clip length is unrestricted — the only requirements are that the start
    time is non-negative, the resulting clip has positive duration (i.e. the
    end time is after the start time), and the clip doesn't run past the end
    of the source video."""
    input_path = os.path.join(UPLOAD_DIR, filename)
    if not os.path.exists(input_path):
        raise HTTPException(status_code=404, detail="That source video hasn't finished importing yet.")
    if start < 0:
        raise HTTPException(status_code=400, detail="Start time can't be negative.")
    if duration <= 0:
        raise HTTPException(status_code=400, detail="End time must be after the start time.")

    source_info = get_video_info(input_path)
    source_duration = source_info.get("duration")
    if source_duration and (start + duration) > source_duration + 0.5:
        raise HTTPException(
            status_code=400,
            detail="That start time + length runs past the end of the source video.",
        )

    job_id = str(uuid.uuid4())
    update_progress(job_id, status="queued", percent=0)
    thread = threading.Thread(
        target=process_clip_extraction_job, args=(job_id, filename, start, duration, label), daemon=True
    )
    thread.start()
    return {"job_id": job_id, "clip_id": job_id}


@app.get("/api/download_clip/{clip_id}")
def download_clip(clip_id: str):
    with CLIP_LOCK:
        meta = CLIP_STORE.get(clip_id)
    if not meta or not os.path.exists(meta["path"]):
        raise HTTPException(status_code=404, detail="Clip not found.")
    return FileResponse(meta["path"], media_type="video/mp4", filename=f"{clip_id}.mp4")


def process_merge_job(job_id: str, clip_ids: list):
    with CLIP_LOCK:
        clips = []
        for cid in clip_ids:
            meta = CLIP_STORE.get(cid)
            if not meta or not os.path.exists(meta["path"]):
                update_progress(job_id, status="error", detail=f"A clip in your timeline is missing (id: {cid}) — it may have failed extraction. Remove it and try again.")
                return
            clips.append(meta)

    if not clips:
        update_progress(job_id, status="error", detail="No clips to merge.")
        return

    # Normalize every clip to the first clip's resolution and a common 30fps
    # / 48kHz stereo audio format inside the filtergraph, so clips pulled
    # from different source videos (different resolutions/framerates) still
    # concatenate cleanly into one continuous output.
    target_w = clips[0].get("width") or 1920
    target_h = clips[0].get("height") or 1080
    total_duration = sum((c.get("duration") or 0) for c in clips) or 1.0

    cmd = ["ffmpeg", "-y", "-progress", "pipe:1", "-nostats"]
    for c in clips:
        cmd += ["-i", c["path"]]

    filter_parts = []
    concat_inputs = ""
    for i in range(len(clips)):
        filter_parts.append(
            f"[{i}:v]scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
            f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30[v{i}]"
        )
        filter_parts.append(f"[{i}:a]aresample=48000,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]")
        concat_inputs += f"[v{i}][a{i}]"
    filter_parts.append(f"{concat_inputs}concat=n={len(clips)}:v=1:a=1[outv][outa]")
    filter_complex = ";".join(filter_parts)

    output_name = f"{job_id}_final.mp4"
    output_path = os.path.join(MERGED_DIR, output_name)
    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[outv]", "-map", "[outa]",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        output_path,
    ]

    update_progress(job_id, status="processing", percent=0)
    returncode, stderr_text = run_ffmpeg_with_progress(cmd, job_id, total_duration)

    if returncode != 0:
        update_progress(job_id, status="error", detail=stderr_text[-1500:] or "Merging clips failed.")
        return

    update_progress(job_id, status="done", percent=100, result={
        "download": f"/api/download_merged/{output_name}",
        "clip_count": len(clips),
        "duration": total_duration,
    })


@app.post("/api/merge_clips")
def merge_clips(clip_ids: str = Form(...)):
    """Merges an ordered list of previously-extracted clip_ids into one final
    video, preserving the given order."""
    if not ffmpeg_available():
        raise HTTPException(status_code=500, detail="FFmpeg not found on server.")
    try:
        ids = json.loads(clip_ids)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid clip_ids JSON — expected a JSON array of clip ids.")
    if not isinstance(ids, list) or not ids:
        raise HTTPException(status_code=400, detail="Select at least one clip to merge.")

    job_id = str(uuid.uuid4())
    update_progress(job_id, status="queued", percent=0)
    thread = threading.Thread(target=process_merge_job, args=(job_id, ids), daemon=True)
    thread.start()
    return {"job_id": job_id}


@app.get("/api/download_merged/{filename}")
def download_merged(filename: str):
    file_path = os.path.join(MERGED_DIR, os.path.basename(filename))
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File missing")
    return FileResponse(file_path, media_type="video/mp4", filename=filename)