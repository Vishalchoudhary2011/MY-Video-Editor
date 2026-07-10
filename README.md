# Reel — Local Video Enhancer

Ek local video enhancement tool: resolution upscale, denoise, color correction, crop, trim — sab CPU pe, koi GPU nahi chahiye.

Tested aur working confirm hai: crop + denoise + color correction + 1080p upscale + trim, sab ek saath verify kiya gaya (640×360 → 1920×1080, trimmed).

## Do backend options hain is folder me

- **`backend-fastapi/`** ← ye use karo (FastAPI, recommended)
- `backend/` ← purana Flask version (reference ke liye rakha hai, dono same kaam karte hain)

Dono ka frontend same hai (`frontend/index.html`), kyunki API routes identical hain.

## Setup (ek baar karna hai)

### 1. FFmpeg install karo
Ye sabse zaroori step hai — backend isi se video process karta hai.

- **Windows:** [ffmpeg.org/download](https://ffmpeg.org/download.html) se download karo, ya agar `winget` hai to: `winget install ffmpeg`
- **Mac:** `brew install ffmpeg`
- **Linux:** `sudo apt install ffmpeg`

Verify karo terminal me:
```
ffmpeg -version
```

### 2. Backend setup (FastAPI)
```bash
cd backend-fastapi
pip install -r requirements.txt
uvicorn app:app --reload --port 5000
```
Ye `http://localhost:5000` pe chalega. Terminal me "Uvicorn running on http://0.0.0.0:5000" dikhna chahiye.

**Bonus:** FastAPI free me auto-generated API docs deta hai — browser me `http://localhost:5000/docs` khol ke dekho, har endpoint test bhi kar sakte ho wahi se.

### 3. Frontend chalao
Bas `frontend/index.html` ko double-click karo — seedha browser me khul jayega (Chrome/Edge/Firefox). Koi npm install ya build step nahi chahiye, React CDN se load hota hai.

**Important:** Backend (step 2) chalu hona chahiye tabhi frontend kaam karega — top-right corner me "backend connected" (green dot) dikhna chahiye.

## Use kaise karo

1. Browser me video drag-drop karo ya click karke select karo
2. Left panel me features on/off karo aur adjust karo:
   - **Resolution** — 720p / 1080p / 1440p / 4K
   - **Noise removal** — light / medium / strong
   - **Color correction** — brightness, contrast, saturation sliders
   - **Crop** — width, height, x/y offset (pixels me)
   - **Trim** — start time aur duration (seconds me)
3. "Run enhancement" click karo
4. Processing complete hone ke baad "Download enhanced video" button milega

## Kaise kaam karta hai (short)

```
Browser (React, CDN) → FastAPI (/api/enhance) → FFmpeg command build hota hai
    → subprocess se run hota hai → processed video outputs/ folder me save hota hai
    → download link return hota hai
```

Saare filters ek hi FFmpeg command me chain hote hain (crop → denoise → color → scale), taaki video sirf ek baar re-encode ho, multiple baar nahi.

## Limitations (important, honestly bata raha hoon)

- Ye **basic quality enhancement** hai (FFmpeg ke traditional algorithms se) — AI-based super-resolution (jaise Topaz Video AI) jitna sharp/detailed nahi milega. 480p ko 4K karoge to upscale hoga lekin naya detail "generate" nahi hoga.
- CPU pe processing time video length aur target resolution pe depend karta hai — 30 sec video, 1080p tak, aam CPU pe usually 1-3 minutes lagte hain. 4K upscale isse zyada time lega.
- Ye tool sirf **quality/format modify** karta hai — kisi bhi video ka **copyright status nahi badalta**. Apna khud ka content ya properly licensed footage hi use karo.

## Agla step (optional)

Agar future me AI-based upscaling chahiye (zyada sharp result), to Real-ESRGAN jaisa lightweight model add kiya ja sakta hai — CPU pe chalega, per slow hoga. Bata dena agar wo add karna ho.
