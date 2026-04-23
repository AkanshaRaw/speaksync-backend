"""
SpeakSync Backend API
=====================
FastAPI server: LLaMA poem generation + Whisper audio analysis.
Render free-tier optimised: CPU-only, low RAM, graceful fallbacks.
"""

import os
import tempfile
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────
GGUF_PATH    = Path("models/tinyllama.gguf")
WHISPER_SIZE = "tiny"          # ~39 MB, fast on CPU

# ── Global model handles ───────────────────────────────────────────────────
whisper_model = None
llama_model   = None

# ── Fallback responses (API never crashes) ─────────────────────────────────
FALLBACK_POEM = "\n".join([
    "Words rise like morning mist,",
    "Thoughts bloom where silence kissed,",
    "In every breath a song exists,",
    "A poem in the world persists.",
])
FALLBACK_TRANSCRIPTION = "Audio could not be processed at this time."
FALLBACK_FEEDBACK = (
    "Your voice carries meaning and emotion. "
    "Keep practising — clarity and confidence grow with every word you speak."
)

# ── ChatML token pieces (built at runtime to avoid parser issues) ──────────
_PIPE   = chr(124)   # |
_LT     = chr(60)    # <
_GT     = chr(62)    # >
_SL     = chr(47)    # /
_S_TAG  = "s"
_SYS    = "system"
_USR    = "user"
_ASST   = "assistant"

def _tag(role: str) -> str:
    return _LT + _PIPE + role + _PIPE + _GT

def _close() -> str:
    return _LT + _SL + _S_TAG + _GT

SYS_OPEN  = _tag(_SYS)
SYS_CLOSE = _close()
USR_OPEN  = _tag(_USR)
USR_CLOSE = _close()
AST_OPEN  = _tag(_ASST)


# ── Lifespan: load models once at startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    global whisper_model, llama_model

    # Whisper tiny
    try:
        import whisper as _whisper
        log.info("Loading Whisper '%s' ...", WHISPER_SIZE)
        whisper_model = _whisper.load_model(WHISPER_SIZE, device="cpu")
        log.info("Whisper ready.")
    except Exception as exc:
        log.error("Whisper load failed: %s", exc)
        whisper_model = None

    # TinyLlama GGUF
    try:
        if GGUF_PATH.exists():
            from llama_cpp import Llama
            log.info("Loading LLaMA from %s ...", GGUF_PATH)
            llama_model = Llama(
                model_path=str(GGUF_PATH),
                n_ctx=512,
                n_threads=2,
                verbose=False,
            )
            log.info("LLaMA ready.")
        else:
            log.warning(
                "GGUF not found at %s. "
                "Run download_model.py first. Poem endpoint will use fallback.",
                GGUF_PATH,
            )
    except Exception as exc:
        log.error("LLaMA load failed: %s", exc)
        llama_model = None

    yield  # ── server running ──

    llama_model   = None
    whisper_model = None
    log.info("Models released.")


# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="SpeakSync API",
    description="AI-powered poetry generation and speech analysis for iOS",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ────────────────────────────────────────────────────────────────
class PoemRequest(BaseModel):
    text: str

class PoemResponse(BaseModel):
    poem: str

class AudioResponse(BaseModel):
    transcription: str
    feedback: str


# ── Internal helpers ───────────────────────────────────────────────────────
def _make_poem_prompt(theme: str) -> str:
    """Build a TinyLlama ChatML prompt for poem generation."""
    system_msg = (
        "You are a creative poet. "
        "Write a short, beautiful 4-line poem inspired by the given theme. "
        "Return ONLY the 4 poem lines, nothing else."
    )
    user_msg = f"Write a 4-line poem about: {theme}"
    return (
        SYS_OPEN  + "\n" + system_msg + "\n" + SYS_CLOSE + "\n"
        + USR_OPEN  + "\n" + user_msg   + "\n" + USR_CLOSE + "\n"
        + AST_OPEN  + "\n"
    )


def _run_llama(theme: str) -> str:
    """Call LLaMA and return poem text. Raises on failure."""
    prompt = _make_poem_prompt(theme)
    output = llama_model(
        prompt,
        max_tokens=120,
        temperature=0.8,
        top_p=0.9,
        stop=[SYS_CLOSE, SYS_OPEN],
        echo=False,
    )
    return output["choices"][0]["text"].strip()


def _evaluate_speech(transcription: str) -> str:
    """Produce simple heuristic feedback from a transcription string."""
    if not transcription or not transcription.strip():
        return FALLBACK_FEEDBACK

    word_count = len(transcription.split())
    sentences  = [s.strip() for s in transcription.replace("!", ".").replace("?", ".").split(".") if s.strip()]
    sent_count = len(sentences)

    parts = []

    # Fluency heuristic
    if word_count >= 20:
        parts.append("Great job — your response was detailed and well-developed.")
    elif word_count >= 8:
        parts.append("Good effort! You expressed your ideas clearly.")
    else:
        parts.append("Try to elaborate more — longer responses build speaking fluency.")

    # Sentence variety heuristic
    if sent_count >= 3:
        parts.append("You used multiple sentences, which shows good sentence variety.")
    elif sent_count == 2:
        parts.append("Consider adding one more sentence to fully develop your idea.")
    else:
        parts.append("Try breaking your response into more than one sentence for better flow.")

    # Confidence encouragement
    parts.append("Keep practising — every word brings you closer to confident speech!")

    return " ".join(parts)


# ── Routes ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Utility"])
async def health_check():
    """Render health-check endpoint."""
    return {
        "status": "ok",
        "whisper": whisper_model is not None,
        "llama":   llama_model   is not None,
    }


@app.post("/generate-poem", response_model=PoemResponse, tags=["AI"])
async def generate_poem(request: PoemRequest):
    """
    Generate a 4-line poem based on the provided text/theme.

    - **text**: the theme or inspiration text for the poem
    """
    user_text = request.text.strip()
    if not user_text:
        return PoemResponse(poem=FALLBACK_POEM)

    if llama_model is None:
        log.warning("/generate-poem: LLaMA not loaded, returning fallback.")
        return PoemResponse(poem=FALLBACK_POEM)

    try:
        poem = _run_llama(user_text)
        if not poem:
            poem = FALLBACK_POEM
    except Exception as exc:
        log.error("/generate-poem error: %s", exc)
        poem = FALLBACK_POEM

    return PoemResponse(poem=poem)


@app.post("/analyze-audio", response_model=AudioResponse, tags=["AI"])
async def analyze_audio(file: UploadFile = File(...)):
    """
    Transcribe an audio file and return feedback.

    - **file**: WAV or MP3 audio file (multipart/form-data)
    """
    transcription = FALLBACK_TRANSCRIPTION
    feedback      = FALLBACK_FEEDBACK

    # Save upload to a temp file
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        if whisper_model is None:
            log.warning("/analyze-audio: Whisper not loaded, returning fallback.")
            return AudioResponse(transcription=transcription, feedback=feedback)

        # Transcribe
        result = whisper_model.transcribe(tmp_path, fp16=False)
        transcription = (result.get("text") or "").strip() or FALLBACK_TRANSCRIPTION

        # Generate feedback
        feedback = _evaluate_speech(transcription)

    except Exception as exc:
        log.error("/analyze-audio error: %s", exc)
        # Return fallbacks — do NOT re-raise

    finally:
        # Always clean up temp file
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    return AudioResponse(transcription=transcription, feedback=feedback)


# ── Dev entry-point ────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
