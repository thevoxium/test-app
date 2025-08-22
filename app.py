
# app.py
# Minimal, single-button voice interview assistant with resume context.
# Deps:
#   pip install streamlit groq streamlit-mic-recorder pypdf
#
# Env:
#   export GROQ_API_KEY=your_key

import os
import io
from typing import List, Dict, Optional

import streamlit as st
from streamlit_mic_recorder import mic_recorder
from groq import Groq

# -------- Optional PDF extraction (pypdf) --------
_PDF_OK = True
try:
    from pypdf import PdfReader
except Exception:
    _PDF_OK = False


# =========================
# Config
# =========================
STT_MODEL = "whisper-large-v3-turbo"
CHAT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are role-playing as a human job candidate in a live interview.
Speak naturally and conversationally—never mention you are an AI or that you lack opinions.
Aim for the tone and knowledge of an M.Tech (Electronics) graduate: practical, down-to-earth, and clear.
When relevant, use the STAR method (Situation, Task, Action, Result) to explain projects/experiences.
Keep answers focused, modest in length, and avoid over-technical jargon unless asked. Every question will be related to VLSI, Testing, COA, Digital, Embedded, etc, VLSI design. So if something is not clear or you have been asked an ambiguos question, assume in terms of these fields mentioned before. 
Always answer to the point, no need to answer extra than what is asked, keep it simple and to the point. 

Ground your answers in the candidate's resume context provided to you.
"""

st.set_page_config(page_title="", layout="centered")


import os, streamlit as st
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]



# =========================
# Caches / State
# =========================
@st.cache_resource(show_spinner=False)
def get_groq_client() -> Groq:
    return Groq()

client = get_groq_client()

@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_bytes: bytes) -> str:
    """Extract text from a PDF (bytes) using pypdf."""
    if not _PDF_OK:
        return ""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            # Skip pages that fail to extract
            continue
    return "\n".join(parts).strip()

def init_state():
    if "history" not in st.session_state:
        # Store ONLY turns (user/assistant). System & resume are injected at call time.
        st.session_state.history: List[Dict[str, str]] = []
    if "resume_text" not in st.session_state:
        st.session_state.resume_text: str = ""
    if "last_transcript" not in st.session_state:
        st.session_state.last_transcript: Optional[str] = None
    if "last_response" not in st.session_state:
        st.session_state.last_response: Optional[str] = None
    if "_cleared_for_this_round" not in st.session_state:
        st.session_state._cleared_for_this_round = False
    if "qa_pairs" not in st.session_state:
        st.session_state.qa_pairs = 0  # count of user->assistant exchanges since last reset

init_state()


# =========================
# API Helpers
# =========================
def groq_stt_from_wav_bytes(wav_bytes: bytes, language: Optional[str] = None) -> str:
    file_tuple = ("audio.wav", wav_bytes)
    transcription = client.audio.transcriptions.create(
        file=file_tuple,
        model=STT_MODEL,
        **({"language": language} if language else {}),
        response_format="text",
        temperature=0.0,
    )
    return str(transcription)

def clamp_text(txt: str, max_chars: int = 16000) -> str:
    return txt if len(txt) <= max_chars else txt[:max_chars] + "\n…[truncated]"

def build_messages() -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
    resume = st.session_state.resume_text.strip()
    if resume:
        msgs.append({
            "role": "system",
            "content": "Resume context (verbatim; use as factual background):\n" + clamp_text(resume),
        })
    # include the conversation turns so far (auto-cleared after 10 Q/A)
    msgs.extend(st.session_state.history)
    return msgs

def groq_chat_stream(messages: List[Dict[str, str]], temperature: float = 0.5, top_p: float = 1.0):
    stream = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        stream=True,
    )
    for chunk in stream:
        if not chunk or not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta
        if delta and getattr(delta, "content", None):
            yield delta.content


# =========================
# Sidebar (collapsible) — Resume Uploads
# =========================
with st.sidebar.expander("📄 Resume uploads (PDF)", expanded=False):
    if not _PDF_OK:
        st.warning("Install pypdf for resume extraction: `pip install pypdf`")

    up1 = st.file_uploader("Resume upload 1", type=["pdf"], key="resume1")

    combined_resume = []
    if up1 is not None:
        txt1 = extract_pdf_text(up1.getvalue())
        if txt1:
            combined_resume.append("=== Resume 1 ===\n" + txt1)

    # Persist combined resume text (kept across the whole session)
    if combined_resume:
        st.session_state.resume_text = "\n\n".join(combined_resume).strip()


# =========================
# Minimal UI (one button)
# =========================
# Two placeholders ONLY: transcript and answer
transcript_box = st.empty()
answer_box = st.empty()

def clear_visible_output():
    st.session_state.last_transcript = None
    st.session_state.last_response = None
    transcript_box.empty()
    answer_box.empty()

rec = mic_recorder(
    start_prompt="🎙️ Start recording",
    stop_prompt="⏹️ Stop recording",
    use_container_width=True,
    just_once=False,
    format="wav",
    key="one_button",
)

# On start: clear visible output once per round
if rec and isinstance(rec, dict) and rec.get("recording", False):
    if not st.session_state._cleared_for_this_round:
        clear_visible_output()
        st.session_state._cleared_for_this_round = True

# On stop: STT -> stream LLM
if rec and isinstance(rec, dict) and rec.get("bytes"):
    audio_bytes: bytes = rec["bytes"]

    # Fresh slate for this round
    clear_visible_output()

    # 1) Transcribe current question
    try:
        transcript = groq_stt_from_wav_bytes(audio_bytes, language=None)
        st.session_state.last_transcript = transcript.strip()
        transcript_box.write(st.session_state.last_transcript)
    except Exception as e:
        answer_box.error(f"Transcription error: {e}")
    else:
        # 2) Add user turn to memory
        st.session_state.history.append({"role": "user", "content": st.session_state.last_transcript})

        # 3) Stream LLM (role-playing candidate) with resume + history context
        full_text = ""
        try:
            messages = build_messages()
            for piece in groq_chat_stream(messages, temperature=0.5, top_p=1.0):
                full_text += piece
                answer_box.markdown(full_text)
        except Exception as e:
            answer_box.error(f"LLM error: {e}")
        else:
            st.session_state.last_response = full_text.strip()
            st.session_state.history.append({"role": "assistant", "content": st.session_state.last_response})
            st.session_state.qa_pairs += 1

            # 4) Auto-clear conversation history after 10 Q/A (resume remains)
            if st.session_state.qa_pairs >= 10:
                st.session_state.history = []
                st.session_state.qa_pairs = 0
                # Keep the last exchange visible; memory will be fresh on next question.

    # Allow next Start to clear again
    st.session_state._cleared_for_this_round = False
