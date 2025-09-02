
import os
import io
from typing import List, Dict, Optional

import streamlit as st
from streamlit_mic_recorder import mic_recorder
from groq import Groq
import streamlit.components.v1 as components

_PDF_OK = True
try:
    from pypdf import PdfReader
except Exception:
    _PDF_OK = False

STT_MODEL = "whisper-large-v3-turbo"
CHAT_MODEL = "llama-3.3-70b-versatile"

SYSTEM_PROMPT = """
You are role-playing as a human job candidate in a live interview.
Speak naturally and conversationally—never mention you are an AI or that you lack opinions.
Aim for the tone and knowledge of an M.Tech (Electronics) graduate: practical, down-to-earth, and clear.
When relevant, use the STAR method (Situation, Task, Action, Result) to explain projects/experiences.
Remember to always answer in very normal indian english, no need for hifi english words.
Sometimes the user may ask ambiguos questions, you need to make some assumptionsabout it, for example, user may ask which microcontroller was used in a project, so you need to take proper context from the resume, and answer it accordingly.
Sometimes, the user may ask questions such that it may feel like fill in the blanks, for eg, user may ask-- so the gassheild project is.., then you will have to complete the answer further. 
Keep answers focused, modest in length, and avoid over-technical jargon unless asked. Every question will be related to VLSI, Testing, COA, Digital, Embedded, etc, VLSI design. So if something is not clear or you have been asked an ambiguos question, assume in terms of these fields mentioned before. 
Always answer to the point, no need to answer extra than what is asked, keep it simple and to the point. 

Try to answer somewhat in depth, understand what the user is trying to ask, take proper context and answer. 

Ground your answers in the candidate's resume context provided to you.

One more thing to remember-
You are also provided a document along with resume which has many important points to remember while answering question, imformation like which chip, fpga, or microcontroller is used in which project is mentioned there. Please refer to that along with the resume to answer your questions.
Important Point : if you are writing code in response, write comments in the code that clearly explain what that line of code does, please remember this, this is a very important point. 
"""

st.set_page_config(page_title="", layout="centered")

import os
import streamlit as st
from pathlib import Path

def _load_groq_key():
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    local_paths = [
        Path.home() / ".streamlit/secrets.toml",
        Path.cwd() / ".streamlit/secrets.toml",
    ]
    if any(p.exists() for p in local_paths):
        try:
            return st.secrets.get("GROQ_API_KEY", None)
        except Exception:
            return None
    try:
        return st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        return None

GROQ_API_KEY = _load_groq_key()

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
else:
    st.warning("GROQ_API_KEY not found in environment or Streamlit secrets.")

@st.cache_resource(show_spinner=False)
def get_groq_client() -> Groq:
    return Groq()

client = get_groq_client()

@st.cache_data(show_spinner=False)
def extract_pdf_text(pdf_bytes: bytes) -> str:
    if not _PDF_OK:
        return ""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts = []
    for page in reader.pages:
        try:
            parts.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(parts).strip()

def init_state():
    if "history" not in st.session_state:
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
        st.session_state.qa_pairs = 0

init_state()

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

with st.sidebar.expander("📄 Resume uploads (PDF)", expanded=False):
    if not _PDF_OK:
        st.warning("Install pypdf for resume extraction: `pip install pypdf`")
    up1 = st.file_uploader("Resume upload 1", type=["pdf"], key="resume1")
    up2 = st.file_uploader("Important Points To Remember", type=["pdf"], key="resume2")

    combined_resume = []
    if up1 is not None:
        txt1 = extract_pdf_text(up1.getvalue())
        if txt1:
            combined_resume.append("=== Resume 1 ===\n" + txt1)
    if up2 is not None:
        txt2 = extract_pdf_text(up2.getvalue())
        if txt2:
            combined_resume.append("=== Important Points to Remember ===\n" + txt2)
    if combined_resume:
        st.session_state.resume_text = "\n\n".join(combined_resume).strip()

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

components.html(
    """
    <script>
      (function() {
        if (window.__spaceHandlerAttached) return;
        window.__spaceHandlerAttached = true;

        const makeFocusable = () => {
          try {
            const i = document.createElement('input');
            i.type = 'text';
            i.autofocus = true;
            i.style.position = 'fixed';
            i.style.opacity = '0';
            i.style.height = '0';
            i.style.width = '0';
            i.style.pointerEvents = 'none';
            document.body.appendChild(i);
            setTimeout(() => { try { i.blur(); document.body.focus(); } catch(e) {} }, 50);
          } catch(e) {}
        };

        const rootDoc = (function() {
          try { return window.parent && window.parent.document ? window.parent.document : document; } catch(e) { return document; }
        })();

        const isTyping = (el) => {
          if (!el) return false;
          const tag = el.tagName ? el.tagName.toLowerCase() : "";
          const editable = el.isContentEditable;
          return editable || tag === "input" || tag === "textarea" || tag === "select";
        };

        const findRecorderButton = () => {
          const labels = ["🎙️ Start recording", "⏹️ Stop recording"];
          const btns = Array.from(rootDoc.querySelectorAll('button'));
          for (const b of btns) {
            const txt = (b.innerText || "").trim();
            if (labels.some(l => txt.includes(l))) return b;
          }
          return null;
        };

        const handler = function(e) {
          const key = e.code === "Space" || e.key === " " || e.key === "Spacebar";
          const typing = isTyping(rootDoc.activeElement) || isTyping(document.activeElement);
          if (key && !typing) {
            const btn = findRecorderButton();
            if (btn) {
              e.preventDefault();
              e.stopPropagation();
              btn.click();
            }
          }
        };

        makeFocusable();
        try { rootDoc.addEventListener("keydown", handler, true); } catch(e) {}
        window.addEventListener("keydown", handler, true);
      })();
    </script>
    """,
    height=0
)

if rec and isinstance(rec, dict) and rec.get("recording", False):
    if not st.session_state._cleared_for_this_round:
        clear_visible_output()
        st.session_state._cleared_for_this_round = True

if rec and isinstance(rec, dict) and rec.get("bytes"):
    audio_bytes: bytes = rec["bytes"]
    clear_visible_output()
    try:
        transcript = groq_stt_from_wav_bytes(audio_bytes, language=None)
        st.session_state.last_transcript = transcript.strip()
        transcript_box.write(st.session_state.last_transcript)
    except Exception as e:
        answer_box.error(f"Transcription error: {e}")
    else:
        st.session_state.history.append({"role": "user", "content": st.session_state.last_transcript})
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
            if st.session_state.qa_pairs >= 10:
                st.session_state.history = []
                st.session_state.qa_pairs = 0
    st.session_state._cleared_for_this_round = False
