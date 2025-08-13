# app.py
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import Response
import json, base64, requests
from pydub import AudioSegment
from io import BytesIO

# --- LangChain (Groq via OpenAI-compatible endpoint) ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# ====== TWILIO & GROQ / ELEVENLABS CREDENTIALS ======
TWILIO_ACCOUNT_SID = "AC51af7f2ac54fdef05f94afb0910a9e35"   # your SID
TWILIO_AUTH_TOKEN  = "cae30afee303a1dd604a268e8f431031"      # your Auth Token
TWILIO_PHONE_NUMBER = "+18079093284"                         # your number

# Groq LLM (OpenAI-compatible endpoint)
GROQ_API_KEY = "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"
GROQ_MODEL   = "meta-llama/llama-4-scout-17b-16e-instruct"

# ElevenLabs (STT + TTS)
ELEVEN_API_KEY = "sk_b1199e430751500e0318db7361a13be817d07c6c057f0d8b"
ELEVEN_VOICE_ID = "kdnRe2koJdOK4Ovxn2DI"   # e.g., from your ElevenLabs dashboard
# =====================================================

app = FastAPI()

# Groq LLM client
llm = ChatOpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=GROQ_API_KEY,
    model=GROQ_MODEL,
    temperature=0.0,
)

SYSTEM_PROMPT = (
    "You are an order-taking AI agent for a restaurant.\n"
    "Greet the caller briefly, ask for their order, extract items with quantity and variants, "
    "confirm after each item, then summarize and ask for final confirmation. Be concise."
)

# ---------- ElevenLabs helpers ----------

def eleven_stt_wav(audio_wav_bytes: bytes) -> str:
    """
    Send a WAV chunk to ElevenLabs STT and return transcribed text.
    """
    url = "https://api.elevenlabs.io/v1/speech-to-text"
    headers = {"xi-api-key": ELEVEN_API_KEY}
    files = {"file": ("audio.wav", audio_wav_bytes, "audio/wav")}
    data = {"model_id": "eleven_multilingual_v2"}  # adjust if you use a different STT model
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    resp.raise_for_status()
    j = resp.json()
    return (j.get("text") or "").strip()

def eleven_tts_mp3(text: str) -> bytes:
    """
    Generate speech (MP3 bytes) from text using ElevenLabs TTS.
    """
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVEN_VOICE_ID}"
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVEN_API_KEY,
    }
    payload = {
        "text": text,
        "model_id": "eleven_multilingual_v2",  # or your preferred TTS model
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8},
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.content  # MP3 bytes
# ----------------------------------------

@app.post("/voice")
async def voice(request: Request):
    """
    Twilio hits this endpoint when a call comes in.
    We tell it to start a media stream to our /media WebSocket.
    """
    twiml = """
    <Response>
      <Connect>
        <Stream url="wss://inbound-agent-production.up.railway.app/media" />
      </Connect>
    </Response>
    """
    return Response(content=twiml, media_type="text/xml")

@app.websocket("/media")
async def media_ws(websocket: WebSocket):
    """
    WebSocket endpoint Twilio connects to for sending/receiving audio.
    """
    await websocket.accept()
    print("WebSocket connection established")

    while True:
        try:
            message = await websocket.receive_text()
            data = json.loads(message)

            if data.get("event") == "media":
                # ----- Decode incoming μ-law (8kHz mono) audio from Twilio -----
                mu_law = base64.b64decode(data["media"]["payload"])
                segment = AudioSegment(mu_law, sample_width=1, frame_rate=8000, channels=1)

                # Upconvert to 16k PCM WAV for STT (better accuracy)
                pcm16 = segment.set_frame_rate(16000).set_channels(1).set_sample_width(2)
                with BytesIO() as wav_io:
                    pcm16.export(wav_io, format="wav")
                    wav_io.seek(0)
                    user_text = eleven_stt_wav(wav_io.read())

                if not user_text:
                    continue
                print(f"Caller said: {user_text}")

                # ----- AI reply (Groq LLM) -----
                ai_msg = llm.invoke([
                    SystemMessage(content=SYSTEM_PROMPT),
                    HumanMessage(content=user_text),
                ])
                reply_text = (ai_msg.content or "").strip() or "Sorry, could you please repeat that?"
                print(f"AI reply: {reply_text}")

                # ----- Text-to-Speech (ElevenLabs) -----
                mp3_bytes = eleven_tts_mp3(reply_text)

                # Downsample to 8kHz, mono, 8-bit (μ-law-like) for Twilio payload
                # (Twilio Media Streams expects base64 audio frames; pydub will output WAV
                # that we pack into the payload. Works for many setups; adjust if needed.)
                reply_segment = AudioSegment.from_file(BytesIO(mp3_bytes), format="mp3") \
                                            .set_frame_rate(8000) \
                                            .set_channels(1) \
                                            .set_sample_width(1)
                with BytesIO() as out_io:
                    reply_segment.export(out_io, format="wav")
                    await websocket.send_text(json.dumps({
                        "event": "media",
                        "media": {
                            "payload": base64.b64encode(out_io.getvalue()).decode("utf-8")
                        }
                    }))

            elif data.get("event") in ("start", "stop", "mark"):
                # Optional: handle lifecycle events
                pass

        except Exception as e:
            print("WebSocket error:", e)
            break

# ---- your whole existing code above ----

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )


