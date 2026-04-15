"""
Web UI server for A1 Voice Sales Bot.

Runs the voice bot (Azure STT/TTS) in a background thread and streams
transcript + log events to the browser via WebSocket.

Usage:
    cd Team0/starter_kit
    python web_server.py
Then open http://localhost:8000
"""

import asyncio
import json
import logging
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import azure.cognitiveservices.speech as speechsdk
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

# ---------------------------------------------------------------------------
# In-memory event bus: voice thread → asyncio → WebSocket clients
# ---------------------------------------------------------------------------

_event_queue: queue.Queue = queue.Queue()   # thread-safe
_log_records: list[dict] = []               # all captured log entries
_transcript: list[dict] = []               # conversation history

MAX_LOGS = 500


def _push_event(event: dict) -> None:
    """Called from any thread to push an event to WebSocket clients."""
    event.setdefault("ts", datetime.now().isoformat(timespec="milliseconds"))
    _event_queue.put(event)


# ---------------------------------------------------------------------------
# Custom log handler — captures WARNING+ to the log viewer
# ---------------------------------------------------------------------------

class _WebLogHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        entry = {
            "ts": datetime.fromtimestamp(record.created).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": self.format(record),
        }
        _log_records.append(entry)
        if len(_log_records) > MAX_LOGS:
            _log_records.pop(0)
        if record.levelno >= logging.INFO:
            _push_event({"type": "log", **entry})


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
_web_handler = _WebLogHandler()
_web_handler.setLevel(logging.INFO)
_web_handler.setFormatter(logging.Formatter("%(message)s"))
logging.getLogger().addHandler(_web_handler)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Voice bot state
# ---------------------------------------------------------------------------

_bot_thread: Optional[threading.Thread] = None
_shutdown_event: Optional[threading.Event] = None
_bot_active = False


def _add_transcript(role: str, text: str) -> None:
    entry = {"role": role, "text": text, "ts": datetime.now().isoformat(timespec="milliseconds")}
    _transcript.append(entry)
    _push_event({"type": "transcript", **entry})


# ---------------------------------------------------------------------------
# Bot runner (runs in its own thread)
# ---------------------------------------------------------------------------

def _run_bot(shutdown_event: threading.Event) -> None:
    global _bot_active

    # Lazy imports so the module can load without Azure creds present
    from bot_engine import BotEngine
    from conversation_fsm import State

    def _require_env(name):
        val = os.getenv(name)
        if not val:
            raise RuntimeError(f"Missing required env var: {name}")
        return val

    def _first_env(*names):
        for n in names:
            val = os.getenv(n)
            if val:
                return val
        raise RuntimeError(f"None of these env vars set: {names}")

    try:
        speech_key = _require_env("AZURE_SPEECH_KEY")
        speech_region = _require_env("AZURE_SPEECH_REGION")
        recognition_language = os.getenv("AZURE_SPEECH_RECOGNITION_LANGUAGE", "en-US")
        voice_name = _first_env("AZURE_SPEECH_VOICE_NAME", "AZURE_SPEECH_VOICE")
    except RuntimeError as exc:
        logger.error("Configuration error: %s", exc)
        _push_event({"type": "status", "status": "error", "msg": str(exc)})
        _bot_active = False
        return

    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    speech_config.speech_recognition_language = recognition_language
    speech_config.speech_synthesis_voice_name = voice_name

    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    speaker_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=speaker_config)

    recognized_queue: queue.Queue = queue.Queue()
    speaking_lock = threading.Lock()
    state_lock = threading.Lock()

    current_request_id = 0
    active_request_id = 0
    speech_started_at: Optional[float] = None
    bot_is_speaking = False
    bot_stopped_speaking_at: float = 0.0
    POST_SPEECH_MUTE_SEC = 0.5
    barge_in_seconds = float(os.getenv("REALTIME_BARGE_IN_SECONDS", "2.0"))

    bot = BotEngine()
    _push_event({"type": "status", "status": "running"})

    import re

    def _split_sentences(text: str) -> list[str]:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [p.strip() for p in parts if p.strip()]

    def speak_text(text: str) -> None:
        nonlocal bot_is_speaking, bot_stopped_speaking_at, speech_started_at
        if not text:
            return
        with speaking_lock:
            bot_is_speaking = True
            speech_started_at = None
            logger.info("Bot: %s", text)
            _add_transcript("bot", text)
            try:
                sentences = _split_sentences(text)
                for sentence in sentences:
                    if shutdown_event.is_set():
                        break
                    if not sentence:
                        continue
                    result = speech_synthesizer.speak_text_async(sentence).get()
                    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                        logger.error("TTS error: reason=%s", result.reason)
                        break
            finally:
                bot_is_speaking = False
                bot_stopped_speaking_at = time.monotonic()

    def interrupt_speech() -> None:
        try:
            speech_synthesizer.stop_speaking_async().get()
        except Exception:
            pass

    def _clear_queue() -> None:
        while True:
            try:
                item = recognized_queue.get_nowait()
            except queue.Empty:
                break
            if item == "__STOP__":
                recognized_queue.task_done()
                recognized_queue.put("__STOP__")
                break
            recognized_queue.task_done()

    def _queue_user_text(text: str) -> int:
        nonlocal current_request_id, active_request_id
        with state_lock:
            current_request_id += 1
            rid = current_request_id
            active_request_id = rid
        _clear_queue()
        interrupt_speech()
        recognized_queue.put((rid, text))
        return rid

    def worker() -> None:
        nonlocal active_request_id
        while not shutdown_event.is_set():
            try:
                item = recognized_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if item == "__STOP__":
                recognized_queue.task_done()
                break

            rid, user_text = item
            logger.info("User said: %s", user_text)
            _add_transcript("user", user_text)

            try:
                reply = bot.handle_turn(user_text)
                with state_lock:
                    if rid != active_request_id:
                        logger.info("Skipping stale reply (interrupted).")
                        continue
                if reply:
                    speak_text(reply)
                if bot.fsm.is_terminal():
                    logger.info("Conversation complete.")
                    _push_event({"type": "status", "status": "done", "summary": json.loads(bot.fsm.session_summary())})
                    shutdown_event.set()
            except Exception as exc:
                logger.error("Worker error: %s", exc)
                speak_text("I'm sorry, something went wrong. Could you repeat that?")
            finally:
                recognized_queue.task_done()

    def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        nonlocal speech_started_at
        if bot_is_speaking:
            logger.info("STT: ignored (bot speaking): %r", evt.result.text)
            return
        mute_remaining = POST_SPEECH_MUTE_SEC - (time.monotonic() - bot_stopped_speaking_at)
        if mute_remaining > 0:
            logger.info("STT: ignored (post-speech mute %.1fs left): %r", mute_remaining, evt.result.text)
            return
        if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            text = evt.result.text.strip()
            logger.info("STT recognized: %r", text)
            if text and len(text) >= 2:
                speech_started_at = None
                _queue_user_text(text)
            else:
                logger.info("STT: too short, ignored")
        elif evt.result.reason == speechsdk.ResultReason.NoMatch:
            speech_started_at = None
            logger.info("STT: no match")

    def on_recognizing(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
        nonlocal speech_started_at
        if evt.result.reason != speechsdk.ResultReason.RecognizingSpeech:
            return
        partial = evt.result.text.strip()
        if not partial or bot_is_speaking:
            return
        logger.info("STT partial: %r", partial)
        now = time.monotonic()
        if speech_started_at is None:
            speech_started_at = now
            return
        if now - speech_started_at >= barge_in_seconds:
            interrupt_speech()

    def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
        logger.warning("STT canceled: %s", evt.reason)
        if evt.reason == speechsdk.CancellationReason.Error:
            logger.error("STT error: %s — check AZURE_SPEECH_KEY and AZURE_SPEECH_REGION", evt.error_details)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    speech_recognizer.recognizing.connect(on_recognizing)
    speech_recognizer.recognized.connect(on_recognized)
    speech_recognizer.canceled.connect(on_canceled)
    speech_recognizer.start_continuous_recognition_async().get()
    logger.info("STT started — listening for speech (language: %s)", recognition_language)

    opening = bot.handle_turn(None)
    speak_text(opening)

    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(0.5)
    finally:
        _bot_active = False
        speech_recognizer.stop_continuous_recognition_async().get()
        recognized_queue.put("__STOP__")
        worker_thread.join(timeout=5)

        if bot.fsm.state != State.GREETING:
            summary = bot.fsm.session_summary()
            SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "sessions")
            os.makedirs(SESSIONS_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(SESSIONS_DIR, f"session_{ts}.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write(summary)
            logger.info("Session saved to %s", path)

        _push_event({"type": "status", "status": "stopped"})


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="A1 Voice Bot Web UI")

# WebSocket connection manager
_ws_clients: list[WebSocket] = []
_ws_lock = asyncio.Lock()


async def _broadcast(event: dict) -> None:
    dead = []
    async with _ws_lock:
        for ws in _ws_clients:
            try:
                await ws.send_text(json.dumps(event))
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.remove(ws)


# Background task: drain _event_queue and broadcast to all WS clients
async def _event_pump() -> None:
    while True:
        try:
            event = _event_queue.get_nowait()
            await _broadcast(event)
        except queue.Empty:
            await asyncio.sleep(0.05)


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_event_pump())


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(html_path, encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/start")
async def start_bot() -> dict:
    global _bot_thread, _shutdown_event, _bot_active, _transcript
    if _bot_active:
        return {"ok": False, "msg": "Bot is already running"}
    _transcript.clear()
    _shutdown_event = threading.Event()
    _bot_active = True
    _bot_thread = threading.Thread(target=_run_bot, args=(_shutdown_event,), daemon=True)
    _bot_thread.start()
    return {"ok": True}


@app.post("/stop")
async def stop_bot() -> dict:
    global _bot_active
    if _shutdown_event:
        _shutdown_event.set()
    _bot_active = False
    return {"ok": True}


@app.get("/transcript")
async def get_transcript() -> list:
    return _transcript


@app.get("/logs")
async def get_logs(level: str = "all") -> list:
    if level == "all":
        return _log_records[-200:]
    return [r for r in _log_records if r["level"] == level.upper()][-200:]


@app.get("/status")
async def get_status() -> dict:
    return {"active": _bot_active}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.append(websocket)
    # Send current state immediately on connect
    await websocket.send_text(json.dumps({"type": "status", "status": "running" if _bot_active else "stopped"}))
    for entry in _transcript:
        await websocket.send_text(json.dumps({"type": "transcript", **entry}))
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        async with _ws_lock:
            if websocket in _ws_clients:
                _ws_clients.remove(websocket)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")