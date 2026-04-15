"""
A1 Voice Sales Bot — real-time voice conversation.

Flow:
  Microphone → Azure STT → BotEngine (FSM + LLM) → Azure TTS → Speaker
  Supports barge-in: user can interrupt the bot while it speaks.
"""

import logging
import os
import threading
import time
from queue import Empty, Queue

import azure.cognitiveservices.speech as speechsdk
from dotenv import load_dotenv

from bot_engine import BotEngine
from conversation_fsm import State

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    val = os.getenv(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


def _first_env(*names: str) -> str:
    for n in names:
        val = os.getenv(n)
        if val:
            return val
    raise RuntimeError(f"None of these env vars set: {names}")


# ---------------------------------------------------------------------------
# Azure Speech setup
# ---------------------------------------------------------------------------

speech_key = _require_env("AZURE_SPEECH_KEY")
speech_region = _require_env("AZURE_SPEECH_REGION")
recognition_language = os.getenv("AZURE_SPEECH_RECOGNITION_LANGUAGE", "en-US")
voice_name = _first_env("AZURE_SPEECH_VOICE_NAME", "AZURE_SPEECH_VOICE")

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
speech_config.speech_recognition_language = recognition_language
speech_config.speech_synthesis_voice_name = voice_name

audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
speaker_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)

speech_recognizer = speechsdk.SpeechRecognizer(
    speech_config=speech_config,
    audio_config=audio_config,
)
speech_synthesizer = speechsdk.SpeechSynthesizer(
    speech_config=speech_config,
    audio_config=speaker_config,
)

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

recognized_queue: Queue = Queue()
shutdown_event = threading.Event()
speaking_lock = threading.Lock()
state_lock = threading.Lock()

current_request_id = 0
active_request_id = 0
speech_started_at: float | None = None
bot_is_speaking = False       # True while TTS is playing — mutes STT to prevent echo
bot_stopped_speaking_at: float = 0.0  # monotonic time when TTS last finished
POST_SPEECH_MUTE_SEC = 1.5    # ignore STT for this long after bot finishes speaking

barge_in_seconds = float(os.getenv("REALTIME_BARGE_IN_SECONDS", "2.0"))

bot = BotEngine()

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)


def _save_session(summary_json: str) -> None:
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(SESSIONS_DIR, f"session_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_json)
    logger.info("Session saved to %s", path)

# ---------------------------------------------------------------------------
# TTS
# ---------------------------------------------------------------------------

def speak_text(text: str) -> None:
    global bot_is_speaking, bot_stopped_speaking_at
    if not text:
        return
    with speaking_lock:
        bot_is_speaking = True
        logger.info("Bot: %s", text)
        # Stop STT while bot speaks to prevent echo
        speech_recognizer.stop_continuous_recognition_async().get()
        try:
            result = speech_synthesizer.speak_text_async(text).get()
            if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
                logger.error("TTS error: reason=%s", result.reason)
        finally:
            bot_is_speaking = False
            bot_stopped_speaking_at = time.monotonic()
            # Resume STT after bot finishes
            speech_recognizer.start_continuous_recognition_async().get()


def interrupt_speech() -> None:
    try:
        speech_synthesizer.stop_speaking_async().get()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Worker thread: processes recognized speech → LLM → TTS
# ---------------------------------------------------------------------------

def _clear_queue() -> None:
    while True:
        try:
            item = recognized_queue.get_nowait()
        except Empty:
            break
        if item == "__STOP__":
            recognized_queue.task_done()
            recognized_queue.put("__STOP__")
            break
        recognized_queue.task_done()


def _queue_user_text(text: str) -> int:
    global current_request_id, active_request_id
    with state_lock:
        current_request_id += 1
        rid = current_request_id
        active_request_id = rid
    _clear_queue()
    interrupt_speech()
    recognized_queue.put((rid, text))
    return rid


def worker() -> None:
    while not shutdown_event.is_set():
        item = recognized_queue.get()
        if item == "__STOP__":
            recognized_queue.task_done()
            break

        rid, user_text = item
        logger.info("User said: %s", user_text)

        try:
            reply = bot.handle_turn(user_text)

            with state_lock:
                if rid != active_request_id:
                    logger.info("Skipping stale reply (interrupted).")
                    continue

            if reply:
                speak_text(reply)

            # If FSM reached a terminal state, initiate shutdown
            if bot.fsm.is_terminal():
                logger.info("Conversation complete. Session data:\n%s", bot.fsm.session_summary())
                shutdown_event.set()

        except Exception as exc:
            logger.error("Worker error: %s", exc)
            speak_text("I'm sorry, something went wrong. Could you repeat that?")
        finally:
            recognized_queue.task_done()


# ---------------------------------------------------------------------------
# STT callbacks
# ---------------------------------------------------------------------------

def on_recognized(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
    global speech_started_at
    if bot_is_speaking:
        return  # bot's own voice echoing back
    if time.monotonic() - bot_stopped_speaking_at < POST_SPEECH_MUTE_SEC:
        return  # STT result arrived just after bot finished — still echo
    if evt.result.reason == speechsdk.ResultReason.RecognizedSpeech:
        text = evt.result.text.strip()
        if text:
            speech_started_at = None
            _queue_user_text(text)
    elif evt.result.reason == speechsdk.ResultReason.NoMatch:
        speech_started_at = None
        logger.debug("STT: no match")


def on_recognizing(evt: speechsdk.SpeechRecognitionEventArgs) -> None:
    global speech_started_at
    if evt.result.reason != speechsdk.ResultReason.RecognizingSpeech:
        return
    partial = evt.result.text.strip()
    if not partial:
        return
    # If bot is speaking and user starts talking — that's a barge-in, allow it
    if bot_is_speaking:
        now = time.monotonic()
        if speech_started_at is None:
            speech_started_at = now
            return
        if now - speech_started_at >= barge_in_seconds:
            interrupt_speech()
        return
    now = time.monotonic()
    if speech_started_at is None:
        speech_started_at = now
        return
    if now - speech_started_at >= barge_in_seconds:
        interrupt_speech()


def on_canceled(evt: speechsdk.SpeechRecognitionCanceledEventArgs) -> None:
    logger.warning("STT canceled: %s", evt.reason)
    if evt.reason == speechsdk.CancellationReason.Error:
        logger.error("STT error: %s", evt.error_details)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  A1 Voice Sales Bot")
    print("  Speak into your microphone. Press Ctrl+C to stop.")
    print("=" * 60)

    # Start worker thread
    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    # Wire up STT callbacks
    speech_recognizer.recognizing.connect(on_recognizing)
    speech_recognizer.recognized.connect(on_recognized)
    speech_recognizer.canceled.connect(on_canceled)
    speech_recognizer.start_continuous_recognition_async().get()

    # Bot opens the conversation (no user input yet)
    opening = bot.handle_turn(None)
    speak_text(opening)

    try:
        while not shutdown_event.is_set():
            shutdown_event.wait(0.5)
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        shutdown_event.set()
        speech_recognizer.stop_continuous_recognition_async().get()
        recognized_queue.put("__STOP__")
        worker_thread.join(timeout=5)

        if bot.fsm.state != State.GREETING:
            print("\n--- Session Summary ---")
            print(bot.fsm.session_summary())
            _save_session(bot.fsm.session_summary())


if __name__ == "__main__":
    main()
