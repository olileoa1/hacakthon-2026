"""
Speech Services Utilities

This module provides utility functions for audio transcription and synthesis
using Azure Speech Services.
"""

import os
import tempfile
import time
from pathlib import Path
import azure.cognitiveservices.speech as speechsdk


def require_env(name: str) -> str:
    """Get required environment variable"""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def first_env(*names: str) -> str:
    """Get first available environment variable from a list"""
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    raise RuntimeError(f"Missing required environment variable. Tried: {', '.join(names)}")


def get_speech_config() -> speechsdk.SpeechConfig:
    """Initialize and return Azure Speech Config"""
    speech_key = require_env("AZURE_SPEECH_KEY")
    speech_region = require_env("AZURE_SPEECH_REGION")
    speech_recognition_language = os.getenv("AZURE_SPEECH_RECOGNITION_LANGUAGE", "en-US")
    speech_voice_name = first_env("AZURE_SPEECH_VOICE_NAME", "AZURE_SPEECH_VOICE")
    
    config = speechsdk.SpeechConfig(subscription=speech_key, region=speech_region)
    config.speech_recognition_language = speech_recognition_language
    config.speech_synthesis_voice_name = speech_voice_name
    return config


def transcribe_audio(audio_bytes: bytes) -> str:
    """
    Transcribe audio bytes using Azure Speech Services.
    
    Args:
        audio_bytes: The audio data in bytes
        
    Returns:
        Transcribed text or error message
    """
    tmp_path = None
    try:
        speech_config = get_speech_config()
        
        # Write audio bytes to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_bytes)
            tmp_path = tmp_file.name
        
        # Create audio config from file
        audio_config = speechsdk.audio.AudioConfig(filename=tmp_path)
        speech_recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )
        
        # Perform recognition
        result = speech_recognizer.recognize_once()
        
        if result.reason == speechsdk.ResultReason.RecognizedSpeech:
            return result.text.strip()
        elif result.reason == speechsdk.ResultReason.NoMatch:
            return "No speech could be recognized."
        elif result.reason == speechsdk.ResultReason.Canceled:
            details = speechsdk.SpeechRecognitionCancellationDetails.from_result(result)
            return f"Error: {details.error_details}"
    except Exception as e:
        return f"Error transcribing audio: {str(e)}"
    finally:
        # Clean up temporary file - with retry logic
        if tmp_path:
            try:
                time.sleep(0.1)  # Give the file a moment to be released
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass  # Silently ignore cleanup errors


def synthesize_speech(text: str) -> bytes:
    """
    Convert text to speech using Azure Speech Services.
    
    Args:
        text: The text to synthesize
        
    Returns:
        Audio data in bytes
        
    Raises:
        Exception: If speech synthesis fails
    """
    tmp_path = None
    try:
        speech_config = get_speech_config()
        
        # Create a temporary file for audio output
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_path = tmp_file.name
        
        # Create audio config to write to file
        audio_config = speechsdk.audio.AudioOutputConfig(filename=tmp_path)
        speech_synthesizer = speechsdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=audio_config,
        )
        
        # Perform synthesis
        result = speech_synthesizer.speak_text_async(text).get()
        
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Read the generated audio file
            time.sleep(0.1)  # Give the file a moment to be fully written
            
            with open(tmp_path, "rb") as audio_file:
                audio_data = audio_file.read()
            return audio_data
        else:
            details = speechsdk.SpeechSynthesisCancellationDetails.from_result(result)
            raise Exception(f"Speech synthesis failed: {details.error_details}")
    except Exception as e:
        raise Exception(f"Error synthesizing speech: {str(e)}")
    finally:
        # Clean up temporary file - with retry logic
        if tmp_path:
            try:
                time.sleep(0.1)  # Give the file a moment to be released
                Path(tmp_path).unlink(missing_ok=True)
            except Exception:
                pass  # Silently ignore cleanup errors
