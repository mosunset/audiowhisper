#!/usr/bin/env python
"""
audiowhisper - OpenAI Whisperを使用した高精度音声文字起こしツール
"""

from .audio_utils import find_silence_segments, get_audio_duration
from .transcription import transcribe_file
from .utils import secondtotime

__version__ = "0.1.0"
__author__ = "audiowhisper"
__description__ = "OpenAI Whisperを使用した高精度音声文字起こしツール"

__all__ = [
    "transcribe_file",
    "get_audio_duration",
    "find_silence_segments",
    "secondtotime",
]
