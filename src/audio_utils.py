#!/usr/bin/env python
"""
音声処理関連のユーティリティ関数
"""

import librosa
import numpy as np


def get_audio_duration(filename):
    """
    音声ファイルの長さを取得する関数
    """
    try:
        duration = librosa.get_duration(path=filename)
        return duration
    except Exception as e:
        print(f"音声ファイルの長さ取得に失敗しました: {e}")
        return None


def find_silence_segments(
    audio_path, segment_duration=1800, silence_threshold=-40, min_silence_duration=2.0
):
    """
    音声ファイルを読み込んで、指定された長さのセグメントに分割するための
    無音区間の位置を検出する関数

    Args:
        audio_path: 音声ファイルのパス
        segment_duration: 目標セグメント長（秒、デフォルト30分=1800秒）
        silence_threshold: 無音判定の閾値（dB）
        min_silence_duration: 最小無音区間長（秒）

    Returns:
        segments: [(start_time, end_time), ...] のリスト
    """
    print(f"音声ファイルを分析中: {audio_path}")

    # 音声ファイルを読み込み
    y, sr = librosa.load(audio_path, sr=None)

    # 音声の長さを取得
    total_duration = len(y) / sr
    print(f"総再生時間: {total_duration:.1f}秒 ({total_duration / 60:.1f}分)")

    # 音声レベルを計算
    hop_length = 512
    frame_length = 2048

    # RMS（二乗平均平方根）を計算して音量レベルを取得
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # dBに変換
    db = 20 * np.log10(rms + 1e-10)

    # 無音判定
    silence_mask = db < silence_threshold

    # 無音区間の開始・終了位置を検出
    silence_starts = []
    silence_ends = []

    for i in range(1, len(silence_mask)):
        if silence_mask[i] and not silence_mask[i - 1]:  # 無音開始
            silence_starts.append(i)
        elif not silence_mask[i] and silence_mask[i - 1]:  # 無音終了
            silence_ends.append(i)

    # 最初と最後の処理
    if silence_mask[0]:
        silence_starts.insert(0, 0)
    if silence_mask[-1]:
        silence_ends.append(len(silence_mask))

    # 無音区間の長さを計算
    silence_segments = []
    for start, end in zip(silence_starts, silence_ends):
        start_time = start * hop_length / sr
        end_time = end * hop_length / sr
        duration = end_time - start_time

        if duration >= min_silence_duration:
            silence_segments.append((start_time, end_time))

    print(f"検出された無音区間数: {len(silence_segments)}")

    # セグメント分割点を決定
    segments = []
    current_start = 0.0

    for silence_start, silence_end in silence_segments:
        silence_center = (silence_start + silence_end) / 2

        # 現在位置から目標セグメント長を超えた場合、最も近い無音区間で分割
        if silence_center - current_start >= segment_duration:
            segments.append((current_start, silence_center))
            current_start = silence_center

    # 最後のセグメントを追加
    if current_start < total_duration:
        segments.append((current_start, total_duration))

    # セグメントが長すぎる場合は強制的に分割
    final_segments = []
    for start, end in segments:
        if end - start > segment_duration * 1.5:  # 45分を超える場合
            # 強制的に30分ずつに分割
            current = start
            while current < end:
                segment_end = min(current + segment_duration, end)
                final_segments.append((current, segment_end))
                current = segment_end
        else:
            final_segments.append((start, end))

    print(f"分割セグメント数: {len(final_segments)}")
    for i, (start, end) in enumerate(final_segments):
        print(
            f"  セグメント {i + 1}: {start:.1f}s - {end:.1f}s ({(end - start) / 60:.1f}分)"
        )

    return final_segments
