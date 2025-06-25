#!/usr/bin/env python
"""
audiowhisper - メインエントリーポイント
"""

import argparse
import os

from src.transcription import transcribe_file


def main():
    parser = argparse.ArgumentParser(
        description="Whisperによる文字起こしを行うツール",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="入力する音声ファイルのパス。複数指定可。",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="文字起こし時のビームサイズ",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="turbo",
        help="Whisperモデルのサイズ (tiny, base, small, medium, large, turbo など)",
    )
    parser.add_argument(
        "--output_mask",
        type=int,
        default=7,
        help="出力ファイルのマスク。1=<filename>.time.txt, 2=<filename>.srt, 4=<filename>.txt の組合せ",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="処理終了後にシャットダウンするかどうか",
    )
    parser.add_argument(
        "--high_quality",
        action="store_true",
        help="より高精度なモードを使用するかどうか",
    )
    parser.add_argument(
        "--no_auto_segment",
        action="store_true",
        help="長時間音声ファイルの自動セグメント分割を無効にする",
    )
    parser.add_argument(
        "--segment_duration",
        type=int,
        default=1800,
        help="セグメント分割時の目標長（秒、デフォルト30分=1800秒）",
    )
    args = parser.parse_args()

    for file in args.input:
        transcribe_file(
            file,
            beam_size=args.beam_size,
            model_name=args.model,
            output_mask=args.output_mask,
            high_quality=args.high_quality,
            auto_segment=not args.no_auto_segment,
            segment_duration=args.segment_duration,
        )

    if args.shutdown:
        print("10秒後にシャットダウンします")
        os.system("timeout 10")
        os.system("shutdown -s -t 0")


if __name__ == "__main__":
    main()
