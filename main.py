#!/usr/bin/env python
"""
audiowhisper - メインエントリーポイント
"""

import argparse
import os

from src.file_manager import (
    create_output_folders,
    move_transcription_files,
    process_input_folder,
)
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
        help="入力する音声ファイルのパス。複数指定可。",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="inputフォルダ内のファイルを連続処理する",
    )
    parser.add_argument(
        "--input_dir",
        default="input",
        help="inputフォルダのパス（--batchオプション使用時）",
    )
    parser.add_argument(
        "--output_dir",
        default="output",
        help="出力先フォルダ（--batchオプション使用時）",
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
    parser.add_argument(
        "--setup",
        action="store_true",
        help="必要なフォルダ（input, output）を作成する",
    )
    parser.add_argument(
        "--move_transcriptions",
        action="store_true",
        help="既存の文字起こしファイルをinputフォルダからoutputフォルダに移動する",
    )
    args = parser.parse_args()

    # セットアップモード
    if args.setup:
        create_output_folders()
        return

    # 文字起こしファイル移動モード
    if args.move_transcriptions:
        move_transcription_files(input_dir=args.input_dir, output_dir=args.output_dir)
        return

    # バッチ処理モード
    if args.batch:
        transcribe_kwargs = {
            "beam_size": args.beam_size,
            "model_name": args.model,
            "output_mask": args.output_mask,
            "high_quality": args.high_quality,
            "auto_segment": not args.no_auto_segment,
            "segment_duration": args.segment_duration,
        }
        process_input_folder(
            input_dir=args.input_dir, output_dir=args.output_dir, **transcribe_kwargs
        )
    # 個別ファイル処理モード
    elif args.input:
        for file in args.input:
            transcribe_file(
                file,
                beam_size=args.beam_size,
                model_name=args.model,
                output_mask=args.output_mask,
                high_quality=args.high_quality,
                auto_segment=not args.no_auto_segment,
                segment_duration=args.segment_duration,
                output_dir=args.output_dir,
            )
    else:
        parser.error("--input または --batch のいずれかを指定してください。")

    if args.shutdown:
        print("10秒後にシャットダウンします")
        os.system("timeout 10")
        os.system("shutdown -s -t 0")


if __name__ == "__main__":
    main()
