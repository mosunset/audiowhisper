#!/usr/bin/env python
import whisper
import torch
import json
import re
import argparse
import os


def secondtotime(time):
    """
    秒数を時分秒.ミリ秒の形式に変換する関数
    """
    time = round(time, 1)
    h = int(time // 3600)
    m = int((time - (3600 * h)) // 60)
    s = (time - (3600 * h)) - m * 60
    mm = int(time % 1 * 10)

    sh = str(h).zfill(2)
    sm = str(m).zfill(2)
    ss = str(int(s)).zfill(2)
    return f"{sh}:{sm}:{ss}.{mm}"


def transcribe_file(filename, beam_size=5, model_name="turbo", output_mask=7):
    """
    Whisperモデルを使って音声ファイルを認識し、結果を複数のファイルに出力する関数

    出力ファイル:
      - <filename>.time.txt : タイムスタンプ付きテキスト
      - <filename>.srt      : SRT形式字幕ファイル
      - <filename>.txt      : 全文テキスト
    """
    print(f"Transcribing: {filename} --beam_size {beam_size} --model {model_name}")

    # Whisperモデルの読み込み
    # whisper.load_model(モデルサイズ, device=デバイス) を使って指定したサイズのモデルを読み込みます。
    # | モデルサイズ | パラメータ数 | 必要 VRAM | 相対速度 |
    # | tiny       | 39M        | ~1GB     | ~10x    |
    # | base       | 74M        | ~1GB     | ~7x     |
    # | small      | 244M       | ~2GB     | ~4x     |
    # | medium     | 769M       | ~5GB     | ~2x     |
    # | large      | 1550M      | ~10GB    | 1x      | "large-v3", "large-v2"
    # | turbo      | 809M       | ~6GB     | ~8x     |

    model = whisper.load_model(
        model_name, device="cpu"
    )  # CPU上で "small" サイズのモデルを読み込み

    # モデルのパラメータを半精度 (float16) に変換
    # → GPUでの推論時に計算速度とメモリ使用効率が向上する可能性があります。
    _ = model.half()

    # モデルをGPUに移動
    # → GPUが利用可能な場合、計算速度が大幅に向上します。
    _ = model.cuda()

    # ※ 注意：LayerNorm レイヤーは半精度では数値精度が低下しやすいため、float32に戻します。
    # モデル内の全モジュールを走査し、LayerNorm のインスタンスであれば精度を float32 に設定。
    for m in model.modules():
        if isinstance(m, whisper.model.LayerNorm):
            m.float()

    # 勾配計算を無効化（推論時は不要なため、メモリ使用量と計算コストを削減）
    with torch.no_grad():
        # 音声認識（文字起こし）を実行
        result = model.transcribe(
            filename,  # ★ filename: 認識対象の音声ファイルのパスまたはオーディオデータを指定します。
            verbose=True,  # ★ verbose: Trueの場合、認識処理の詳細なログ（進行状況や内部情報）を出力します。
            language="japanese",  # ★ language: 音声の言語を指定します。ここでは "japanese" と明示して、言語判定の負荷を軽減し認識精度向上を狙います。
            beam_size=beam_size,  # ★ beam_size: ビームサーチのビーム幅を指定します。
            #     ビーム幅が大きいほど、複数の候補を同時に検討してより最適な結果を得やすくなりますが、計算負荷も増加します。
            # 以下はコメントアウトされていますが、必要に応じて利用可能な引数です:
            # fp16=False,         # ★ fp16: 半精度 (float16) での計算を有効にするかどうかを指定します。
            #     デフォルトではGPU環境での高速化のためにTrueとなることが多いですが、環境や精度との兼ね合いで変更可能です。
            # without_timestamps=False  # ★ without_timestamps: Falseの場合、認識結果にタイムスタンプ情報が含まれます。
            #     Trueにするとタイムスタンプを除外した結果となります。
        )

    # 出力先フォルダを作成
    folder_name = os.path.splitext(filename)[0]
    os.makedirs(folder_name, exist_ok=True)

    # <filename>.time.txt : タイムスタンプ付きテキスト
    if output_mask & 1:
        time_txt_filename = os.path.join(
            folder_name, f"{os.path.basename(filename)}.time.txt"
        )
        with open(time_txt_filename, "w", encoding="UTF-8") as f:
            count = 0
            temp = -1
            for s in result["segments"]:
                count += 1
                start_formatted = re.sub(r"\..$", "", secondtotime(s["start"]))
                end_formatted = re.sub(r"\..$", "", secondtotime(s["end"]))
                if temp == start_formatted:
                    f.write("\n")
                else:
                    f.write(
                        ("" if count == 1 else "\n\n")
                        + start_formatted
                        + " "
                        + str(count)
                        + "\n"
                    )
                temp = end_formatted
                f.write(json.dumps(s["text"], ensure_ascii=False).replace('"', ""))
        print(f"Created {time_txt_filename}")

    # <filename>.srt      : SRT形式字幕ファイル
    if output_mask & 2:
        srt_filename = os.path.join(folder_name, f"{os.path.basename(filename)}.srt")
        with open(srt_filename, "w", encoding="UTF-8") as f:
            count = 0
            for s in result["segments"]:
                count += 1
                replaced_text3 = json.dumps(s["text"], ensure_ascii=False).replace(
                    '"', ""
                )
                f.write(
                    f"{count}\n"
                    f"{secondtotime(s['start'])} --> {secondtotime(s['end'])}\n"
                    f"{replaced_text3}\n\n"
                )
        print(f"Created {srt_filename}")

    # <filename>.txt      : 全文テキスト
    if output_mask & 4:
        txt_filename = os.path.join(folder_name, f"{os.path.basename(filename)}.txt")
        document = ""
        for s in result["segments"]:
            document += (
                json.dumps(s["text"], ensure_ascii=False).replace('"', "") + "。"
            )
        with open(txt_filename, "w", encoding="UTF-8") as f:
            f.write(document)
        print(f"Created {txt_filename}")


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
        help="入力ファイルのパス。複数指定可能。",
    )
    parser.add_argument(
        "--beam_size",
        type=int,
        default=5,
        help="文字起こし時のbeam size",
    )
    parser.add_argument(
        "-m",
        "--model",
        default="turbo",
        help="Whisperモデルのサイズ (tiny, base, small, medium, large, turbo)",
    )
    parser.add_argument(
        "--output_mask",
        type=int,
        default=7,
        help="出力ファイルのマスク (0～7)。1=<filename>.time.txt : タイムスタンプ付きテキスト, 2=<filename>.srt : SRT形式字幕ファイル, 4=<filename>.txt : 全文テキスト の組み合わせ",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="処理終了後にシャットダウンする",
    )
    args = parser.parse_args()

    for file in args.input:
        transcribe_file(
            file,
            beam_size=args.beam_size,
            model_name=args.model,
            output_mask=args.output_mask,
        )

    if args.shutdown:
        print("10秒後にシャットダウンします")
        os.system("timeout 10")
        os.system("shutdown -s -t 0")


if __name__ == "__main__":
    main()
