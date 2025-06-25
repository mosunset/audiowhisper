#!/usr/bin/env python
import argparse
import json
import os
import re

import torch
import whisper


def secondtotime(time):
    """
    指定された秒数を「時:分:秒.ミリ秒」の形式に変換する関数
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


def transcribe_file(
    filename, beam_size=5, model_name="turbo", output_mask=7, high_quality=False
):
    """
    Whisperモデルを使用して音声ファイルを文字起こしし、複数のファイルに出力する関数。

    出力ファイル:
      - <filename>.time.txt : タイムスタンプ付きテキスト
      - <filename>.srt      : SRT形式の字幕ファイル
      - <filename>.txt      : 全文テキスト
    """
    quality_str = " (高精度モード)" if high_quality else ""

    # 高精度モードが有効な場合、モデル名およびビームサイズを調整
    if high_quality:
        if model_name == "turbo":
            model_name = "large"
        beam_size = max(beam_size, 6)

    print(
        f"Transcribing: {filename} --beam_size {beam_size} --model {model_name}{quality_str}"
    )

    # CUDAの利用可能性をチェック
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Whisperモデルを読み込む
    # whisper.load_model(モデルサイズ, device=デバイス)
    # | モデルサイズ | パラメータ数  | 必要VRAM  | 相対速度  |
    # | tiny        | 39M         | ~1GB     | ~10x     |
    # | base        | 74M         | ~1GB     | ~7x      |
    # | small       | 244M        | ~2GB     | ~4x      |
    # | medium      | 769M        | ~5GB     | ~2x      |
    # | large       | 1550M       | ~10GB    | 1x       |
    # | turbo       | 809M        | ~6GB     | ~8x      |
    model = whisper.load_model(model_name, device=device)

    # GPUが利用可能な場合のみ半精度変換とLayerNorm調整を実行
    if device == "cuda":
        # モデルのパラメータを半精度へ変換する（GPU推論時に性能向上が見込まれる）
        _ = model.half()

        # モデルをGPUに移動（GPUが使用可能なら推論速度が大幅に上昇）
        _ = model.cuda()

        # LayerNormレイヤーはfloat16で精度が下がりやすいため、float32に戻す
        for m in model.modules():
            if isinstance(m, whisper.model.LayerNorm):
                m.float()

    # ここから音声ファイルの文字起こし設定説明（日本語翻訳）
    #
    # model : Whisper
    #   Whisperモデルのインスタンス
    #
    # audio : Union[str, np.ndarray, torch.Tensor]
    #   音声ファイルパス、または音声波形データ
    #
    # verbose : bool
    #   Trueの場合は詳細なログを表示、Falseは簡易ログ、Noneは出力なし
    #
    # temperature : Union[float, Tuple[float, ...]]
    #   サンプリング時の温度。複数指定の場合、それぞれのしきい値を超えた際に段階的に適用される
    #
    # compression_ratio_threshold : float
    #   この値を超えるgzip圧縮率になった場合は失敗とみなす
    #
    # logprob_threshold : float
    #   平均対数確率がこの値を下回った場合は失敗とみなす
    #
    # no_speech_threshold : float
    #   no_speechの確率がこの値を上回り、かつ平均対数確率が閾値を下回っている場合、
    #   セグメントを無音とみなす
    #
    # condition_on_previous_text : bool
    #   Trueのとき、前のセグメントの推定を次のセグメントの推定に反映する
    #
    # word_timestamps : bool
    #   単語レベルのタイムスタンプを取得し、各セグメントに含める
    #
    # prepend_punctuations : str
    #   word_timestampsがTrueの場合、指定した句読点を次の単語に結合
    #
    # append_punctuations : str
    #   word_timestampsがTrueの場合、指定した句読点を前の単語に結合
    #
    # initial_prompt : Optional[str]
    #   最初のセグメントに与えるテキスト。固有名詞等の補助として使用
    #
    # carry_initial_prompt : bool
    #   Trueの場合、initial_promptを各内部decode()呼び出しに持ち越す
    #   スペースが足りない場合は先頭が切り捨てられる
    #
    # decode_options : dict
    #   DecodingOptionsインスタンスの生成に使用されるキーワード引数
    #
    # clip_timestamps : Union[str, List[float]]
    #   秒単位の開始と終了のタイムスタンプをカンマ区切りで指定。最後の終了時刻はデフォルトでファイル終端
    #
    # hallucination_silence_threshold : Optional[float]
    #   word_timestampsがTrueのとき、誤判定（幻覚）を避けるため特定秒数以上の無音区間を処理から除外
    #
    # 戻り値:
    #   結果のテキスト("text")、セグメント単位の詳細("segments")、自動検出された言語("language")を含む辞書
    #
    transcribe_options = {
        "verbose": True,
        "language": "japanese",
        "beam_size": beam_size,
    }

    # 高精度モードの場合、追加パラメータを設定
    if high_quality:
        transcribe_options.update(
            {
                "condition_on_previous_text": True,
                "temperature": 0,
                "compression_ratio_threshold": 2.4,
                "no_speech_threshold": 0.6,
            }
        )

    # 推論時は勾配計算を無効化してメモリ使用量・計算量を削減
    with torch.no_grad():
        result = model.transcribe(filename, **transcribe_options)

    # 出力用のフォルダを作成
    folder_name = os.path.splitext(filename)[0]
    os.makedirs(folder_name, exist_ok=True)

    # <filename>.time.txt（タイムスタンプ付きテキスト出力）
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

    # <filename>.srt（SRT形式字幕ファイル）
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

    # <filename>.txt（全文テキスト）
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
    args = parser.parse_args()

    for file in args.input:
        transcribe_file(
            file,
            beam_size=args.beam_size,
            model_name=args.model,
            output_mask=args.output_mask,
            high_quality=args.high_quality,
        )

    if args.shutdown:
        print("10秒後にシャットダウンします")
        os.system("timeout 10")
        os.system("shutdown -s -t 0")


if __name__ == "__main__":
    main()
