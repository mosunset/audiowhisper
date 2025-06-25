#!/usr/bin/env python
"""
文字起こしのコア機能
"""

import torch
import whisper

from .audio_utils import find_silence_segments, get_audio_duration
from .output_formats import write_output_files


def transcribe_segment(
    model,
    audio_path,
    start_time,
    end_time,
    segment_index,
    total_segments,
    beam_size=5,
    high_quality=False,
):
    """
    音声ファイルの特定セグメントを文字起こしする関数
    """
    print(
        f"\nセグメント {segment_index}/{total_segments} を処理中: {start_time:.1f}s - {end_time:.1f}s"
    )

    transcribe_options = {
        "verbose": False,  # セグメント処理時は詳細ログを無効化
        "language": "japanese",
        "beam_size": beam_size,
        "clip_timestamps": f"{start_time},{end_time}",
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
        result = model.transcribe(audio_path, **transcribe_options)

    # セグメントのタイムスタンプを調整
    for segment in result["segments"]:
        segment["start"] += start_time
        segment["end"] += start_time

    return result


def merge_results(results_list):
    """
    複数の文字起こし結果を統合する関数
    """
    merged_result = {"text": "", "segments": [], "language": "japanese"}

    for result in results_list:
        merged_result["text"] += result["text"] + " "
        merged_result["segments"].extend(result["segments"])

    # セグメントを開始時間でソート
    merged_result["segments"].sort(key=lambda x: x["start"])

    return merged_result


def setup_model(model_name, device):
    """
    Whisperモデルをセットアップする関数
    """
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

    return model


def transcribe_file(
    filename,
    beam_size=5,
    model_name="turbo",
    output_mask=7,
    high_quality=False,
    auto_segment=True,
    segment_duration=1800,
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

    # モデルをセットアップ
    model = setup_model(model_name, device)

    # 音声ファイルの長さをチェック
    audio_duration = get_audio_duration(filename)
    if audio_duration is None:
        print("音声ファイルの長さ取得に失敗しました。通常の処理を続行します。")
        use_segmentation = False
    else:
        print(
            f"音声ファイルの長さ: {audio_duration:.1f}秒 ({audio_duration / 60:.1f}分)"
        )
        # 30分（1800秒）以上の場合、自動セグメント分割を使用
        use_segmentation = auto_segment and audio_duration > segment_duration

    if use_segmentation:
        print(
            f"長時間音声ファイルのため、{segment_duration / 60:.0f}分単位でセグメント分割して処理します。"
        )

        # 音声ファイルをセグメントに分割
        segments = find_silence_segments(filename, segment_duration=segment_duration)

        if len(segments) <= 1:
            print("セグメント分割が不要なため、通常の処理を続行します。")
            use_segmentation = False
        else:
            # 各セグメントを個別に処理
            results_list = []
            for i, (start_time, end_time) in enumerate(segments):
                result = transcribe_segment(
                    model,
                    filename,
                    start_time,
                    end_time,
                    i + 1,
                    len(segments),
                    beam_size,
                    high_quality,
                )
                results_list.append(result)

            # 結果を統合
            print("セグメント結果を統合中...")
            result = merge_results(results_list)
            print("統合完了")

    if not use_segmentation:
        # 通常の処理（元のコード）
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

    # 出力ファイルを作成
    write_output_files(result, filename, output_mask)
