#!/usr/bin/env python
"""
ファイル管理機能
"""

import glob
import os
import shutil


def get_input_files(input_dir="input"):
    """
    inputフォルダ内の音声・動画ファイルを取得する関数

    Args:
        input_dir: inputフォルダのパス

    Returns:
        input_files: 処理対象のファイルパスのリスト
    """
    if not os.path.exists(input_dir):
        print(f"inputフォルダが見つかりません: {input_dir}")
        return []

    # 対応する音声・動画ファイルの拡張子
    audio_extensions = ["*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg", "*.aac"]
    video_extensions = ["*.mp4", "*.avi", "*.mov", "*.mkv", "*.wmv", "*.flv"]
    all_extensions = audio_extensions + video_extensions

    input_files = []

    for extension in all_extensions:
        pattern = os.path.join(input_dir, extension)
        files = glob.glob(pattern)
        input_files.extend(files)

    # 大文字小文字を区別しない場合の拡張子も検索
    for extension in all_extensions:
        pattern = os.path.join(input_dir, extension.upper())
        files = glob.glob(pattern)
        input_files.extend(files)

    # 重複を除去してソート
    input_files = sorted(list(set(input_files)))

    print(f"inputフォルダ内のファイル数: {len(input_files)}")
    for file in input_files:
        print(f"  処理対象: {os.path.basename(file)}")

    return input_files


def move_processed_file(file_path, output_dir="output"):
    """
    処理済みファイルをoutputフォルダに移動する関数

    Args:
        file_path: 移動するファイルのパス
        output_dir: 移動先のディレクトリ
    """
    try:
        # ファイル名を取得（拡張子なし）
        filename = os.path.basename(file_path)
        name_without_ext = os.path.splitext(filename)[0]

        # output/<ファイル名>/フォルダを作成
        output_folder = os.path.join(output_dir, name_without_ext)
        os.makedirs(output_folder, exist_ok=True)

        # 移動先のパスを作成
        destination = os.path.join(output_folder, filename)

        # 同名ファイルが存在する場合の処理
        counter = 1
        original_destination = destination
        while os.path.exists(destination):
            name, ext = os.path.splitext(filename)
            destination = os.path.join(output_folder, f"{name}_{counter}{ext}")
            counter += 1

        # ファイルを移動
        shutil.move(file_path, destination)
        print(f"処理済みファイルを移動: {filename} -> {output_folder}/")

        return True

    except Exception as e:
        print(f"ファイル移動に失敗しました: {file_path} - {e}")
        return False


def process_input_folder(input_dir="input", output_dir="output", **transcribe_kwargs):
    """
    inputフォルダ内のファイルを連続処理する関数

    Args:
        input_dir: inputフォルダのパス
        output_dir: 出力ディレクトリ
        **transcribe_kwargs: transcribe_file関数に渡す引数
    """
    from .transcription import transcribe_file

    # inputフォルダ内のファイルを取得
    input_files = get_input_files(input_dir)

    if not input_files:
        print("処理対象のファイルが見つかりませんでした。")
        return

    print("\n=== inputフォルダの連続処理を開始 ===")
    print(f"処理対象ファイル数: {len(input_files)}")
    print(f"出力先: {output_dir}")
    print("=" * 50)

    # 各ファイルを順次処理
    for i, file_path in enumerate(input_files, 1):
        print(f"\n[{i}/{len(input_files)}] 処理中: {os.path.basename(file_path)}")
        print("-" * 40)

        try:
            # 文字起こし処理
            transcribe_file(file_path, **transcribe_kwargs)

            # 処理が成功したらファイルを移動
            if move_processed_file(file_path, output_dir):
                print(f"✓ 処理完了: {os.path.basename(file_path)}")
            else:
                print(f"⚠ 処理完了（移動失敗）: {os.path.basename(file_path)}")

        except Exception as e:
            print(f"✗ 処理失敗: {os.path.basename(file_path)} - {e}")
            continue

    print("\n=== 連続処理完了 ===")
    print(f"処理済みファイルは '{output_dir}' フォルダに移動されました。")


def create_output_folders():
    """
    必要なフォルダを作成する関数
    """
    folders = ["input", "output"]

    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            print(f"フォルダを作成しました: {folder}")
        else:
            print(f"フォルダが既に存在します: {folder}")
