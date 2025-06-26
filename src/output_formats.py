#!/usr/bin/env python
"""
出力形式の処理関数
"""

import json
import os
import re

from .utils import secondtotime


def write_time_txt(result, output_folder, filename):
    """
    <filename>.time.txt（タイムスタンプ付きテキスト出力）
    """
    time_txt_filename = os.path.join(
        output_folder, f"{os.path.basename(filename)}.time.txt"
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


def write_srt(result, output_folder, filename):
    """
    <filename>.srt（SRT形式字幕ファイル）
    """
    srt_filename = os.path.join(output_folder, f"{os.path.basename(filename)}.srt")
    with open(srt_filename, "w", encoding="UTF-8") as f:
        count = 0
        for s in result["segments"]:
            count += 1
            replaced_text3 = json.dumps(s["text"], ensure_ascii=False).replace('"', "")
            f.write(
                f"{count}\n"
                f"{secondtotime(s['start'])} --> {secondtotime(s['end'])}\n"
                f"{replaced_text3}\n\n"
            )
    print(f"Created {srt_filename}")


def write_txt(result, output_folder, filename):
    """
    <filename>.txt（全文テキスト）
    """
    txt_filename = os.path.join(output_folder, f"{os.path.basename(filename)}.txt")
    document = ""
    for s in result["segments"]:
        document += json.dumps(s["text"], ensure_ascii=False).replace('"', "") + "。"
    with open(txt_filename, "w", encoding="UTF-8") as f:
        f.write(document)
    print(f"Created {txt_filename}")


def write_output_files(result, filename, output_mask, output_dir="output"):
    """
    指定されたマスクに基づいて出力ファイルを作成
    """
    # ファイル名を取得（拡張子なし）
    filename_without_ext = os.path.splitext(os.path.basename(filename))[0]

    # output/<ファイル名>/フォルダを作成
    output_folder = os.path.join(output_dir, filename_without_ext)
    os.makedirs(output_folder, exist_ok=True)

    # <filename>.time.txt（タイムスタンプ付きテキスト出力）
    if output_mask & 1:
        write_time_txt(result, output_folder, filename)

    # <filename>.srt（SRT形式字幕ファイル）
    if output_mask & 2:
        write_srt(result, output_folder, filename)

    # <filename>.txt（全文テキスト）
    if output_mask & 4:
        write_txt(result, output_folder, filename)
