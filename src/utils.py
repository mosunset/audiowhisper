#!/usr/bin/env python
"""
共通のユーティリティ関数
"""


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
