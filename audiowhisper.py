import whisper
import torch
import json
import re

# 自動要約
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.mecab_tokenizer import MeCabTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from pysummarization.nlp_base import NlpBase
from pysummarization.similarityfilter.tfidf_cosine import TfIdfCosine
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
import MeCab


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


def wis(filename):
    """
    Whisperモデルを使って音声ファイルを認識し、結果をファイルに出力する関数
    """
    model = whisper.load_model(
        "small"
    )  # , device="cpu")  # large-3, large-2, large, medium, small, base, tiny
    # _ = model.half()
    # _ = model.cuda()

    # for m in model.modules():
    #     if isinstance(m, whisper.model.LayerNorm):
    #         m.float()

    print(filename)
    with torch.no_grad():
        result = model.transcribe(
            f"{filename}",
            verbose=True,
            language="japanese",
            beam_size=4,
            # fp16=True,
            # without_timestamps=False
        )

    # with open(f"{filename}.json", "w", encoding="UTF-8") as f:
    #     json.dump(result, f, indent=4, ensure_ascii=False)

    with open(f"{filename}.time.txt", "w", encoding="UTF-8") as f:
        count = 0
        temp = -1
        for s in result["segments"]:
            count += 1
            if temp == re.sub(r"\..$", "", secondtotime(s["start"])):
                f.write("\n")
            else:
                f.write(
                    ("" if count == 1 else "\n\n")
                    + re.sub(r"\..$", "", secondtotime(s["start"]))
                    + " "
                    + str(count)
                    + "\n"
                )

            temp = re.sub(r"\..$", "", secondtotime(s["end"]))
            f.write(json.dumps(s["text"], ensure_ascii=False).replace('"', ""))

    with open(f"{filename}.srt", "w", encoding="UTF-8") as f:
        count = 0
        for s in result["segments"]:
            count += 1
            f.write(
                str(count)
                + "\n"
                + secondtotime(s["start"])
                + " --> "
                + secondtotime(s["end"])
                + "\n"
                + json.dumps(s["text"], ensure_ascii=False).replace('"', "")
                + "\n\n"
            )

    with open(f"{filename}.txt", "w", encoding="UTF-8") as f:
        count = 0
        temp = -1
        document = ""
        for s in result["segments"]:
            document += (
                json.dumps(s["text"], ensure_ascii=False).replace('"', "") + "。"
            )
        f.write(document)


def summarization(filename, document):
    """
    入力文書を要約し、結果をファイルに出力する関数
    """
    print("summarization")
    similarity_limit = 0.65
    # 自動要約のオブジェクト
    auto_abstractor = AutoAbstractor()
    # トークナイザー設定（MeCab使用）
    auto_abstractor.tokenizable_doc = MeCabTokenizer()
    # 区切り文字設定
    auto_abstractor.delimiter_list = ["。", "\n"]
    # 抽象化&フィルタリングオブジェクト
    abstractable_doc = TopNRankAbstractor()
    # 文書要約
    result_dict1 = auto_abstractor.summarize(document, abstractable_doc)

    print(f"summarize_result")
    with open(f"{filename}.summarize.txt", "w", encoding="UTF-8") as f:
        # f.write("[要約結果]")
        for sentence in result_dict1["summarize_result"]:
            f.write(sentence)
        f.write("\n\n")
        f.write(json.dumps(result_dict1, ensure_ascii=False))

    # NLPオブジェクト
    nlp_base = NlpBase()
    # トークナイザー設定（MeCab使用）
    nlp_base.tokenizable_doc = MeCabTokenizer()
    # 類似性フィルター
    similarity_filter = TfIdfCosine()
    # NLPオブジェクト設定
    similarity_filter.nlp_base = nlp_base
    # 類似性limit：limit超える文は切り捨て
    similarity_filter.similarity_limit = similarity_limit

    # 自動要約のオブジェクト
    auto_abstractor = AutoAbstractor()
    # トークナイザー設定（MeCab使用）
    auto_abstractor.tokenizable_doc = MeCabTokenizer()
    # 区切り文字設定
    auto_abstractor.delimiter_list = ["。", "\n"]
    # 抽象化&フィルタリングオブジェクト
    abstractable_doc = TopNRankAbstractor()
    # 文書要約（similarity_filter機能追加）
    result_dict2 = auto_abstractor.summarize(
        document, abstractable_doc, similarity_filter
    )

    print("summarize_result2")
    with open(f"{filename}.summarize_sf.txt", "w", encoding="UTF-8") as f:
        # f.write("[要約結果：Similarity Filter機能]")
        for sentence in result_dict2["summarize_result"]:
            f.write(sentence)
        f.write("\n\n")
        f.write(json.dumps(result_dict2, ensure_ascii=False))


def main(files):
    """
    メイン関数
    """
    for file in files:
        wis(file)


def summarize(files):
    """
    要約処理を行う関数
    """
    for file in files:
        with open(f"{file}.txt", encoding="UTF-8") as f:
            summarization(file, f.readline().replace(" ", ""))


if __name__ == "__main__":
    files = [
        "./物理学概論4.mp4",
    ]
    main(files)
    summarize(files)
