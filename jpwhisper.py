import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json

# 設定
model_id = "kotoba-tech/kotoba-whisper-v1.0"  # 使用するモデルのID
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # GPUが利用可能な場合はfloat16、そうでない場合はfloat32を使用
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # GPUが利用可能な場合はGPU、そうでない場合はCPUを使用

# モデルのロード
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)  # モデルをデバイスに移動
processor = AutoProcessor.from_pretrained(model_id)  # プロセッサのロード

# パイプラインの作成
pipe = pipeline(
    "automatic-speech-recognition",  # タスクの指定
    model=model,
    language="japanese",  # 入力言語の指定
    tokenizer=processor.tokenizer,  # トークナイザの指定
    feature_extractor=processor.feature_extractor,  # 特徴量抽出器の指定
    max_new_tokens=128,  # 生成する最大トークン数
    torch_dtype=torch_dtype,  # 使用するデータ型
    device=device  # 使用するデバイス
)

audio_file = "./物理学概論4.mp3"  # 入力音声ファイル

# 音声認識の実行
result = pipe(audio_file)
print(result["text"])  # 認識結果の出力

# 認識結果をJSONファイルに保存
with open(f"{audio_file}.json", "w", encoding="UTF-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
