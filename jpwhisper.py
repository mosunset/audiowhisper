# コードの詳細
# https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0

import json
import os
import time

import torch
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from datasets import load_dataset, Audio

# 設定
model_id = "kotoba-tech/kotoba-whisper-v1.0"  # 使用するモデルのID
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # GPUが利用可能な場合はfloat16、そうでない場合はfloat32を使用
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # GPUが利用可能な場合はGPU、そうでない場合はCPUを使用

# load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True, attn_implementation="sdpa"
)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=25,
    batch_size=16,
    torch_dtype=torch_dtype,
    device=device,
    return_timestamps=True,
)

audio_file = "./物理学概論4.mp3"  # 入力音声ファイル

# # load sample audio & downsample to 16kHz
# dataset = load_dataset("japanese-asr/ja_asr.reazonspeech_test", split="test")
# dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
# sample = {"array": np.concatenate([i["array"] for i in dataset[:20]["audio"]]), "sampling_rate": dataset[0]['audio']['sampling_rate'], "path": "tmp"}


# 音声認識の実行
start_time = time.time()
print(f"{start_time:.2f}")
result = pipe(audio_file)
end_time = time.time()
print(f"{end_time:.2f}")

print(f"Audio recognition completed in {end_time - start_time:.2f} seconds.")
print(result["text"])  # 認識結果の出力

# 認識結果をJSONファイルに保存
with open(f"{audio_file}.json", "w", encoding="UTF-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
