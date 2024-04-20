import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import json

# config
model_id = "kotoba-tech/kotoba-whisper-v1.0"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load model
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True)
model.to(device)
processor = AutoProcessor.from_pretrained(model_id)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    language="japanese",
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    torch_dtype=torch_dtype,
    device=device,
)

audio_file = "./物理学概論4.mp3"

result = pipe(audio_file)
print(result["text"])

with open(f"{audio_file}.json", "w", encoding="UTF-8") as f:
    json.dump(result, f, indent=4, ensure_ascii=False)
