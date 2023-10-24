# ffmpeg -i input.mp4 -f mp3 -vn output.mp3
# ffmpeg -i ../○○.mp4 -filter:v fps=fps=0.1:round=down -vcodec png
# image_%04d.png

import whisper
import torch
import json


def secondtotime(time):
    time = round(time, 1)
    h = int(time // 3600)
    m = int((time - (3600 * h)) // 60)
    s = (time - (3600 * h)) - m * 60
    mm = int(time % 1 * 10)

    sh = str(h).zfill(2)
    sm = str(m).zfill(2)
    ss = str(int(s)).zfill(2)
    return f'{sh}:{sm}:{ss}.{mm}'


def wis(mp3file):
    model = whisper.load_model("large", device="cpu")  # large-v2, medium
    _ = model.half()
    _ = model.cuda()

    for m in model.modules():
        if isinstance(m, whisper.model.LayerNorm):
            m.float()

    print(mp3file)
    with torch.no_grad():
        result = model.transcribe(
            f"{mp3file}",
            verbose=True,
            language='japanese',
            # beam_size=5,
            # fp16=True,
            # without_timestamps=False
        )

    f = open(f'{mp3file}.srt', 'a', encoding='UTF-8')
    count = 0
    f.write("WEBVTT\n\n")
    for s in result["segments"]:
        count += 1
        f.write(str(count) + '\n' +
                secondtotime(s["start"]) + " --> " +
                secondtotime(s["end"]) + "\n" +
                json.dumps(s["text"], ensure_ascii=False).replace("\"", "") + "\n\n")

    f.close()



wis("./信号理論基礎18.mkv")
