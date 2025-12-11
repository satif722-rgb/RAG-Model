import whisper
import os
import json
model=whisper.load_model("large-v2")
audios=os.listdir("audios")
for audio in audios:
    if ("_" in audio):
        number=audio.split("_")[0]
        title=audio.split("_")[1]
        print(number,title)
        result=model.transcribe(audio=f"audios/{audio}",
                                language='hi',
                                task="translate",
                                word_timestamps=False)
        chunks=[]
        for segment in result["segments"]:
            chunks.append({"number":number,
                           "titel":title,
                           "start":segment["start"],
                           "end":segment["end"],
                           "text":segment["text"]})
        chunks_with_metadata={"chunks":chunks,"text":result["text"]}
        with open (f"json/{audio}.json","w") as f:
            json.dump(chunks_with_metadata,f) 
