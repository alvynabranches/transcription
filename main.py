import os
from speechbrain.pretrained import EncoderDecoderASR as ASR
from tqdm import tqdm
from youtube_dl import YoutubeDL as YDL
from pydub import AudioSegment
from pydub.silence import split_on_silence

def download_ytvid(vid_url:str, outfile:str):
    with YDL({"format": "mp4", "outtmpl": outfile}) as ydl:
        ydl.download([vid_url])

def transcribe_1(file:str):
    asr_model = ASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech", savedir="pretrained_models/asr-transformer-treansformlm-librispeech")
    return asr_model.transcribe_file(file)

def transcribe_2(file:str):
    asr_model = ASR.from_hparams(source="speechbrain/asr-crdnn-transformerlm-librispeech", savedir="pretrained_models/asr-crdnn-transformerlm-librispeech")
    return asr_model.transcribe_file(file)

def transcribe(audios, transcribe_func):
        full_text = ""
        for audio in tqdm(audios):
            text = transcribe_func(audio)
            full_text = full_text + " " + text
        return full_text

yt_link = "https://www.youtube.com/watch?v=ZZG9a42b5nE"
temp_dir = os.path.join(os.path.dirname(__file__), "temp")
audio_file = os.path.join(temp_dir, "audio_{}.wav")
video_file = os.path.join(temp_dir, "video.mp4")

if __name__ == "__main__":
    # if not os.path.isfile(video_file):
    #     download_ytvid([yt_link], video_file)

    # else:
    #     print("Already Downloaded!")
    
    # if not os.path.isfile(audio_file):
    #     print("Exported to Audio File!")
    # else:
    #     print("Already Exported to Audio File!")
    
    # audio_lst = []
    # audio = AudioSegment.from_file(video_file)
    # audio.export(os.path.join(temp_dir, "audio.wav"))
    # audios = split_on_silence(audio, silence_thresh=audio.dBFS-16)
    # for n, aud in tqdm(enumerate(audios)):
    #     aud_file = audio_file.format(n)
    #     aud.export(aud_file, format="wav")
    #     audio_lst.append(aud_file)
    # del audio, audios
    # print()

    audio_lst = [
        os.path.join(temp_dir,  "audio_0.wav"),
        os.path.join(temp_dir,  "audio_1.wav"),
        os.path.join(temp_dir,  "audio_2.wav"),
        os.path.join(temp_dir,  "audio_3.wav"),
        os.path.join(temp_dir,  "audio_4.wav"),
        os.path.join(temp_dir,  "audio_5.wav"),
        os.path.join(temp_dir,  "audio_6.wav"),
        os.path.join(temp_dir,  "audio_7.wav"),
        os.path.join(temp_dir,  "audio_8.wav"),
        os.path.join(temp_dir,  "audio_9.wav"),
        os.path.join(temp_dir, "audio_10.wav"),
        os.path.join(temp_dir, "audio_11.wav"),
    ]
    
    transformerlm_full_text = transcribe(audio_lst, transcribe_1)
    print(transformerlm_full_text)
    
    wav2vec2_full_text = transcribe(audio_lst, transcribe_2)
    print(wav2vec2_full_text)
