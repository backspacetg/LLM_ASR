import numpy as np
import librosa
import soundfile as sf
from paraformer.auto.auto_model import AutoModel

class ASR:
    def __init__(self, model_path): # 加载模型
        self.model = AutoModel(model=model_path, device='cpu')
        print("warming the asr model...")
        self.model.generate(input=".\\example\\nihao.wav", key="001", disable_pbar=False)

    def transcribe(self, wav) -> str:
        sr, wav_data = wav[0], wav[1]
        wav_data = wav_data.astype(np.float32)
        wav_data = wav_data/np.max(np.abs(wav_data)+1.0e-5)
        print(sr, wav_data)
        if len(wav_data.shape) > 1: #2 channels
            wav_data = np.mean(wav_data, axis=1)
        wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=16000)
        sf.write("models\\tmp\\tmp.wav", wav_data, samplerate=16000)
        result = self.model.generate(input="models\\tmp\\tmp.wav", key="001", disable_pbar=False)[0]["text"]
        print("asr result", result)
        return result
