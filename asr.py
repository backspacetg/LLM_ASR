import os
import time
import numpy as np
import librosa
import soundfile as sf
from omegaconf import OmegaConf
from paraformer.auto.auto_model import AutoModel

class ASR:
    def __init__(self, model_path, use_svd_model=True): # 加载模型
        if use_svd_model:
            model_args = OmegaConf.load(os.path.join(model_path, "config.yaml"))
            model_args["model_path"] = model_path # os.path.join(model_path, "model.pt.avg3")
            model_args["init_param"] = os.path.join(model_path, "model.pt.avg3")
            model_args["tokenizer_conf"]["token_list"] = os.path.join(model_path, "tokens.json")
            model_args["tokenizer_conf"]["seg_dict"] = os.path.join(model_path, "seg_dict")
            model_args["frontend_conf"]["cmvn_file"] = os.path.join(model_path, "am.mvn")
            model_args["output_dir"] = None
            self.model = AutoModel(**model_args)
        else:
            self.model = AutoModel(model=model_path, device='cpu')
        print("warming the asr model...")
        res = self.model.generate(input=".\\example\\nihao.wav", key="001", disable_pbar=False)
        print(res)

    def transcribe(self, wav) -> str:
        sr, wav_data = wav[0], wav[1]
        wav_data = wav_data.astype(np.float32)
        wav_data = wav_data/np.max(np.abs(wav_data)+1.0e-5)
        print(sr, wav_data)
        if len(wav_data.shape) > 1: #2 channels
            wav_data = np.mean(wav_data, axis=1)
        wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=16000)
        sf.write("models\\tmp\\tmp.wav", wav_data, samplerate=16000)
        wav_time = len(wav_data)/16000
        start_time = time.time()
        result = self.model.generate(input="models\\tmp\\tmp.wav", key="001", disable_pbar=False)[0]["text"]
        time_used = time.time() - start_time
        print("asr result: {}, RTF: {}".format(result, time_used/wav_time))
        return result
