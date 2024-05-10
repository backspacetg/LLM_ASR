import os
import time
import numpy as np

import torch
import torchaudio
import torchaudio.transforms as T
from basic_wenet.paraformer_svd.transcriber import ParaformerTranscriber as SVDTranscriber

class ASR:
    def __init__(self, model_path, use_svd_model=True): # 加载模型
        self.model = SVDTranscriber(
            model_path=os.path.join(model_path, "model.pt"),
            config_path=os.path.join(model_path, "train.yaml"),
            use_svd=use_svd_model,
            keep_rate_att=0.25,
            keep_rate_linear=0.6
            )
        print("warming the asr model...")
        res = self.model.transcibe(source_path=os.path.join("example", "nihao.wav"), key="001")
        print(res)

    def transcribe(self, wav) -> str:
        sr, wav_data = wav[0], wav[1]
        wav_data = wav_data.astype(np.float32)
        wav_data = wav_data/np.max(np.abs(wav_data)+1.0e-5)
        print(sr, wav_data)
        if len(wav_data.shape) > 1: #2 channels
            wav_data = np.mean(wav_data, axis=1)
        resampler = T.Resample(orig_freq=sr, new_freq=16000)
        wav_data = torch.from_numpy(wav_data).unsqueeze(0)
        wav_data = resampler(wav_data)
        torchaudio.save(os.path.join("models", "tmp", "tmp.wav"), wav_data, sample_rate=16000)
        wav_time = wav_data.shape[1]/16000
        start_time = time.time()
        result = self.model.transcibe(source_path=os.path.join("models", "tmp", "tmp.wav"), key="001")
        time_used = time.time() - start_time
        print("asr result: {}, RTF: {}".format(result, time_used/wav_time))
        return result
