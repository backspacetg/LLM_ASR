import kaldiio

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from .load_utils import load_model_and_tokenizer


def compute_fbank(sample,
                  num_mel_bins=23,
                  frame_length=25,
                  frame_shift=10,
                  dither=0.0):
    """ Extract fbank

        Args:
            sample: {key, wav, sample_rate, ...}

        Returns:
            {key, feat, wav, sample_rate, ...}
    """
    assert 'sample_rate' in sample
    assert 'wav' in sample
    assert 'key' in sample
    sample_rate = sample['sample_rate']
    waveform = sample['wav']
    waveform = waveform * (1 << 15)
    # Only keep key, feat, label
    mat = kaldi.fbank(waveform,
                      num_mel_bins=num_mel_bins,
                      frame_length=frame_length,
                      frame_shift=frame_shift,
                      dither=dither,
                      energy_floor=0.0,
                      sample_frequency=sample_rate)
    sample['feat'] = mat
    return sample


class ParaformerTranscriber:

    def __init__(self, model_path: str, config_path: str, device=torch.device("cpu")):
        model, config, tokenizer = load_model_and_tokenizer(
            model_path=model_path,
            cfg_path=config_path
        )
        self.config = config
        self.tokenizer = tokenizer
        self.ctc_blank_id = tokenizer.symbol_table['<blank>']
        self.config['dataset_conf']['fbank_conf']['dither'] = 0.0
        self.device = device
        self.model = model.to(device=device)
        self.model.eval()

    def transcibe(self, source_path:str, key: str="test", source_type="raw"):
        sample = {"key": key}
        if source_type == "raw":
            wavform, sr = torchaudio.load(source_path)
            sample['wav'] = wavform
            sample['sample_rate'] = sr
            sample = compute_fbank(sample=sample, **self.config['dataset_conf']['fbank_conf'])
            sample['feat'] = sample['feat'].unsqueeze(0).to(self.device)
        elif source_type == "kaldi_fbank":
            feat = torch.Tensor(kaldiio.load_mat(source_path)).unsqueeze(0).to(self.device)
            sample = {"feat": feat, "key": key}
        else:
            raise NotImplementedError(source_type)
        sample["feat_length"] = torch.Tensor([sample["feat"].shape[1]]).to(dtype=torch.int32, device=self.device)
        results = self.model.decode(
            "paraformer_greedy_search",
            sample["feat"],
            sample['feat_length'],
            beam_size=10,
            decoding_chunk_size=-1,
            blank_id=self.ctc_blank_id,
            )
        
        for _, hyps in results.items():
            tokens = hyps[0].tokens
            hyp = self.tokenizer.detokenize(tokens)[0]
            break

        return hyp
