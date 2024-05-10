import logging

import torch
import torchaudio

from .load_utils import load_model_and_tokenizer
from ..paraformer.transcriber import compute_fbank


logger = logging.getLogger(__name__)


class ParaformerTranscriber:

    def __init__(self, model_path: str, config_path: str, use_svd=True, keep_rate_linear=0.5, keep_rate_att=0.3, device=torch.device("cpu")):
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
        if use_svd:  
            logger.info("SVD stats of the 1st encoder layer: ")       
            for i, l in enumerate(self.model.encoder.encoders):
                log = i==0
                l.self_attn.linear_out.apply_svd(keep_rate_att, log=log)
                l.self_attn.linear_q_k_v.apply_svd(keep_rate_att, log=log)
                l.feed_forward.w_1.apply_svd(keep_rate_linear, log=log)
                l.feed_forward.w_2.apply_svd(keep_rate_linear, log=log)
        total = 0
        encoder_count = 0
        for n, p in self.model.named_parameters():
            total += p.numel()/1e6
            if "encoder" in n:
                encoder_count += p.numel()/1e6
        logger.info("{:.4f} for encoder and {:.4f}M in total parameters after SVD (linear: {}, attention: {})".format(
            encoder_count, total, keep_rate_linear, keep_rate_att
        ))  
        self.model.eval()


    def transcibe(self, source_path:str, key: str="test", source_type="raw"):
        sample = {"key": key}
        if source_type == "raw":
            wavform, sr = torchaudio.load(source_path)
            sample['wav'] = wavform
            sample['sample_rate'] = sr
            sample = compute_fbank(sample=sample, **self.config['dataset_conf']['fbank_conf'])
            sample['feat'] = sample['feat'].unsqueeze(0).to(self.device)
        else:
            raise NotImplementedError(source_type)
        sample["feat_length"] = torch.Tensor([sample["feat"].shape[1]]).to(dtype=torch.int32, device=self.device)
        with torch.no_grad():
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