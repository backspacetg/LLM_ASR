import re
import os
import logging

import yaml

import torch

from ..utils.cmvn import load_cmvn
from ..transformer.cmvn import GlobalCMVN
from ..transformer.ctc import CTC
from .layers import SanmEncoder, SanmDecoder
from .paraformer import Predictor, Paraformer
from .tokenizer import ParaformerTokenizer

logger = logging.getLogger(__name__)

def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    logging.info('Checkpoint: loading from checkpoint %s' % path)
    checkpoint = torch.load(path, map_location='cpu')
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint,
                                                          strict=False)
    for key in missing_keys:
        logging.info("missing tensor: {}".format(key))
    for key in unexpected_keys:
        logging.info("unexpected tensor: {}".format(key))
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def load_model_and_tokenizer(model_path, cfg_path):

    model_path = model_path

    configs = yaml.load(
        open(cfg_path, "r"), 
        Loader=yaml.FullLoader
        )

    mean, istd = load_cmvn(configs['cmvn_conf']['cmvn_file'],
                            configs['cmvn_conf']['is_json_cmvn'])
    global_cmvn = GlobalCMVN(
        torch.from_numpy(mean).float(),
        torch.from_numpy(istd).float())

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    encoder = SanmEncoder(
        input_dim,
        global_cmvn=global_cmvn,
        **configs['encoder_conf'])

    decoder = SanmDecoder(vocab_size,
        encoder.output_size(),
        **configs['decoder_conf']
    )

    ctc = CTC(
        vocab_size,
        encoder.output_size(),
        blank_id=configs['ctc_conf']['ctc_blank_id']
        if 'ctc_conf' in configs else 0)

    predictor = Predictor(**configs['predictor_conf'])
    model = Paraformer(
        vocab_size=vocab_size,
        encoder=encoder,
        decoder=decoder,
        predictor=predictor,
        ctc=ctc,
        **configs['model_conf'],
        special_tokens=configs.get('tokenizer_conf',
                                    {}).get('special_tokens', None),
    )

    infos = load_checkpoint(model, model_path)
    configs["init_infos"] = infos
    logger.info(configs)

    tokenizer = ParaformerTokenizer(
        symbol_table=configs['tokenizer_conf']['symbol_table_path'],
        seg_dict=configs['tokenizer_conf']['seg_dict_path'])
    assert tokenizer.symbol_table['<blank>'] == configs['ctc_conf']['ctc_blank_id']

    return model, configs, tokenizer