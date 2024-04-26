import re
from typing import Dict, List, Optional, Union, Tuple

from os import PathLike
from .search import paraformer_beautify_result
from ..utils.tokenize_utils import tokenize_by_seg_dict

from ..utils.file_utils import read_symbol_table, read_non_lang_symbols

T = Union[str, bytes]


def read_seg_dict(path):
    seg_table = {}
    with open(path, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split('\t')
            assert len(arr) == 2
            seg_table[arr[0]] = arr[1]
    return seg_table


class CharTokenizer:

    def __init__(
        self,
        symbol_table: Union[str, PathLike, Dict],
        non_lang_syms: Optional[Union[str, PathLike, List]] = None,
        split_with_space: bool = False,
        connect_symbol: str = '',
        unk='<unk>',
    ) -> None:
        self.non_lang_syms_pattern = None
        if non_lang_syms is not None:
            self.non_lang_syms_pattern = re.compile(
                r"(\[[^\[\]]+\]|<[^<>]+>|{[^{}]+})")
        if not isinstance(symbol_table, Dict):
            self._symbol_table = read_symbol_table(symbol_table)
        else:
            # symbol_table = {"我": 1, "是": 2, "{NOISE}": 3}
            self._symbol_table = symbol_table
        if not isinstance(non_lang_syms, List):
            self.non_lang_syms = read_non_lang_symbols(non_lang_syms)
        else:
            # non_lang_syms=["{NOISE}"]
            self.non_lang_syms = non_lang_syms
        self.char_dict = {v: k for k, v in self._symbol_table.items()}
        self.split_with_space = split_with_space
        self.connect_symbol = connect_symbol
        self.unk = unk

    def tokenize(self, line: str) -> Tuple[List[T], List[int]]:
        tokens = self.text2tokens(line)
        ids = self.tokens2ids(tokens)
        return tokens, ids

    def detokenize(self, ids: List[int]) -> Tuple[str, List[T]]:
        tokens = self.ids2tokens(ids)
        text = self.tokens2text(tokens)
        return text, tokens

    def text2tokens(self, line: str) -> List[str]:
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line.upper())
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                if self.split_with_space:
                    part = part.split(" ")
                for ch in part:
                    if ch == ' ':
                        ch = "▁"
                    tokens.append(ch)
        return tokens

    def tokens2text(self, tokens: List[str]) -> str:
        return self.connect_symbol.join(tokens)

    def tokens2ids(self, tokens: List[str]) -> List[int]:
        ids = []
        for ch in tokens:
            if ch in self._symbol_table:
                ids.append(self._symbol_table[ch])
            elif self.unk in self._symbol_table:
                ids.append(self._symbol_table[self.unk])
        return ids

    def ids2tokens(self, ids: List[int]) -> List[str]:
        content = [self.char_dict[w] for w in ids]
        return content

    def vocab_size(self) -> int:
        return len(self.char_dict)

    @property
    def symbol_table(self) -> Dict[str, int]:
        return self._symbol_table


class ParaformerTokenizer(CharTokenizer):

    def __init__(self,
                 symbol_table: Union[str, PathLike, Dict],
                 seg_dict: Optional[Union[str, PathLike, Dict]] = None,
                 split_with_space: bool = False,
                 connect_symbol: str = '',
                 unk='<unk>') -> None:
        super().__init__(symbol_table, None, split_with_space, connect_symbol,
                         unk)
        self.seg_dict = seg_dict
        if seg_dict is not None and not isinstance(seg_dict, Dict):
            self.seg_dict = read_seg_dict(seg_dict)

    def text2tokens(self, line: str) -> List[str]:
        assert self.seg_dict is not None

        # TODO(Mddct): duplicated here, refine later
        line = line.strip()
        if self.non_lang_syms_pattern is not None:
            parts = self.non_lang_syms_pattern.split(line)
            parts = [w for w in parts if len(w.strip()) > 0]
        else:
            parts = [line]

        tokens = []
        for part in parts:
            if part in self.non_lang_syms:
                tokens.append(part)
            else:
                tokens.extend(tokenize_by_seg_dict(self.seg_dict, part))
        return tokens

    def tokens2text(self, tokens: List[str]) -> str:
        return paraformer_beautify_result(tokens)