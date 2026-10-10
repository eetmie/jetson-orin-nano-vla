"""Optional runtime memory changes for the measured SmolVLA bundle."""
import ctypes
import json

import numpy as np

PROMPT_CACHE_LIMIT = 8


def install_lean_tokenizer():
    """Use the saved Rust backend directly, without importing Transformers.

    This path is restricted to the exported GPT2 tokenizer tested in this
    experiment. Its postprocessor/special tokens already live in tokenizer.json.
    Padding is right-sided and truncation follows the saved configuration.
    """
    from bench.vendor.smolvla_trt import Bundle

    def tokens(self, task):
        if self._tokenizer is None:
            from tokenizers import Tokenizer
            root = self.root/'tokenizer'
            config = json.loads((root/'tokenizer_config.json').read_text())
            if config.get('tokenizer_class') != 'GPT2Tokenizer':
                raise ValueError('lean tokenizer supports only the validated GPT2 export')
            tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
            pad = config.get('pad_token')
            if isinstance(pad, dict):
                pad = pad.get('content')
            pad_id = tokenizer.token_to_id(pad) if isinstance(pad, str) else None
            if pad_id is None:
                raise ValueError('saved padding token is absent from tokenizer.json')
            tokenizer.enable_truncation(self.lang_len, direction=config.get('truncation_side', 'right'))
            tokenizer.enable_padding(direction='right', pad_id=pad_id, pad_token=pad, length=self.lang_len)
            self._tokenizer = tokenizer
        task = task if task.endswith('\n') else task+'\n'
        encoded = self._tokenizer.encode(task, add_special_tokens=True)
        ids = np.asarray(encoded.ids, np.int64)[None]
        mask = np.asarray(encoded.attention_mask, bool)[None]
        if ids.shape != (1, self.lang_len) or mask.shape != ids.shape:
            raise ValueError('lean tokenizer produced an invalid fixed-length contract')
        return ids, mask

    Bundle.tokens = tokens

    def language(self, task):
        if task in self._lang:
            value = self._lang.pop(task)
        else:
            ids, mask = self.tokens(task)
            value = self.embed_ids(ids), mask
        self._lang[task] = value
        if len(self._lang) > PROMPT_CACHE_LIMIT:
            self._lang.pop(next(iter(self._lang)))
        return value

    Bundle.language = language


def trim_host_heap():
    """Release freed glibc pages once outside CUDA graph capture/inference."""
    libc = ctypes.CDLL(None)
    trim = libc.malloc_trim
    trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
    return bool(trim(0))
