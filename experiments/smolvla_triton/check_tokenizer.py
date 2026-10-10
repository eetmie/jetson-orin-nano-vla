"""Compare the lean path with the installed Transformers tokenizer, offline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.vendor.smolvla_trt import Bundle
from runtime_memory import PROMPT_CACHE_LIMIT, install_lean_tokenizer

parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    raise FileExistsError(args.out)
root = Path.home()/'bundles/smolvla-base-split'
reference = Bundle(root)
prompts = ['', '\n', ' ', 'pick up the blue block', 'place the red block in the tray',
           'pick up the blue block\n', '  move\t the\n cup  ',
           'siirrä kuppi pöydälle', '把红色方块放入托盘', '🤖 ➡️ 🧊',
           '<|im_start|>user<|im_end|><image><end_of_utterance>',
           'word '*47, 'word '*48, 'word '*49, 'word '*1000]
rng = np.random.default_rng(20261010)
alphabet = list('abc XYZ123\t\n!?.äé你🤖')
for length in [0, 1, 5, 40, 47, 48, 49, 100, 300, 1000]:
    for _ in range(20):
        prompts.append(''.join(rng.choice(alphabet, size=length)))
expected = [reference.tokens(task) for task in prompts]
install_lean_tokenizer()
candidate = Bundle(root)
rows = []
for task, (ids, mask) in zip(prompts, expected):
    actual_ids, actual_mask = candidate.tokens(task)
    if not np.array_equal(ids, actual_ids) or not np.array_equal(mask, actual_mask):
        raise ValueError(f'tokenizer mismatch: {task!r}')
    if not np.array_equal(reference.embed_ids(ids), candidate.embed_ids(actual_ids)):
        raise ValueError('embedding mismatch')
    embedding, cached_mask = candidate.language(task)
    if not np.array_equal(embedding, reference.embed_ids(ids)) or not np.array_equal(cached_mask, mask):
        raise ValueError('cached language mismatch')
    if len(candidate._lang) > PROMPT_CACHE_LIMIT:
        raise ValueError('prompt cache exceeds its memory bound')
    rows.append(dict(prompt=task, token_ids=ids.tolist(), attention_mask=mask.tolist()))
first_ids, first_mask = expected[0]
if prompts[0] in candidate._lang:
    raise ValueError('old prompt was not evicted')
embedding, mask = candidate.language(prompts[0])
if not np.array_equal(embedding, reference.embed_ids(first_ids)) or not np.array_equal(mask, first_mask):
    raise ValueError('evicted prompt changed on reuse')
with np.load(root/'fixture.npz') as fixture:
    # The native policy fixture separately checks its recorded task token IDs.
    fixture_shape = list(fixture['lang_tokens'].shape)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(dict(status='PASS', cases=len(rows), seed=20261010,
    token_ids_bit_identical=True, masks_bit_identical=True, embeddings_bit_identical=True,
    prompt_cache_limit=PROMPT_CACHE_LIMIT, evicted_prompt_reuse_bit_identical=True,
    tokenizer_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (root/'tokenizer').glob('*.json')}, fixture_tokens_shape=fixture_shape,
    checks=rows), indent=2, ensure_ascii=False))
print('PASS', len(rows), 'prompts; exact token IDs, masks and embedding rows')
