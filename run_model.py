import argparse
import json
import numpy as np
from core_transformer import TransformerConfig, EinsumTransformer

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--tokens", type=str, default=None)

args = parser.parse_args()

with open(args.config) as f:
    cfg_dict = json.load(f)

cfg = TransformerConfig(cfg_dict)
model = EinsumTransformer(cfg)

if args.tokens:
    tokens = np.load(args.tokens)
else:
    tokens = np.random.randint(0, cfg.vocab_size, size=(cfg.batch_size, cfg.seq_len))

# Start the full generation loop (Prefill + Iterative Decode)
generated_tokens = model.generate(tokens)

