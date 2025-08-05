# evaluate_holdout.py
import os, math, json, argparse, torch
import torch.nn.functional as F
from typing import List, Tuple

from pretrain_text import PretrainConfig, init_train_state, create_dataloader
from utils.tokenizer_char import CharTokenizer, PAD_ID

@torch.no_grad()
def _make_sequences(token_ids, seq_len: int, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    inputs  = tokens[t : t+seq_len]
    labels  = tokens[t+1 : t+seq_len+1]
    Drop last window if shorter than seq_len+1.
    """
    import numpy as np
    token_ids = np.asarray(token_ids, dtype=np.int32)
    T = token_ids.shape[0]
    window = seq_len + 1
    if T < window:
        return torch.empty(0, seq_len, dtype=torch.int32), torch.empty(0, seq_len, dtype=torch.int32)

    xs, ys = [], []
    for start in range(0, T - window + 1, stride):
        s = token_ids[start : start + window]
        xs.append(s[:-1].astype(np.int32))
        ys.append(s[1: ].astype(np.int32))
    return torch.from_numpy(np.stack(xs, 0)), torch.from_numpy(np.stack(ys, 0))

def main(ckpt_dir: str, input_file: str, vocab_json: str | None, data_path: str | None, device: str = "cuda",
         stride: int | None = None, batch_size: int | None = None):

    # ---- load training config (defines model arch & defaults) ----
    import yaml
    with open(os.path.join(ckpt_dir, "all_config.yaml"), "r") as f:
        cfg = PretrainConfig(**yaml.safe_load(f))

    if data_path:
        cfg.data_path = data_path

    # single-GPU eval
    rank, world_size = 0, 1

    # we only need metadata (seq_len, pad id, etc.); build loaders quickly
    _, train_meta = create_dataloader(cfg, "train", test_set_mode=False, epochs_per_iter=1,
                                      global_batch_size=cfg.global_batch_size, rank=rank, world_size=world_size)
    seq_len = train_meta.seq_len
    pad_id  = train_meta.pad_id

    # instantiate model & load latest checkpoint
    train_state = init_train_state(cfg, train_meta, world_size=world_size)
    ckpts = sorted([p for p in os.listdir(ckpt_dir) if p.startswith("step_")])
    assert ckpts, f"No checkpoints in {ckpt_dir}"
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])

    sd = torch.load(ckpt_path, map_location=device)
    try:
        train_state.model.load_state_dict(sd, strict=False, assign=True)
    except Exception:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()},
                                          strict=False, assign=True)
    train_state.model.eval().to(device)

    # ---- tokenizer ----
    if vocab_json is None:
        vocab_json = os.path.join(cfg.data_path, "vocab.json")
    tok = CharTokenizer.load(vocab_json)

    # ---- read & encode file ----
    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    ids = tok.encode(text)

    # ---- build non-overlapping windows (or custom stride) ----
    if stride is None:
        stride = seq_len
    x, y = _make_sequences(ids, seq_len=seq_len, stride=stride)
    assert x.numel() > 0, f"File too short for seq_len={seq_len}"

    # ---- batching ----
    if batch_size is None:
        batch_size = min(cfg.global_batch_size, 2048)
    total_nats = 0.0
    total_tokens = 0

    for start in range(0, x.shape[0], batch_size):
        xb = x[start:start+batch_size].to(device)
        yb = y[start:start+batch_size].to(device)
        # model expects a batch dict like the dataset
        batch = {
            "inputs": xb,
            "labels": yb,
            # model ignores this when puzzle_emb_ndim=0 but it's required by the signature
            "puzzle_identifiers": torch.ones(xb.shape[0], dtype=torch.int32, device=xb.device),
        }

        with torch.device(device):
            carry = train_state.model.initial_carry(batch)  # type: ignore

        # Ask the loss head to give us logits (single step when halt_max_steps=1)
        carry, _loss, _metrics, outputs, _done = train_state.model(
            carry=carry, batch=batch, return_keys=["logits"]
        )
        logits = outputs["logits"].float()    # [B, T, V]
        labels = yb.long()                    # [B, T]

        # CE sum over valid (ignore pad)
        ce_sum = F.cross_entropy(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
            ignore_index=pad_id,
            reduction="sum",
        ).item()
        valid = (labels != pad_id).sum().item()
        total_nats += ce_sum
        total_tokens += valid

    avg_nats = total_nats / max(1, total_tokens)
    ppl = math.exp(avg_nats)
    bpc = avg_nats / math.log(2.0)

    print(f"File: {input_file}")
    print(f"Tokens: {total_tokens:,}")
    print(f"Avg CE (nats/token): {avg_nats:.4f}")
    print(f"Perplexity: {ppl:.3f}")
    print(f"Bits per char: {bpc:.3f}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="folder with step_* and all_config.yaml")
    ap.add_argument("--input-file", required=True, help="path to a .txt file to evaluate")
    ap.add_argument("--vocab", default=None, help="path to vocab.json (defaults to <data_path>/vocab.json)")
    ap.add_argument("--data-path", default=None, help="override data path used in config")
    ap.add_argument("--stride", type=int, default=None, help="window stride; default = seq_len (no overlap)")
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()
    main(args.ckpt_dir, args.input_file, args.vocab, args.data_path, stride=args.stride, batch_size=args.batch_size)
