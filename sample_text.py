# sample_text.py
import os, json, yaml, math, torch
import torch.nn.functional as F
from utils.tokenizer_char import CharTokenizer
from pretrain_text import PretrainConfig, init_train_state, create_dataloader

@torch.no_grad()
def main(ckpt_dir: str, data_path: str | None, prompt: str, max_new_tokens: int = 200,
         temperature: float = 1.0, top_k: int | None = None, device: str = "cuda"):
    # load cfg + tokenizer
    with open(os.path.join(ckpt_dir, "all_config.yaml"), "r") as f:
        cfg = PretrainConfig(**yaml.safe_load(f))
    if data_path:
        cfg.data_path = data_path

    with open(os.path.join(cfg.data_path, "vocab.json"), "r") as f:
        tok = CharTokenizer.from_json(json.load(f))

    # init model
    rank, world_size = 0, 1
    _, train_meta = create_dataloader(cfg, "train", test_set_mode=False, epochs_per_iter=1,
                                      global_batch_size=cfg.global_batch_size, rank=rank, world_size=world_size)
    train_state = init_train_state(cfg, train_meta, world_size=world_size)

    # load weights
    ckpts = sorted([p for p in os.listdir(ckpt_dir) if p.startswith("step_")])
    assert ckpts, f"No checkpoints in {ckpt_dir}"
    sd = torch.load(os.path.join(ckpt_dir, ckpts[-1]), map_location=device)
    try:
        train_state.model.load_state_dict(sd, strict=False, assign=True)
    except:
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()},
                                          strict=False, assign=True)
    train_state.model.eval().to(device)

    # seed with prompt
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.int32, device=device)  # [1, L]
    for _ in range(max_new_tokens):
        # clamp length to model seq len
        x = ids[:, -train_meta.seq_len:].contiguous()

        # build a fake batch with labels=PAD so CE ignored
        batch = {
            "inputs": x,
            "labels": torch.full_like(x, fill_value=train_meta.pad_id),
            "puzzle_identifiers": torch.ones((1,), dtype=torch.int32, device=device)
        }
        with torch.device(device):
            carry = train_state.model.initial_carry(batch)  # type: ignore
        carry, outputs = train_state.model.model(carry.inner_carry, batch)  # type: ignore[attr-defined]
        logits = outputs["logits"]  # [1, T, V]

        last = logits[:, -1, :] / max(1e-6, temperature)
        if top_k is not None and top_k < last.shape[-1]:
            v, ix = torch.topk(last, top_k)
            last = torch.full_like(last, -float("inf")).scatter(1, ix, v)
        probs = F.softmax(last.float(), dim=-1)
        nxt = torch.multinomial(probs, num_samples=1)  # [1,1]

        ids = torch.cat([ids, nxt.int()], dim=1)

    print(tok.decode(ids[0].tolist()))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--data-path", default=None)
    ap.add_argument("--prompt", default="Once upon a time ")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-k", type=int, default=None)
    args = ap.parse_args()
    main(args.ckpt_dir, args.data_path, args.prompt, args.max_new_tokens, args.temperature, args.top_k)
