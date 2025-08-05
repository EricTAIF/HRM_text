# evaluate_text.py
import os, math, yaml, torch
from torch.nn import functional as F
from pretrain_text import PretrainConfig, init_train_state, create_dataloader

@torch.no_grad()
def main(ckpt_dir: str, data_path: str | None = None, device: str = "cuda"):
    # load config written during training
    with open(os.path.join(ckpt_dir, "all_config.yaml"), "r") as f:
        cfg = PretrainConfig(**yaml.safe_load(f))

    if data_path:
        cfg.data_path = data_path

    # single-GPU eval
    rank, world_size = 0, 1
    _, train_meta = create_dataloader(cfg, "train", test_set_mode=False, epochs_per_iter=1,
                                      global_batch_size=cfg.global_batch_size, rank=rank, world_size=world_size)
    train_state = init_train_state(cfg, train_meta, world_size=world_size)

    # pick latest `step_*` file
    ckpts = sorted([p for p in os.listdir(ckpt_dir) if p.startswith("step_")])
    assert ckpts, f"No checkpoints in {ckpt_dir}"
    ckpt_path = os.path.join(ckpt_dir, ckpts[-1])

    # load state dict (works for compile/_orig_mod)
    sd = torch.load(ckpt_path, map_location=device)
    try:
        train_state.model.load_state_dict(sd, strict=False, assign=True)
    except:  # unwrap _orig_mod
        train_state.model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in sd.items()},
                                          strict=False, assign=True)
    train_state.model.eval().to(device)

    # dataloader for test
    eval_loader, eval_meta = create_dataloader(cfg, "test", test_set_mode=True, epochs_per_iter=1,
                                               global_batch_size=cfg.global_batch_size, rank=rank, world_size=world_size)

    total_nats = 0.0
    total_tokens = 0
    for _set_name, batch, _gbs in eval_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.device(device):
            carry = train_state.model.initial_carry(batch)  # type: ignore

        # single step (halt_max_steps=1 in text runs)
        #carry, outputs = train_state.model.model(carry.inner_carry, batch)  # type: ignore[attr-defined]
        #carry, outputs = train_state.model(carry=carry, batch=batch, return_keys=[])
        carry, _, _, outputs, _ = train_state.model(carry=carry, batch=batch, return_keys=["logits"])


        logits = outputs["logits"]  # [B, T, V]
        labels = batch["labels"]    # [B, T]

        # average CE in nats per token
        IGNORE_LABEL_ID = -100  # matches training pipeline
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]).float(),
            labels.reshape(-1).long(),
            ignore_index=IGNORE_LABEL_ID,
            reduction="sum"
        ).item()


        valid = (labels != eval_meta.pad_id).sum().item()
        total_nats += loss
        total_tokens += valid

    avg_nats = total_nats / max(1, total_tokens)
    ppl = math.exp(avg_nats)
    bpc = avg_nats / math.log(2.0)
    print(f"Tokens: {total_tokens:,}")
    print(f"Avg CE (nats/token): {avg_nats:.4f}")
    print(f"Perplexity: {ppl:.3f}")
    print(f"Bits per char: {bpc:.3f}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True, help="folder with step_* and all_config.yaml")
    ap.add_argument("--data-path", default=None)
    args = ap.parse_args()
    main(args.ckpt_dir, args.data_path)
