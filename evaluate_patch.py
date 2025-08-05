from typing import List, Optional
import yaml
import os

import torch
import torch.distributed as dist

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    # Optional: override dataset path used by the checkpoint's all_config.yaml
    data_path: Optional[str] = None

    save_outputs: List[str] = [
        "inputs",
        "labels",
        "puzzle_identifiers",
        "logits",
        "q_halt_logits",
        "q_continue_logits",
    ]


def _load_checkpoint_flex(model: torch.nn.Module, ckpt_path: str) -> None:
    """
    Load a checkpoint robustly:
      - Handles torch.compile wrappers (_orig_mod.)
      - Adapts puzzle_emb.weights rows to match current dataset size by pad/slice
      - Loads with strict=False to tolerate harmless name diffs
    """
    sd = torch.load(ckpt_path, map_location="cuda")

    model_sd = model.state_dict()
    has_prefix = any(k.startswith("_orig_mod.") for k in model_sd.keys())
    prefix = "_orig_mod." if has_prefix else ""

    # Remap keys to match the model's prefixing
    remapped = {}
    for k, v in sd.items():
        if has_prefix and not k.startswith("_orig_mod."):
            k2 = prefix + k
        elif (not has_prefix) and k.startswith("_orig_mod."):
            k2 = k.removeprefix("_orig_mod.")
        else:
            k2 = k
        remapped[k2] = v

    # Fix puzzle_emb row count mismatch by pad/slice
    pem_key = f"{prefix}model.inner.puzzle_emb.weights"
    if pem_key in remapped and pem_key in model_sd:
        w_ckpt = remapped[pem_key]
        w_cur = model_sd[pem_key]
        if w_ckpt.shape != w_cur.shape:
            if w_ckpt.shape[0] < w_cur.shape[0]:
                # pad extra rows
                pad_rows = w_cur.shape[0] - w_ckpt.shape[0]
                pad = w_cur.new_empty(pad_rows, w_cur.shape[1])
                torch.nn.init.normal_(pad, mean=0.0, std=0.02)
                remapped[pem_key] = torch.cat([w_ckpt, pad], dim=0)
                print(
                    f"[checkpoint] padded puzzle_emb from {tuple(w_ckpt.shape)} to {tuple(remapped[pem_key].shape)}"
                )
            else:
                # slice extra rows
                remapped[pem_key] = w_ckpt[: w_cur.shape[0], :]
                print(
                    f"[checkpoint] sliced puzzle_emb from {tuple(w_ckpt.shape)} to {tuple(remapped[pem_key].shape)}"
                )

    # Load with relaxed matching
    incompat = model.load_state_dict(remapped, strict=False, assign=True)
    # In PyTorch this returns IncompatibleKeys(missing_keys, unexpected_keys)
    if hasattr(incompat, "missing_keys") and hasattr(incompat, "unexpected_keys"):
        mk = incompat.missing_keys
        uk = incompat.unexpected_keys
        if mk:
            print(f"[checkpoint] missing keys (first 8): {mk[:8]}{' ...' if len(mk)>8 else ''}")
        if uk:
            print(f"[checkpoint] unexpected keys (first 8): {uk[:8]}{' ...' if len(uk)>8 else ''}")
    print("[checkpoint] loaded with relaxed matching.")


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore

    RANK = 0
    WORLD_SIZE = 1
    # Initialize distributed if using torchrun
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load training config bundled with the checkpoint
    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    # Optional: override dataset path (useful when your local path differs)
    if eval_cfg.data_path:
        print(f"[config] overriding data_path -> {eval_cfg.data_path}")
        config.data_path = eval_cfg.data_path

    # Dataloaders
    train_loader, train_metadata = create_dataloader(
        config,
        "train",
        test_set_mode=False,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata = create_dataloader(
        config,
        "test",
        test_set_mode=True,
        epochs_per_iter=1,
        global_batch_size=config.global_batch_size,
        rank=RANK,
        world_size=WORLD_SIZE,
    )

    # Model
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)

    # Robust checkpoint loading (handles _orig_mod and puzzle_emb size)
    _load_checkpoint_flex(train_state.model, eval_cfg.checkpoint)

    # Step from filename if available
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        try:
            train_state.step = int(ckpt_filename.removeprefix("step_"))
        except Exception:
            pass

    # Evaluate
    print("Starting evaluation")
    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if metrics is not None:
        print(metrics)


if __name__ == "__main__":
    launch()
