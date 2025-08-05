from typing import List, Optional
import yaml
import os
import time
import math

import torch
import torch.distributed as dist
import numpy as np  # NEW

import pydantic
from omegaconf import OmegaConf
from pretrain import PretrainConfig, init_train_state, evaluate, create_dataloader

# optional progress bars
try:
    from tqdm import tqdm
except Exception:  # tqdm not installed
    tqdm = None


class EvalConfig(pydantic.BaseModel):
    checkpoint: str
    data_path: Optional[str] = None

    use_tqdm: bool = True
    log_every: int = 50
    log_rank0_only: bool = True

    save_outputs: List[str] = [
        "inputs",
        "labels",
        "puzzle_identifiers",
        "logits",
        "q_halt_logits",
        "q_continue_logits",
    ]


def _print_rank(msg: str, rank: int, rank0_only: bool = True) -> None:
    if (not rank0_only) or rank == 0:
        print(msg, flush=True)


def _load_checkpoint_flex(model: torch.nn.Module, ckpt_path: str, rank: int = 0, rank0_only: bool = True) -> None:
    sd = torch.load(ckpt_path, map_location="cuda")

    model_sd = model.state_dict()
    has_prefix = any(k.startswith("_orig_mod.") for k in model_sd.keys())
    prefix = "_orig_mod." if has_prefix else ""

    remapped = {}
    for k, v in sd.items():
        if has_prefix and not k.startswith("_orig_mod."):
            k2 = prefix + k
        elif (not has_prefix) and k.startswith("_orig_mod."):
            k2 = k.removeprefix("_orig_mod.")
        else:
            k2 = k
        remapped[k2] = v

    pem_key = f"{prefix}model.inner.puzzle_emb.weights"
    if pem_key in remapped and pem_key in model_sd:
        w_ckpt = remapped[pem_key]
        w_cur = model_sd[pem_key]
        if w_ckpt.shape != w_cur.shape:
            if w_ckpt.shape[0] < w_cur.shape[0]:
                pad_rows = w_cur.shape[0] - w_ckpt.shape[0]
                pad = w_cur.new_empty(pad_rows, w_cur.shape[1])
                torch.nn.init.normal_(pad, mean=0.0, std=0.02)
                remapped[pem_key] = torch.cat([w_ckpt, pad], dim=0)
                _print_rank(f"[checkpoint] padded puzzle_emb from {tuple(w_ckpt.shape)} to {tuple(remapped[pem_key].shape)}", rank, rank0_only)
            else:
                remapped[pem_key] = w_ckpt[: w_cur.shape[0], :]
                _print_rank(f"[checkpoint] sliced puzzle_emb from {tuple(w_ckpt.shape)} to {tuple(remapped[pem_key].shape)}", rank, rank0_only)

    incompat = model.load_state_dict(remapped, strict=False, assign=True)
    if hasattr(incompat, "missing_keys") and hasattr(incompat, "unexpected_keys"):
        mk = incompat.missing_keys
        uk = incompat.unexpected_keys
        if mk:
            _print_rank(f"[checkpoint] missing keys (first 8): {mk[:8]}{' ...' if len(mk)>8 else ''}", rank, rank0_only)
        if uk:
            _print_rank(f"[checkpoint] unexpected keys (first 8): {uk[:8]}{' ...' if len(uk)>8 else ''}", rank, rank0_only)
    _print_rank("[checkpoint] loaded with relaxed matching.", rank, rank0_only)


def _estimate_eval_steps(data_path: str, global_batch_size: int) -> Optional[int]:
    """NEW: estimate number of eval steps from the NPY header (cheap via mmap)."""
    npy = os.path.join(data_path, "test", "all__inputs.npy")
    try:
        arr = np.load(npy, mmap_mode="r")
        n_examples = int(arr.shape[0])
        # Steps are roughly total_examples / global_batch_size (DDP synchronizes steps across ranks)
        steps = math.ceil(n_examples / max(1, global_batch_size))
        return steps
    except Exception:
        return None


class LoggingDataLoader:
    def __init__(
        self,
        loader,
        name: str,
        rank: int,
        rank0_only: bool = True,
        use_tqdm: bool = True,
        every: int = 50,
        estimated_total: Optional[int] = None,  # NEW
    ):
        self._loader = loader
        self._name = name
        self._rank = rank
        self._rank0_only = rank0_only
        self._use_tqdm = use_tqdm and (tqdm is not None)
        self._every = max(1, int(every))
        self._estimated_total = estimated_total  # NEW
        try:
            self._len = len(loader)
        except Exception:
            self._len = None

    def __len__(self):
        if self._len is not None:
            return self._len
        return self._estimated_total if self._estimated_total is not None else 0  # NEW

    def __getattr__(self, name):
        return getattr(self._loader, name)

    def __iter__(self):
        should_log = (not self._rank0_only) or (self._rank == 0)
        if not should_log:
            for batch in self._loader:
                yield batch
            return

        total = self._len if self._len is not None else self._estimated_total  # NEW
        if self._use_tqdm:
            bar = tqdm(total=total, desc=f"{self._name}[rank{self._rank}]", leave=True)
            for batch in self._loader:
                bar.update(1)
                yield batch
            bar.close()
        else:
            start = time.time()
            for i, batch in enumerate(self._loader, start=1):
                if (i % self._every == 0) or (total is not None and i == total):
                    elapsed = time.time() - start
                    if total is not None:
                        pct = 100.0 * i / max(1, total)
                        _print_rank(f"[{self._name} rank{self._rank}] {i}/{total} ({pct:.1f}%) elapsed {elapsed:.1f}s", self._rank, self._rank0_only)
                    else:
                        _print_rank(f"[{self._name} rank{self._rank}] {i} batches elapsed {elapsed:.1f}s", self._rank, self._rank0_only)
                yield batch


def launch():
    eval_cfg = EvalConfig(**OmegaConf.to_container(OmegaConf.from_cli()))  # type: ignore

    RANK = 0
    WORLD_SIZE = 1
    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        RANK = dist.get_rank()
        WORLD_SIZE = dist.get_world_size()
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    # Load training config from checkpoint
    with open(os.path.join(os.path.dirname(eval_cfg.checkpoint), "all_config.yaml"), "r") as f:
        config = PretrainConfig(**yaml.safe_load(f))
        config.eval_save_outputs = eval_cfg.save_outputs
        config.checkpoint_path = os.path.dirname(eval_cfg.checkpoint)

    if eval_cfg.data_path:
        _print_rank(f"[config] overriding data_path -> {eval_cfg.data_path}", RANK, eval_cfg.log_rank0_only)
        config.data_path = eval_cfg.data_path

    # Dataloaders
    train_loader, train_metadata = create_dataloader(
        config, "train", test_set_mode=False, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE,
    )
    eval_loader, eval_metadata = create_dataloader(
        config, "test", test_set_mode=True, epochs_per_iter=1,
        global_batch_size=config.global_batch_size, rank=RANK, world_size=WORLD_SIZE,
    )

    # NEW: estimate total eval steps (for nicer tqdm display)
    est_total_steps = _estimate_eval_steps(config.data_path, config.global_batch_size)

    # Wrap eval loader with progress logging
    eval_loader = LoggingDataLoader(
        eval_loader,
        name="eval",
        rank=RANK,
        rank0_only=eval_cfg.log_rank0_only,
        use_tqdm=eval_cfg.use_tqdm,
        every=eval_cfg.log_every,
        estimated_total=est_total_steps,  # NEW
    )

    # Model
    train_state = init_train_state(config, train_metadata, world_size=WORLD_SIZE)
    _load_checkpoint_flex(train_state.model, eval_cfg.checkpoint, rank=RANK, rank0_only=eval_cfg.log_rank0_only)

    # Step from filename, if present
    train_state.step = 0
    ckpt_filename = os.path.basename(eval_cfg.checkpoint)
    if ckpt_filename.startswith("step_"):
        try:
            train_state.step = int(ckpt_filename.removeprefix("step_"))
        except Exception:
            pass

    _print_rank("Starting evaluation", RANK, eval_cfg.log_rank0_only)
    train_state.model.eval()
    metrics = evaluate(config, train_state, eval_loader, eval_metadata, rank=RANK, world_size=WORLD_SIZE)

    if metrics is not None:
        _print_rank(str(metrics), RANK, eval_cfg.log_rank0_only)


if __name__ == "__main__":
    launch()
