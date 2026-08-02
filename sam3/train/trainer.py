# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import contextlib
import fnmatch
import gc
import json
import logging
import math
import os
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np

import torch
import torch.distributed as dist
import torch.nn as nn
from hydra.utils import instantiate
from iopath.common.file_io import g_pathmgr

from sam3.model.data_misc import BatchedDatapoint
from sam3.model.model_misc import SAM3Output
from sam3.model.utils.misc import copy_data_to_device

from sam3.train.optim.optimizer import construct_optimizer

from sam3.train.utils.checkpoint_utils import (
    assert_skipped_parameters_are_frozen,
    exclude_params_matching_unix_pattern,
    load_state_dict_into_model,
    with_check_parameter_frozen,
)

from sam3.train.utils.distributed import all_reduce_max, barrier, get_rank

from sam3.train.utils.logger import Logger, setup_logging
from sam3.train.utils.train_utils import (
    AverageMeter,
    collect_dict_keys,
    DurationMeter,
    get_amp_type,
    get_machine_local_and_dist_rank,
    get_resume_checkpoint,
    human_readable_time,
    is_dist_avail_and_initialized,
    log_env_variables,
    makedir,
    MemMeter,
    Phase,
    ProgressMeter,
    set_seeds,
    setup_distributed_backend,
)


CORE_LOSS_KEY = "core_loss"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def unwrap_ddp_if_wrapped(model):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        return model.module
    return model


@dataclass
class OptimAMPConf:
    enabled: bool = False
    amp_dtype: str = "float16"
    # (Ha) Added 2026-07-18: see the matching config comment in
    # full_train_config.yaml (optim.amp.bypass_dynamic_scale_for_bf16) and
    # .github/skills/loss-debugging/SKILL.md Step 9/10 for the full
    # investigation. Defaults True: pins GradScaler's scale at a constant 1.0
    # for bfloat16 instead of letting it dynamically decay/grow, since
    # bfloat16 never needed loss scaling and the decay was confirmed to
    # corrupt weights (scale reached 2.94e-39, a subnormal fp32 value, right
    # when backbone.vision_backbone.trunk.patch_embed.proj went 100% NaN).
    bypass_dynamic_scale_for_bf16: bool = True


@dataclass
class OptimConf:
    optimizer: torch.optim.Optimizer = None
    options: Optional[Dict[str, Any]] = None
    param_group_modifiers: Optional[List] = None
    amp: Optional[Dict[str, Any]] = None
    gradient_clip: Any = None
    gradient_logger: Any = None
    # (Ha) Added 2026-07-20: toggle between fine-tuning the FULL model (every
    # param group in `options.lr` below stays trainable -- the existing/
    # default behavior) and fine-tuning ONLY the decoder/segmentation head
    # (freezes everything else via requires_grad=False -- see
    # Trainer._maybe_freeze_non_decoder_params()). Frozen params are NOT
    # removed from their optimizer param group, so validate_param_group_params'
    # "must cover every model parameter" check still passes -- they just never
    # receive a gradient, so AdamW.step() skips them (PyTorch optimizers skip
    # any param whose .grad is None). Set to False for quick decoder-only
    # iteration; set back to True (the default) for full fine-tuning.
    train_full_model: bool = True
    # Unix-style fnmatch patterns identifying which params stay trainable when
    # train_full_model=False. Defaults to the same patterns already used for
    # the transformer-decoder LR group in `options.lr` (see config comment).
    decoder_only_trainable_patterns: Optional[List[str]] = None

    def __post_init__(self):
        # amp
        if not isinstance(self.amp, OptimAMPConf):
            if self.amp is None:
                self.amp = {}
            assert isinstance(self.amp, Mapping)
            self.amp = OptimAMPConf(**self.amp)
        if not self.decoder_only_trainable_patterns:
            self.decoder_only_trainable_patterns = [
                "transformer.decoder.*",
                "dot_prod_scoring.*",
                "segmentation_head.*",
            ]


@dataclass
class DistributedConf:
    backend: Optional[str] = None  # inferred from accelerator type
    comms_dtype: Optional[str] = None
    find_unused_parameters: bool = False
    timeout_mins: int = 30
    gradient_as_bucket_view: bool = False  # PyTorch DDP default is False
    static_graph: bool = False  # PyTorch DDP default is False


@dataclass
class CudaConf:
    cudnn_deterministic: bool = False
    cudnn_benchmark: bool = True
    allow_tf32: bool = False
    # if not None, `matmul_allow_tf32` key will override `allow_tf32` for matmul
    matmul_allow_tf32: Optional[bool] = None
    # if not None, `cudnn_allow_tf32` key will override `allow_tf32` for cudnn
    cudnn_allow_tf32: Optional[bool] = None


@dataclass
class CheckpointConf:
    save_dir: str
    save_freq: int
    save_list: List[int] = field(default_factory=list)
    model_weight_initializer: Any = None
    save_best_meters: List[str] = None
    # (Ha) Added, 2026-07-17: restrict which metric sub-keys under
    # `save_best_meters` actually trigger a full checkpoint save (each save is a
    # full ~10GB model checkpoint written to a network mount). Without this, the
    # first validation of a meter with N sub-metrics (e.g. AP, AP_50, AP_75,
    # AR@1, ...) "improves" on all N simultaneously (no prior best exists yet),
    # writing N nearly-identical checkpoints sequentially — several minutes of
    # near-zero GPU utilization that tripped this HPC's idle-GPU auto-kill
    # policy (TERM_OWNER / exit 143) during a real repro run. If None (default),
    # preserves the old behavior (save for every improving sub-key).
    save_best_meter_subkeys: Optional[List[str]] = None
    skip_saving_parameters: List[str] = field(default_factory=list)
    initialize_after_preemption: Optional[bool] = None
    # if not None, training will be resumed from this checkpoint
    resume_from: Optional[str] = None

    def infer_missing(self):
        if self.initialize_after_preemption is None:
            with_skip_saving = len(self.skip_saving_parameters) > 0
            self.initialize_after_preemption = with_skip_saving
        return self


@dataclass
class LoggingConf:
    log_dir: str
    log_freq: int  # In iterations
    tensorboard_writer: Any
    log_level_primary: str = "INFO"
    log_level_secondary: str = "ERROR"
    log_scalar_frequency: int = 100
    log_visual_frequency: int = 100
    scalar_keys_to_log: Optional[Dict[str, Any]] = None
    log_batch_stats: bool = False
    wandb_writer: Optional[Any] = None


def _save_anomalous_batch_jpegs(
    payload: Dict[str, Any],
    jpeg_dir: str,
    mean=(0.5, 0.5, 0.5),
    std=(0.5, 0.5, 0.5),
    stage_id: int = 0,
) -> None:
    """(Ha) Added, 2026-07-18: automatically render an anomalous batch's
    post-augmentation images to JPEG (with box + mask overlays) right at
    dump time, so a JPEG is already on disk to look at without a manual
    follow-up run of `.github/scripts/inspect_anomalous_batch.py`.

    `mean`/`std` must match `scratch.train_norm_mean`/`train_norm_std` in
    full_train_config.yaml (both default to 0.5/0.5/0.5 currently).

    Mirrors `save_post_augmentation_jpegs()` in
    `.github/scripts/inspect_anomalous_batch.py` (kept as a separate,
    dependency-light standalone copy there for offline/CPU-only
    re-inspection of any past dump) -- if one needs a fix, check the other.
    Kept intentionally best-effort / non-fatal: any failure here must never
    break the training loop itself, since this is a debugging aid only.
    """
    from PIL import Image, ImageDraw

    os.makedirs(jpeg_dir, exist_ok=True)
    img_batch = payload["img_batch"]
    find_targets = payload["find_targets"]
    if not find_targets:
        return
    stage_id = stage_id if stage_id < len(find_targets) else 0
    target = find_targets[stage_id]
    boxes = target.get("boxes")
    num_boxes = target.get("num_boxes")
    segments = target.get("segments")

    n = img_batch.shape[0]
    counts = num_boxes.tolist() if num_boxes is not None else [0] * n
    # `counts` has one entry per (image, category) query (e.g. 32 entries for
    # 4 images x category_chunk_size=8), NOT one entry per image -- group
    # every len(counts)/n consecutive entries per image before slicing the
    # flat boxes/segments list (see 2026-07-18 fix in the standalone script).
    per_image_query_count = max(len(counts) // n, 1) if counts else 1
    image_totals = [
        sum(counts[i * per_image_query_count : (i + 1) * per_image_query_count])
        for i in range(n)
    ]

    def _chunk(flat_list, totals):
        out, idx = [], 0
        for c in totals:
            c = int(c)
            out.append(list(flat_list[idx : idx + c]))
            idx += c
        return out

    boxes_per_image = _chunk(list(boxes), image_totals) if boxes is not None else [[] for _ in range(n)]
    segs_per_image = _chunk(segments, image_totals) if segments is not None else None

    mean_t = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    palette = [
        (255, 0, 0, 90), (0, 200, 0, 90), (0, 0, 255, 90),
        (255, 255, 0, 90), (255, 0, 255, 90), (0, 255, 255, 90),
    ]

    for i in range(n):
        img_t = (img_batch[i].float() * std_t + mean_t).clamp(0, 1)
        arr = (img_t.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        img = Image.fromarray(arr)
        width, height = img.size
        draw_img = img.convert("RGBA")

        if segs_per_image is not None:
            overlay = Image.new("RGBA", draw_img.size, (0, 0, 0, 0))
            for k, mask in enumerate(segs_per_image[i]):
                mask_np = mask.numpy().astype(bool)
                if not mask_np.any():
                    continue
                color = palette[k % len(palette)]
                mask_img = Image.fromarray((mask_np * 255).astype("uint8"), mode="L")
                colored = Image.new("RGBA", draw_img.size, color)
                overlay.paste(colored, (0, 0), mask_img)
            draw_img = Image.alpha_composite(draw_img, overlay)

        draw_img = draw_img.convert("RGB")
        ddraw = ImageDraw.Draw(draw_img)
        for box in boxes_per_image[i]:
            box_flat = box.reshape(-1)
            if box_flat.numel() < 4:
                continue
            cx, cy, w, h = (float(v) for v in box_flat[:4])
            x1, y1 = (cx - w / 2) * width, (cy - h / 2) * height
            x2, y2 = (cx + w / 2) * width, (cy + h / 2) * height
            ddraw.rectangle([x1, y1, x2, y2], outline=(255, 255, 0), width=2)

        draw_img.save(os.path.join(jpeg_dir, f"post_image_{i}_stage{stage_id}.jpg"), "JPEG", quality=92)


def _install_nan_debug_hooks(model: nn.Module) -> None:
    """
    (Ha) Added, 2026-07-18: install a forward hook on every named submodule
    that checks the module's OUTPUT for non-finite (NaN/Inf) values, so we
    can pinpoint exactly where in the forward pass NaN first appears --
    instead of only knowing it exists somewhere, which is as far as the
    `model_output_debug` capture in sam3_loss.py / `_maybe_dump_anomalous_batch`
    below could get us (it only sees the FINAL pred_boxes/pred_logits/
    pred_masks, already NaN by the time they get there -- see
    experiments/8cat_bondwire dump train_all_epoch5_step9768_0.pt, confirmed
    2026-07-18: matched query predictions were NaN, not the padding-slot
    theory originally suspected -- the existing nan_to_num loss guards
    (added 2026-07-16/17 to stop the FloatingPointError crash) were silently
    converting these NaN losses to exactly 0.0 instead of crashing, which is
    the real explanation for the "zero-loss collapse").

    Execution order of forward hooks follows the network's own topological
    (call) order, so the FIRST submodule whose hook reports non-finite output
    in a given forward pass is very likely at or immediately downstream of
    the true origin.

    Gated behind the `SAM3_NAN_DEBUG_HOOKS=1` environment variable -- checking
    every submodule's output on every forward pass has real overhead, so this
    must stay off by default and only be turned on for a debugging run.
    """
    if os.environ.get("SAM3_NAN_DEBUG_HOOKS") != "1":
        return

    state = {"logged_this_call": False, "call_count": 0}

    def _check_tensor(name, module_type, t):
        if state["logged_this_call"]:
            return
        if isinstance(t, torch.Tensor) and t.is_floating_point() and not torch.isfinite(t).all():
            state["logged_this_call"] = True
            n_nan = torch.isnan(t).sum().item()
            n_inf = torch.isinf(t).sum().item()
            logging.error(
                "(Ha) NaN-debug: FIRST non-finite output at module=%s type=%s "
                "shape=%s n_nan=%d n_inf=%d forward_call=%d",
                name, module_type, tuple(t.shape), n_nan, n_inf, state["call_count"],
            )

    def _make_hook(name, module_type):
        def _hook(module, inputs, output):
            if isinstance(output, torch.Tensor):
                _check_tensor(name, module_type, output)
            elif isinstance(output, (list, tuple)):
                for o in output:
                    _check_tensor(name, module_type, o)
            elif isinstance(output, dict):
                for o in output.values():
                    _check_tensor(name, module_type, o)
        return _hook

    n_hooked = 0
    for name, module in model.named_modules():
        if name == "":
            continue
        module.register_forward_hook(_make_hook(name, type(module).__name__))
        n_hooked += 1

    def _reset_state(module, inputs):
        state["logged_this_call"] = False
        state["call_count"] += 1

    model.register_forward_pre_hook(_reset_state)
    logging.info("(Ha) NaN-debug hooks installed on %d submodules (SAM3_NAN_DEBUG_HOOKS=1)", n_hooked)


_patch_embed_nan_first_seen_step = None


def _check_patch_embed_weight_nan(model: nn.Module, step: int, scaler) -> None:
    """
    (Ha) Added, 2026-07-18: check specifically whether
    `backbone.vision_backbone.trunk.patch_embed.proj`'s weight/bias have gone
    non-finite immediately after this step's optimizer update, together with
    the AMP GradScaler's current scale factor.

    Confirmed 2026-07-18 (see `.github/skills/loss-debugging/SKILL.md` Step
    9) that this EXACT Conv2d's weights are the origin of the "zero-loss
    collapse": every one of 255 NaN-debug hook hits across a persistent
    collapse pointed at this same module, with 100% of its output tensor
    NaN -- consistent with the layer's own parameters (not just its
    activations for one bad input) having gone NaN.

    The gradient-clipping/AMP order in `_train_epoch` (`scaler.unscale_()`
    -> `gradient_clipper()` -> `scaler.step()` -> `scaler.update()`) is the
    textbook-correct order: `unscale_()` should record `found_inf` from the
    PRE-clipping gradient values, so `scaler.step()` should skip applying
    ANY update when a gradient was non-finite, regardless of what clipping
    does to the (discarded) values afterward. This check exists to catch the
    exact step + scaler scale value where this protection apparently fails,
    since the theory says it shouldn't be possible for these weights to
    become NaN through a normal skipped-step path.

    Only logs once (the FIRST time this is detected this run) to avoid
    spamming every subsequent step once corrupted -- gated behind
    `SAM3_NAN_DEBUG_HOOKS=1`, same as `_install_nan_debug_hooks`.
    """
    global _patch_embed_nan_first_seen_step
    if os.environ.get("SAM3_NAN_DEBUG_HOOKS") != "1":
        return
    if _patch_embed_nan_first_seen_step is not None:
        return  # already caught and logged once this run

    try:
        m = model.module if hasattr(model, "module") else model
        proj = m.backbone.vision_backbone.trunk.patch_embed.proj
    except AttributeError:
        return

    for pname, p in (("weight", proj.weight), ("bias", proj.bias)):
        if p is None:
            continue
        if not torch.isfinite(p).all():
            _patch_embed_nan_first_seen_step = step
            n_nan = torch.isnan(p).sum().item()
            n_inf = torch.isinf(p).sum().item()
            logging.error(
                "(Ha) WEIGHT-NaN: patch_embed.proj.%s went non-finite at step=%d, "
                "scaler.get_scale()=%s, n_nan=%d, n_inf=%d, param_numel=%d",
                pname, step, scaler.get_scale(), n_nan, n_inf, p.numel(),
            )


class Trainer:
    """
    Trainer supporting the DDP training strategies.
    """

    EPSILON = 1e-8

    def __init__(
        self,
        *,  # the order of these args can change at any time, so they are keyword-only
        data: Dict[str, Any],
        model: Dict[str, Any],
        logging: Dict[str, Any],
        checkpoint: Dict[str, Any],
        max_epochs: int,
        mode: str = "train",
        accelerator: str = "cuda",
        seed_value: int = 123,
        val_epoch_freq: int = 1,
        distributed: Dict[str, bool] = None,
        cuda: Dict[str, bool] = None,
        env_variables: Optional[Dict[str, Any]] = None,
        optim: Optional[Dict[str, Any]] = None,
        optim_overrides: Optional[List[Dict[str, Any]]] = None,
        meters: Optional[Dict[str, Any]] = None,
        loss: Optional[Dict[str, Any]] = None,
        skip_first_val: bool = False,
        skip_saving_ckpts: bool = False,
        empty_gpu_mem_cache_after_eval: bool = True,
        gradient_accumulation_steps: int = 1,
    ):
        self._setup_env_variables(env_variables)
        self._setup_timers()

        self.data_conf = data
        self.model_conf = model
        self.logging_conf = LoggingConf(**logging)
        self.checkpoint_conf = CheckpointConf(**checkpoint).infer_missing()
        self.max_epochs = max_epochs
        self.mode = mode
        self.val_epoch_freq = val_epoch_freq
        self.optim_conf = OptimConf(**optim) if optim is not None else OptimConf()
        self.meters_conf = meters
        self.loss_conf = loss
        self.gradient_accumulation_steps = gradient_accumulation_steps
        distributed = DistributedConf(**distributed or {})
        cuda = CudaConf(**cuda or {})
        self.where = 0.0

        self.skip_first_val = skip_first_val
        self.skip_saving_ckpts = skip_saving_ckpts
        self.empty_gpu_mem_cache_after_eval = empty_gpu_mem_cache_after_eval

        self._infer_distributed_backend_if_none(distributed, accelerator)

        self._setup_device(accelerator)

        self._setup_torch_dist_and_backend(cuda, distributed)

        makedir(self.logging_conf.log_dir)
        setup_logging(
            __name__,
            output_dir=self.logging_conf.log_dir,
            rank=self.rank,
            log_level_primary=self.logging_conf.log_level_primary,
            log_level_secondary=self.logging_conf.log_level_secondary,
        )

        set_seeds(seed_value, self.max_epochs, self.distributed_rank)
        log_env_variables()

        assert (
            is_dist_avail_and_initialized()
        ), "Torch distributed needs to be initialized before calling the trainer."

        self._setup_components()  # Except Optimizer everything is setup here.
        self._move_to_device()
        self._construct_optimizers()
        self._setup_dataloaders()

        self.time_elapsed_meter = DurationMeter("Time Elapsed", self.device, ":.2f")

        if self.checkpoint_conf.resume_from is not None:
            assert os.path.exists(
                self.checkpoint_conf.resume_from
            ), f"The 'resume_from' checkpoint {self.checkpoint_conf.resume_from} does not exist!"
            dst = os.path.join(self.checkpoint_conf.save_dir, "checkpoint.pt")
            if self.distributed_rank == 0 and not os.path.exists(dst):
                # Copy the "resume_from" checkpoint to the checkpoint folder
                # if there is not a checkpoint to resume from already there
                makedir(self.checkpoint_conf.save_dir)
                g_pathmgr.copy(self.checkpoint_conf.resume_from, dst)
            barrier()

        self.load_checkpoint()
        self._setup_ddp_distributed_training(distributed, accelerator)
        barrier()

    def _setup_timers(self):
        """
        Initializes counters for elapsed time and eta.
        """
        self.start_time = time.time()
        self.ckpt_time_elapsed = 0
        self.est_epoch_time = dict.fromkeys([Phase.TRAIN, Phase.VAL], 0)

    def _get_meters(self, phase_filters=None):
        if self.meters is None:
            return {}
        meters = {}
        for phase, phase_meters in self.meters.items():
            if phase_filters is not None and phase not in phase_filters:
                continue
            for key, key_meters in phase_meters.items():
                if key_meters is None:
                    continue
                for name, meter in key_meters.items():
                    meters[f"{phase}_{key}/{name}"] = meter
        return meters

    def _infer_distributed_backend_if_none(self, distributed_conf, accelerator):
        if distributed_conf.backend is None:
            distributed_conf.backend = "nccl" if accelerator == "cuda" else "gloo"

    def _setup_env_variables(self, env_variables_conf) -> None:
        if env_variables_conf is not None:
            for variable_name, value in env_variables_conf.items():
                os.environ[variable_name] = value

    def _setup_torch_dist_and_backend(self, cuda_conf, distributed_conf) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = cuda_conf.cudnn_deterministic
            torch.backends.cudnn.benchmark = cuda_conf.cudnn_benchmark
            torch.backends.cuda.matmul.allow_tf32 = (
                cuda_conf.matmul_allow_tf32
                if cuda_conf.matmul_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )
            torch.backends.cudnn.allow_tf32 = (
                cuda_conf.cudnn_allow_tf32
                if cuda_conf.cudnn_allow_tf32 is not None
                else cuda_conf.allow_tf32
            )

        self.rank = setup_distributed_backend(
            distributed_conf.backend, distributed_conf.timeout_mins
        )

    def _setup_device(self, accelerator):
        self.local_rank, self.distributed_rank = get_machine_local_and_dist_rank()
        if accelerator == "cuda":
            self.device = torch.device("cuda", self.local_rank)
            torch.cuda.set_device(self.local_rank)
        elif accelerator == "cpu":
            self.device = torch.device("cpu")
        else:
            raise ValueError(f"Unsupported accelerator: {accelerator}")

    def _setup_ddp_distributed_training(self, distributed_conf, accelerator):
        assert isinstance(self.model, torch.nn.Module)

        self.model = nn.parallel.DistributedDataParallel(
            self.model,
            device_ids=[self.local_rank] if accelerator == "cuda" else [],
            find_unused_parameters=distributed_conf.find_unused_parameters,
            gradient_as_bucket_view=distributed_conf.gradient_as_bucket_view,
            static_graph=distributed_conf.static_graph,
        )
        if distributed_conf.comms_dtype is not None:  # noqa
            from torch.distributed.algorithms import ddp_comm_hooks

            amp_type = get_amp_type(distributed_conf.comms_dtype)
            if amp_type == torch.bfloat16:
                hook = ddp_comm_hooks.default_hooks.bf16_compress_hook
                logging.info("Enabling bfloat16 grad communication")
            else:
                hook = ddp_comm_hooks.default_hooks.fp16_compress_hook
                logging.info("Enabling fp16 grad communication")
            process_group = None
            self.model.register_comm_hook(process_group, hook)

    def _move_to_device(self):
        logging.info(
            f"Moving components to device {self.device} and local rank {self.local_rank}."
        )

        self.model.to(self.device)

        logging.info(
            f"Done moving components to device {self.device} and local rank {self.local_rank}."
        )

    def save_checkpoint(self, epoch, checkpoint_names=None):
        if self.skip_saving_ckpts:
            logging.info(
                "skip_saving_ckpts is set to True. So, no checkpoints have been saved."
            )
            return
        checkpoint_folder = self.checkpoint_conf.save_dir
        makedir(checkpoint_folder)
        if checkpoint_names is None:
            checkpoint_names = ["checkpoint"]
            if (
                self.checkpoint_conf.save_freq > 0
                and (int(epoch) % self.checkpoint_conf.save_freq == 0)
            ) or int(epoch) in self.checkpoint_conf.save_list:
                checkpoint_names.append(f"checkpoint_{int(epoch)}")

        checkpoint_paths = []
        for ckpt_name in checkpoint_names:
            checkpoint_paths.append(os.path.join(checkpoint_folder, f"{ckpt_name}.pt"))

        state_dict = unwrap_ddp_if_wrapped(self.model).state_dict()
        state_dict = exclude_params_matching_unix_pattern(
            patterns=self.checkpoint_conf.skip_saving_parameters, state_dict=state_dict
        )

        checkpoint = {
            "model": state_dict,
            "optimizer": self.optim.optimizer.state_dict(),
            "epoch": epoch,
            "loss": self.loss.state_dict(),
            "steps": self.steps,
            "time_elapsed": self.time_elapsed_meter.val,
            "best_meter_values": self.best_meter_values,
        }
        if self.optim_conf.amp.enabled:
            checkpoint["scaler"] = self.scaler.state_dict()

        # DDP checkpoints are only saved on rank 0 (all workers are identical)
        if self.distributed_rank != 0:
            return

        for checkpoint_path in checkpoint_paths:
            self._save_checkpoint(checkpoint, checkpoint_path)

    def _save_checkpoint(self, checkpoint, checkpoint_path):
        """
        Save a checkpoint while guarding against the job being killed in the middle
        of checkpoint saving (which corrupts the checkpoint file and ruins the
        entire training since usually only the last checkpoint is kept per run).

        We first save the new checkpoint to a temp file (with a '.tmp' suffix), and
        and move it to overwrite the old checkpoint_path.
        """
        checkpoint_path_tmp = f"{checkpoint_path}.tmp"
        with g_pathmgr.open(checkpoint_path_tmp, "wb") as f:
            torch.save(checkpoint, f)
        # after torch.save is completed, replace the old checkpoint with the new one
        if g_pathmgr.exists(checkpoint_path):
            # remove the old checkpoint_path file first (otherwise g_pathmgr.mv fails)
            g_pathmgr.rm(checkpoint_path)
        success = g_pathmgr.mv(checkpoint_path_tmp, checkpoint_path)
        assert success

    def load_checkpoint(self):
        ckpt_path = get_resume_checkpoint(self.checkpoint_conf.save_dir)
        if ckpt_path is None:
            self._init_model_state()
        else:
            if self.checkpoint_conf.initialize_after_preemption:
                self._call_model_initializer()
            self._load_resuming_checkpoint(ckpt_path)

    def _init_model_state(self):
        # Checking that parameters that won't be saved are indeed frozen
        # We do this check here before even saving the model to catch errors
        # are early as possible and not at the end of the first epoch
        assert_skipped_parameters_are_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
        )

        # Checking that parameters that won't be saved are initialized from
        # within the model definition, unless `initialize_after_preemption`
        # is explicitly set to `True`. If not, this is a bug, and after
        # preemption, the `skip_saving_parameters` will have random values
        allow_init_skip_parameters = self.checkpoint_conf.initialize_after_preemption
        with with_check_parameter_frozen(
            patterns=self.checkpoint_conf.skip_saving_parameters,
            model=self.model,
            disabled=allow_init_skip_parameters,
        ):
            self._call_model_initializer()

    def _call_model_initializer(self):
        model_weight_initializer = instantiate(
            self.checkpoint_conf.model_weight_initializer
        )
        if model_weight_initializer is not None:
            logging.info(
                f"Loading pretrained checkpoint from {self.checkpoint_conf.model_weight_initializer}"
            )
            self.model = model_weight_initializer(model=self.model)

    def _load_resuming_checkpoint(self, ckpt_path: str):
        logging.info(f"Resuming training from {ckpt_path}")

        with g_pathmgr.open(ckpt_path, "rb") as f:
            checkpoint = torch.load(f, map_location=DEVICE)
        load_state_dict_into_model(
            model=self.model,
            state_dict=checkpoint["model"],
            ignore_missing_keys=self.checkpoint_conf.skip_saving_parameters,
        )

        self.optim.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss.load_state_dict(checkpoint["loss"], strict=True)
        self.epoch = checkpoint["epoch"]
        self.steps = checkpoint["steps"]
        self.ckpt_time_elapsed = checkpoint.get("time_elapsed")

        # (Ha) Added, 2026-07-18: when amp.bypass_dynamic_scale_for_bf16 is active,
        # deliberately do NOT restore the checkpoint's saved scaler state here.
        # GradScaler.state_dict()/load_state_dict() round-trips ALL FIVE fields --
        # scale, growth_factor, backoff_factor, growth_interval, _growth_tracker --
        # so restoring a checkpoint saved by an OLD run (default dynamic scaler)
        # would silently overwrite the pinned-scale GradScaler constructed above
        # with the old dynamic growth/backoff factors (and whatever scale value
        # -- possibly already decayed -- that old run had reached). Confirmed this
        # exact failure mode: job 237943948 (first attempt at this fix) resumed
        # from this same checkpoint and reproduced the IDENTICAL subnormal scale
        # (2.938735877055719e-39) and weight-NaN collapse the fix was meant to
        # prevent, because the checkpoint's restored state clobbered it. Skipping
        # the restore keeps the freshly pinned scaler (~1.0) across resumes too.
        if (
            self.optim_conf.amp.enabled
            and "scaler" in checkpoint
            and not getattr(self, "_scaler_bypass_dynamic_scale", False)
        ):
            self.scaler.load_state_dict(checkpoint["scaler"])
        elif getattr(self, "_scaler_bypass_dynamic_scale", False):
            logging.info(
                "(Ha) amp.bypass_dynamic_scale_for_bf16=True: skipping restore of "
                "checkpoint's saved GradScaler state (scale/growth_factor/"
                "backoff_factor) to keep the scale pinned at ~1.0 across resumes."
            )

        self.best_meter_values = checkpoint.get("best_meter_values", {})

        if "train_dataset" in checkpoint and self.train_dataset is not None:
            self.train_dataset.load_checkpoint_state(checkpoint["train_dataset"])

    def is_intermediate_val_epoch(self, epoch):
        skip_epoch = self.skip_first_val and epoch == 0
        return (
            epoch % self.val_epoch_freq == 0
            and epoch < self.max_epochs - 1
            and not skip_epoch
        )

    def _find_loss(self, key: str):
        if key in self.loss:
            return self.loss[key]

        assert key != "all", "Loss must be specified for key='all'"
        assert (
            "default" in self.loss
        ), f"Key {key} not found in losss, and no default provided"
        return self.loss["default"]

    def _find_meter(self, phase: str, key: str):
        if key in self.meters[phase]:
            return self.meters[phase][key]

        for cand_key, meter in self.meters[phase].items():
            if fnmatch.fnmatch(key, cand_key):
                return meter
        return None

    def _maybe_dump_anomalous_batch(
        self, batch: BatchedDatapoint, key: str, phase: str, loss: torch.Tensor
    ) -> None:
        """
        (Ha) Added, 2026-07-17: on an anomalous loss (non-finite, or exactly
        zero for a real -- non-DummyLoss -- key), save the batch that
        produced it to disk: the post-augmentation image tensor + boxes
        (`find_targets`), and, if available, the pre-augmentation
        `raw_images`/`raw_boxes` (see `Datapoint.raw_boxes` in
        sam3_image_dataset.py) -- so a future investigation can directly
        compare "before" vs. "after" augmentation and tell whether the
        augmentation pipeline introduced the anomaly, or it was already
        present in the source data.

        Rate-limited to MAX_ANOMALOUS_DUMPS per run: a persistent collapse
        can hold at exactly-zero loss for an entire epoch (as observed on
        experiments/5cat_matrix_chip_v2, epoch 9 of job 237436875) --
        dumping every step would write hundreds of large (~GB-scale) files.

        (Ha) Added, 2026-07-20: restricted to phase == Phase.TRAIN. During
        validation, `key` is the val dict_key (e.g. "pin"), which is never a
        key in `self.loss` (only "all"/"default" are), so `_find_loss()`
        always falls back to `self.loss["default"]` == DummyLoss -- which
        returns exactly 0.0 BY DESIGN every single val step (validation only
        needs predictions for the COCO evaluator, not a real optimized loss).
        The `key != "default"` check above doesn't catch this because `key`
        is the literal string "pin", not "default", even though it resolves
        to DummyLoss. Confirmed real-world impact (job 238286332): all 3
        dumps of the MAX_ANOMALOUS_DUMPS=3 budget were burned within 8
        seconds of the very first validation epoch (val_pin_epoch5_step{0,1,2}),
        all loss=0.0 false positives -- meaning a genuine TRAINING collapse
        later in that same run would NOT have been dumped, since the shared
        budget was already exhausted on harmless validation noise.
        """
        if phase != Phase.TRAIN:
            return
        MAX_ANOMALOUS_DUMPS = 3
        if self.distributed_rank != 0:
            return
        if self._anomalous_dump_count >= MAX_ANOMALOUS_DUMPS:
            return

        loss_val = loss.item() if isinstance(loss, torch.Tensor) else float(loss)
        if math.isfinite(loss_val) and loss_val != 0.0:
            return

        dump_dir = os.path.join(
            os.path.dirname(self.checkpoint_conf.save_dir),
            "dumps",
            "anomalous_batches",
        )
        makedir(dump_dir)
        dump_path = os.path.join(
            dump_dir,
            f"{phase}_{key}_epoch{int(self.epoch)}_step{self.steps[phase]}_"
            f"{self._anomalous_dump_count}.pt",
        )
        payload = {
            "phase": phase,
            "key": key,
            "epoch": int(self.epoch),
            "step": self.steps[phase],
            "loss_value": loss_val,
            # Post-augmentation: exactly what the model saw this step.
            "img_batch": batch.img_batch.detach().cpu(),
            "find_targets": [
                {
                    "boxes": (
                        t.boxes.detach().cpu() if t.boxes is not None else None
                    ),
                    "num_boxes": (
                        t.num_boxes.detach().cpu()
                        if t.num_boxes is not None
                        else None
                    ),
                    # (Ha) Added, 2026-07-18: per-object segmentation masks,
                    # so dumps can be visualized with mask overlays, not just
                    # boxes. `t.segments` is a flat list of per-object bool
                    # mask tensors (same order as `t.boxes`), or None if this
                    # config doesn't load segmentation masks
                    # (`with_seg_masks=False`) -- see collator.py.
                    "segments": (
                        [s.detach().cpu() for s in t.segments]
                        if getattr(t, "segments", None) is not None
                        else None
                    ),
                }
                for t in batch.find_targets
            ],
            # Pre-augmentation: the raw source, if the dataset populated it
            # (see Datapoint.raw_images / Datapoint.raw_boxes). None if the
            # dataset class in use doesn't set these.
            "raw_images": batch.raw_images,
            "raw_boxes": batch.raw_boxes,
        }

        # (Ha) Added, 2026-07-18: if the loss module captured raw model
        # output tensors at the matched indices (only set when this dump was
        # triggered by the zero-loss diagnostic -- see
        # Sam3LossWrapper.last_zero_loss_debug in sam3_loss.py), include them
        # here so we can check whether the matched (prediction, target) pairs
        # are pointing at a zero-valued padding slot on both sides, which
        # would trivially make every individual loss term compute to 0.0.
        try:
            loss_module = self._find_loss(key)
            debug_info = getattr(loss_module, "last_zero_loss_debug", None)
            if debug_info is not None:
                payload["model_output_debug"] = {
                    k: (v.cpu() if isinstance(v, torch.Tensor) else v)
                    for k, v in debug_info.items()
                }
                loss_module.last_zero_loss_debug = None  # consumed, reset for next time
        except Exception:
            logging.exception("(Ha) Failed to attach model_output_debug to dump (non-fatal)")

        try:
            with g_pathmgr.open(dump_path, "wb") as f:
                torch.save(payload, f)
            logging.error(
                "(Ha) Dumped anomalous batch (loss=%s) to %s (dump %d/%d this run)",
                loss_val,
                dump_path,
                self._anomalous_dump_count + 1,
                MAX_ANOMALOUS_DUMPS,
            )
        except Exception:
            logging.exception("(Ha) Failed to dump anomalous batch to %s", dump_path)

        # (Ha) Added, 2026-07-18: also render a JPEG right away (box + mask
        # overlays) so there's already something to look at without a manual
        # follow-up script run. Best-effort: never let a visualization bug
        # take down the training loop -- log and move on.
        try:
            jpeg_dir = os.path.join(
                dump_dir + "_jpeg",
                f"{phase}_{key}_epoch{int(self.epoch)}_step{self.steps[phase]}_"
                f"{self._anomalous_dump_count}",
            )
            _save_anomalous_batch_jpegs(payload, jpeg_dir)
            logging.error("(Ha) Saved anomalous batch JPEGs to %s", jpeg_dir)
        except Exception:
            logging.exception("(Ha) Failed to save anomalous batch JPEGs (non-fatal)")

        self._anomalous_dump_count += 1

    def _step(
        self,
        batch: BatchedDatapoint,
        model: nn.Module,
        phase: str,
    ):
        key, batch = batch.popitem()
        batch = copy_data_to_device(batch, self.device, non_blocking=True)

        find_stages = model(batch)
        find_targets = [
            unwrap_ddp_if_wrapped(model).back_convert(x) for x in batch.find_targets
        ]
        batch_size = len(batch.img_batch)
        loss = self._find_loss(key)(find_stages, find_targets)

        loss_str = f"Losses/{phase}_{key}_loss"

        loss_log_str = os.path.join("Step_Losses", loss_str)

        # loss contains multiple sub-components we wish to log
        step_losses = {}
        if isinstance(loss, dict):
            step_losses.update(
                {f"Losses/{phase}_{key}_{k}": v for k, v in loss.items()}
            )
            loss = self._log_loss_detailed_and_return_core_loss(
                loss, loss_log_str, self.steps[phase]
            )

        # (Ha) Added, 2026-07-17: dump the exact batch (images + boxes, both
        # pre- and post-augmentation) when the real loss goes non-finite or
        # exactly zero, so a future investigation can inspect precisely what
        # data was involved instead of only aggregate loss values. "default"
        # is DummyLoss (always 0 by design, see loss/sam3_loss.py), so it's
        # excluded here to avoid a false-positive dump every step.
        if key != "default":
            self._maybe_dump_anomalous_batch(batch, key, phase, loss)

        if self.steps[phase] % self.logging_conf.log_scalar_frequency == 0:
            self.logger.log(
                loss_log_str,
                loss,
                self.steps[phase],
            )

        self.steps[phase] += 1

        ret_tuple = {loss_str: loss}, batch_size, step_losses

        if phase not in self.meters:
            return ret_tuple

        meters_dict = self._find_meter(phase, key)
        if meters_dict is None:
            return ret_tuple
        if meters_dict is not None:
            for _, meter in meters_dict.items():
                meter.update(
                    find_stages=find_stages,
                    find_metadatas=batch.find_metadatas,
                    model=model,
                    batch=batch,
                    key=key,
                )
            # Cleanup memory
            if isinstance(find_stages, SAM3Output):
                for fs in find_stages:
                    for k in list(fs.keys()):
                        del fs[k]

        return ret_tuple

    def run(self):
        assert self.mode in ["train", "train_only", "val"]
        if self.mode == "train":
            if self.epoch > 0:
                logging.info(f"Resuming training from epoch: {self.epoch}")
                # resuming from a checkpoint
                if self.is_intermediate_val_epoch(self.epoch - 1):
                    logging.info("Running previous val epoch")
                    self.epoch -= 1
                    self.run_val()
                    self.epoch += 1
            self.run_train()
            self.run_val()
        elif self.mode == "val":
            self.run_val()
        elif self.mode == "train_only":
            self.run_train()

    def _setup_dataloaders(self):
        self.train_dataset = None
        self.val_dataset = None

        if self.mode in ["train", "val"]:
            self.val_dataset = instantiate(self.data_conf.get(Phase.VAL, None))

        if self.mode in ["train", "train_only"]:
            self.train_dataset = instantiate(self.data_conf.train)

    def run_train(self):
        while self.epoch < self.max_epochs:
            dataloader = self.train_dataset.get_loader(epoch=int(self.epoch))
            barrier()
            outs = self.train_epoch(dataloader)
            self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

            # log train to text file.
            if self.distributed_rank == 0:
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "train_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(outs) + "\n")

            # Save checkpoint before validating
            self.save_checkpoint(self.epoch + 1)

            del dataloader
            gc.collect()

            # Run val, not running on last epoch since will run after the
            # loop anyway
            if self.is_intermediate_val_epoch(self.epoch):
                self.run_val()
                if torch.cuda.is_available() and self.empty_gpu_mem_cache_after_eval:
                    # release memory buffers held by the model during eval (which typically
                    # involves a lot more frames in video grounding that during training)
                    torch.cuda.empty_cache()

            if self.distributed_rank == 0:
                self.best_meter_values.update(self._get_trainer_state("train"))
                with g_pathmgr.open(
                    os.path.join(self.logging_conf.log_dir, "best_stats.json"),
                    "a",
                ) as f:
                    f.write(json.dumps(self.best_meter_values) + "\n")

            self.epoch += 1
        # epoch was incremented in the loop but the val step runs out of the loop
        self.epoch -= 1

    def run_val(self):
        if not self.val_dataset:
            return

        dataloader = self.val_dataset.get_loader(epoch=int(self.epoch))
        outs = self.val_epoch(dataloader, phase=Phase.VAL)
        del dataloader
        gc.collect()
        self.logger.log_dict(outs, self.epoch)  # Logged only on rank 0

        if self.distributed_rank == 0:
            with g_pathmgr.open(
                os.path.join(self.logging_conf.log_dir, "val_stats.json"),
                "a",
            ) as f:
                f.write(json.dumps(outs) + "\n")

    def val_epoch(self, val_loader, phase):
        batch_time = AverageMeter("Batch Time", self.device, ":.2f")
        data_time = AverageMeter("Data Time", self.device, ":.2f")
        mem = MemMeter("Mem (GB)", self.device, ":.2f")

        iters_per_epoch = len(val_loader)

        curr_phases = [phase]
        curr_models = [self.model]

        loss_names = []
        for p in curr_phases:
            for key in self.loss.keys():
                loss_names.append(f"Losses/{p}_{key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        for model in curr_models:
            model.eval()
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_start"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_start()

        progress = ProgressMeter(
            iters_per_epoch,
            [batch_time, data_time, mem, self.time_elapsed_meter, *loss_mts.values()],
            self._get_meters(curr_phases),
            prefix="Val Epoch: [{}]".format(self.epoch),
        )

        end = time.time()

        for data_iter, batch in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            # batch = batch.to(self.device, non_blocking=True)

            # compute output
            with torch.no_grad():
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=(self.optim_conf.amp.enabled if self.optim_conf else False),
                    dtype=(
                        get_amp_type(self.optim_conf.amp.amp_dtype)
                        if self.optim_conf
                        else None
                    ),
                ):
                    for phase, model in zip(curr_phases, curr_models):
                        loss_dict, batch_size, extra_losses = self._step(
                            batch,
                            model,
                            phase,
                        )

                        assert len(loss_dict) == 1
                        loss_key, loss = loss_dict.popitem()

                        if loss_key in loss_mts:
                            loss_mts[loss_key].update(loss.item(), batch_size)

                        for k, v in extra_losses.items():
                            if k not in extra_loss_mts:
                                extra_loss_mts[k] = AverageMeter(k, self.device, ":.2e")
                            extra_loss_mts[k].update(v.item(), batch_size)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            self.time_elapsed_meter.update(
                time.time() - self.start_time + self.ckpt_time_elapsed
            )

            if torch.cuda.is_available():
                mem.update(reset_peak_usage=True)

            if data_iter % self.logging_conf.log_freq == 0:
                progress.display(data_iter)

            if data_iter % self.logging_conf.log_scalar_frequency == 0:
                # Log progress meters.
                for progress_meter in progress.meters:
                    self.logger.log(
                        os.path.join("Step_Stats", phase, progress_meter.name),
                        progress_meter.val,
                        self.steps[Phase.VAL],
                    )

            if data_iter % 10 == 0:
                dist.barrier()

        self.est_epoch_time[phase] = batch_time.avg * iters_per_epoch
        self._log_timers(phase)
        for model in curr_models:
            if hasattr(unwrap_ddp_if_wrapped(model), "on_validation_epoch_end"):
                unwrap_ddp_if_wrapped(model).on_validation_epoch_end()

        out_dict = self._log_meters_and_save_best_ckpts(curr_phases)

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg

        for phase in curr_phases:
            out_dict.update(self._get_trainer_state(phase))
        self._reset_meters(curr_phases)
        logging.info(f"Meters: {out_dict}")
        return out_dict

    def _get_trainer_state(self, phase):
        return {
            "Trainer/where": self.where,
            "Trainer/epoch": self.epoch,
            f"Trainer/steps_{phase}": self.steps[phase],
        }

    def train_epoch(self, train_loader):
        # Init stat meters
        batch_time_meter = AverageMeter("Batch Time", self.device, ":.2f")
        data_time_meter = AverageMeter("Data Time", self.device, ":.2f")
        mem_meter = MemMeter("Mem (GB)", self.device, ":.2f")
        data_times = []
        phase = Phase.TRAIN

        iters_per_epoch = len(train_loader)

        loss_names = []
        for batch_key in self.loss.keys():
            loss_names.append(f"Losses/{phase}_{batch_key}_loss")

        loss_mts = OrderedDict(
            [(name, AverageMeter(name, self.device, ":.2e")) for name in loss_names]
        )
        extra_loss_mts = {}

        progress = ProgressMeter(
            iters_per_epoch,
            [
                batch_time_meter,
                data_time_meter,
                mem_meter,
                self.time_elapsed_meter,
                *loss_mts.values(),
            ],
            self._get_meters([phase]),
            prefix="Train Epoch: [{}]".format(self.epoch),
        )

        # Model training loop
        self.model.train()
        end = time.time()

        for data_iter, batch in enumerate(train_loader):
            # measure data loading time
            data_time_meter.update(time.time() - end)
            data_times.append(data_time_meter.val)
            # batch = batch.to(
            #     self.device, non_blocking=True
            # )  # move tensors in a tensorclass

            try:
                self._run_step(batch, phase, loss_mts, extra_loss_mts)

                # compute gradient and do optim step
                exact_epoch = self.epoch + float(data_iter) / iters_per_epoch
                self.where = float(exact_epoch) / self.max_epochs
                assert self.where <= 1 + self.EPSILON
                if self.where < 1.0:
                    self.optim.step_schedulers(
                        self.where, step=int(exact_epoch * iters_per_epoch)
                    )
                else:
                    logging.warning(
                        f"Skipping scheduler update since the training is at the end, i.e, {self.where} of [0,1]."
                    )

                # Log schedulers
                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    for j, param_group in enumerate(self.optim.optimizer.param_groups):
                        for option in self.optim.schedulers[j]:
                            optim_prefix = (
                                "" + f"{j}_"
                                if len(self.optim.optimizer.param_groups) > 1
                                else ""
                            )
                            self.logger.log(
                                os.path.join("Optim", f"{optim_prefix}", option),
                                param_group[option],
                                self.steps[phase],
                            )

                # Clipping gradients and detecting diverging gradients
                if self.gradient_clipper is not None:
                    self.scaler.unscale_(self.optim.optimizer)
                    self.gradient_clipper(model=self.model)

                if self.gradient_logger is not None:
                    self.gradient_logger(
                        self.model, rank=self.distributed_rank, where=self.where
                    )

                # Optimizer step: the scaler will make sure gradients are not
                # applied if the gradients are infinite
                self.scaler.step(self.optim.optimizer)
                self.scaler.update()

                # (Ha) Added, 2026-07-18: confirmed the "zero-loss collapse"
                # (loss-debugging SKILL.md Step 9) originates from
                # backbone.vision_backbone.trunk.patch_embed.proj's WEIGHTS
                # going NaN (not just its activations) -- check this specific
                # layer's weight/bias right after every optimizer step, so
                # the next occurrence tells us the EXACT step it happens at
                # and the scaler's scale factor then, to confirm/refute
                # whether scaler.step() is failing to skip a non-finite
                # gradient update (it's supposed to -- see the comment above).
                _check_patch_embed_weight_nan(self.model, self.steps[phase], self.scaler)

                # measure elapsed time
                batch_time_meter.update(time.time() - end)
                end = time.time()

                self.time_elapsed_meter.update(
                    time.time() - self.start_time + self.ckpt_time_elapsed
                )

                mem_meter.update(reset_peak_usage=True)
                if data_iter % self.logging_conf.log_freq == 0:
                    progress.display(data_iter)

                if data_iter % self.logging_conf.log_scalar_frequency == 0:
                    # Log progress meters.
                    for progress_meter in progress.meters:
                        self.logger.log(
                            os.path.join("Step_Stats", phase, progress_meter.name),
                            progress_meter.val,
                            self.steps[phase],
                        )

            # Catching NaN/Inf errors in the loss
            except FloatingPointError as e:
                raise e

        self.est_epoch_time[Phase.TRAIN] = batch_time_meter.avg * iters_per_epoch
        self._log_timers(Phase.TRAIN)
        self._log_sync_data_times(Phase.TRAIN, data_times)

        out_dict = self._log_meters_and_save_best_ckpts([Phase.TRAIN])

        for k, v in loss_mts.items():
            out_dict[k] = v.avg
        for k, v in extra_loss_mts.items():
            out_dict[k] = v.avg
        out_dict.update(self._get_trainer_state(phase))
        logging.info(f"Losses and meters: {out_dict}")
        self._reset_meters([phase])
        return out_dict

    def _log_sync_data_times(self, phase, data_times):
        data_times = all_reduce_max(torch.tensor(data_times)).tolist()
        steps = range(self.steps[phase] - len(data_times), self.steps[phase])
        for step, data_time in zip(steps, data_times):
            if step % self.logging_conf.log_scalar_frequency == 0:
                self.logger.log(
                    os.path.join("Step_Stats", phase, "Data Time Synced"),
                    data_time,
                    step,
                )

    def _run_step(
        self,
        batch: BatchedDatapoint,
        phase: str,
        loss_mts: Dict[str, AverageMeter],
        extra_loss_mts: Dict[str, AverageMeter],
        raise_on_error: bool = True,
    ):
        """
        Run the forward / backward
        """

        # it's important to set grads to None, especially with Adam since 0
        # grads will also update a model even if the step doesn't produce
        # gradients
        self.optim.zero_grad(set_to_none=True)

        if self.gradient_accumulation_steps > 1:
            assert isinstance(
                batch, list
            ), f"Expected a list of batches, got {type(batch)}"
            assert (
                len(batch) == self.gradient_accumulation_steps
            ), f"Expected {self.gradient_accumulation_steps} batches, got {len(batch)}"
            accum_steps = len(batch)
        else:
            accum_steps = 1
            batch = [batch]

        for i, chunked_batch in enumerate(batch):
            ddp_context = (
                self.model.no_sync()
                if i < accum_steps - 1
                else contextlib.nullcontext()
            )
            with ddp_context:
                with torch.amp.autocast(
                    device_type="cuda",
                    enabled=self.optim_conf.amp.enabled,
                    dtype=get_amp_type(self.optim_conf.amp.amp_dtype),
                ):
                    loss_dict, batch_size, extra_losses = self._step(
                        chunked_batch,
                        self.model,
                        phase,
                    )

                assert len(loss_dict) == 1
                loss_key, loss = loss_dict.popitem()

                if not math.isfinite(loss.item()):
                    error_msg = f"Loss is {loss.item()}, attempting to stop training"
                    logging.error(error_msg)
                    if raise_on_error:
                        raise FloatingPointError(error_msg)
                    else:
                        return

                self.scaler.scale(loss).backward()
                loss_mts[loss_key].update(loss.item(), batch_size)
                for extra_loss_key, extra_loss in extra_losses.items():
                    if extra_loss_key not in extra_loss_mts:
                        extra_loss_mts[extra_loss_key] = AverageMeter(
                            extra_loss_key, self.device, ":.2e"
                        )
                    extra_loss_mts[extra_loss_key].update(extra_loss.item(), batch_size)

    def _log_meters_and_save_best_ckpts(self, phases: List[str]):
        logging.info("Synchronizing meters")
        out_dict = {}
        checkpoint_save_keys = []
        for key, meter in self._get_meters(phases).items():
            meter_output = meter.compute_synced()
            is_better_check = getattr(meter, "is_better", None)

            for meter_subkey, meter_value in meter_output.items():
                out_dict[os.path.join("Meters_train", key, meter_subkey)] = meter_value

                if is_better_check is None:
                    continue

                tracked_meter_key = os.path.join(key, meter_subkey)
                if tracked_meter_key not in self.best_meter_values or is_better_check(
                    meter_value,
                    self.best_meter_values[tracked_meter_key],
                ):
                    self.best_meter_values[tracked_meter_key] = meter_value

                    # (Ha) Added, 2026-07-17: only save a checkpoint for
                    # sub-keys explicitly allow-listed in
                    # save_best_meter_subkeys (if set), instead of every
                    # improving sub-key — see CheckpointConf docstring above
                    # for why (avoids the idle-GPU auto-kill from writing many
                    # large checkpoints back-to-back).
                    subkey_allowed = (
                        self.checkpoint_conf.save_best_meter_subkeys is None
                        or meter_subkey in self.checkpoint_conf.save_best_meter_subkeys
                    )
                    if (
                        self.checkpoint_conf.save_best_meters is not None
                        and key in self.checkpoint_conf.save_best_meters
                        and subkey_allowed
                    ):
                        checkpoint_save_keys.append(tracked_meter_key.replace("/", "_"))

        if len(checkpoint_save_keys) > 0:
            self.save_checkpoint(self.epoch + 1, checkpoint_save_keys)

        return out_dict

    def _log_timers(self, phase):
        time_remaining = 0
        epochs_remaining = self.max_epochs - self.epoch - 1
        val_epochs_remaining = sum(
            n % self.val_epoch_freq == 0 for n in range(self.epoch, self.max_epochs)
        )

        # Adding the guaranteed val run at the end if val_epoch_freq doesn't coincide with
        # the end epoch.
        if (self.max_epochs - 1) % self.val_epoch_freq != 0:
            val_epochs_remaining += 1

        # Remove the current val run from estimate
        if phase == Phase.VAL:
            val_epochs_remaining -= 1

        time_remaining += (
            epochs_remaining * self.est_epoch_time[Phase.TRAIN]
            + val_epochs_remaining * self.est_epoch_time[Phase.VAL]
        )

        self.logger.log(
            os.path.join("Step_Stats", phase, self.time_elapsed_meter.name),
            self.time_elapsed_meter.val,
            self.steps[phase],
        )

        logging.info(f"Estimated time remaining: {human_readable_time(time_remaining)}")

    def _reset_meters(self, phases: str) -> None:
        for meter in self._get_meters(phases).values():
            meter.reset()

    def _check_val_key_match(self, val_keys, phase):
        if val_keys is not None:
            # Check if there are any duplicates
            assert len(val_keys) == len(
                set(val_keys)
            ), f"Duplicate keys in val datasets, keys: {val_keys}"

            # Check that the keys match the meter keys
            if self.meters_conf is not None and phase in self.meters_conf:
                assert set(val_keys) == set(self.meters_conf[phase].keys()), (
                    f"Keys in val datasets do not match the keys in meters."
                    f"\nMissing in meters: {set(val_keys) - set(self.meters_conf[phase].keys())}"
                    f"\nMissing in val datasets: {set(self.meters_conf[phase].keys()) - set(val_keys)}"
                )

            if self.loss_conf is not None:
                loss_keys = set(self.loss_conf.keys()) - set(["all"])
                if "default" not in loss_keys:
                    for k in val_keys:
                        assert (
                            k in loss_keys
                        ), f"Error: key {k} is not defined in the losses, and no default is set"

    def _setup_components(self):
        # Get the keys for all the val datasets, if any
        val_phase = Phase.VAL
        val_keys = None
        if self.data_conf.get(val_phase, None) is not None:
            val_keys = collect_dict_keys(self.data_conf[val_phase])
        # Additional checks on the sanity of the config for val datasets
        self._check_val_key_match(val_keys, phase=val_phase)

        logging.info("Setting up components: Model, loss, optim, meters etc.")
        self.epoch = 0
        self.steps = {Phase.TRAIN: 0, Phase.VAL: 0}

        self.logger = Logger(self.logging_conf)

        self.model = instantiate(self.model_conf, _convert_="all")
        #print_model_summary(self.model)
        self._maybe_freeze_non_decoder_params()
        _install_nan_debug_hooks(self.model)

        self.loss = None
        if self.loss_conf:
            self.loss = {
                key: el  # wrap_base_loss(el)
                for (key, el) in instantiate(self.loss_conf, _convert_="all").items()
            }
            self.loss = nn.ModuleDict(self.loss)

        self.meters = {}
        self.best_meter_values = {}
        # (Ha) Added, 2026-07-17: counts how many anomalous-batch dumps have
        # been written this run, to rate-limit _maybe_dump_anomalous_batch
        # (see its docstring) -- a persistent collapse (e.g. an entire epoch
        # at exactly-zero loss) would otherwise write one ~multi-GB dump per
        # step.
        self._anomalous_dump_count = 0
        if self.meters_conf:
            self.meters = instantiate(self.meters_conf, _convert_="all")

        # (Ha) Added, 2026-07-18: for bfloat16 AMP (which has fp32's exponent
        # range and never needed loss scaling -- GradScaler's dynamic scale
        # was designed for float16's narrow range), pin the scale at
        # (effectively) a constant 1.0 instead of letting it dynamically
        # decay. Confirmed this decay is the root cause of the
        # backbone.vision_backbone.trunk.patch_embed.proj weight-NaN collapse
        # -- see .github/skills/loss-debugging/SKILL.md Step 9/10 and the
        # `amp.bypass_dynamic_scale_for_bf16` config comment for the full
        # evidence. `enabled=True` is kept either way, so the found-inf
        # skip-step safety net (unscale_() + step()) still applies -- only
        # the scale VALUE itself is pinned, not the protection mechanism.
        # NOTE: torch.amp.GradScaler's constructor hard-asserts
        # growth_factor > 1.0 and backoff_factor < 1.0 (a literal 1.0 raises
        # AssertionError), so we can't set them to exactly 1.0. Instead use
        # near-identity factors (1 +/- 1e-7) with an astronomically large
        # growth_interval -- growth practically never triggers, and even
        # unlimited consecutive backoffs only erode the scale by ~1e-7 per
        # event, so it would take an unrealistic number of them (vs. the
        # ~143 halvings @ 0.5x that produced the 2.94e-39 subnormal collapse)
        # to meaningfully move the scale away from init_scale=1.0.
        amp_conf = self.optim_conf.amp if self.optim_conf else None
        bypass_dynamic_scale = bool(
            amp_conf
            and getattr(amp_conf, "bypass_dynamic_scale_for_bf16", False)
            and str(getattr(amp_conf, "amp_dtype", "")).lower() in ("bfloat16", "bf16")
        )
        # (Ha) Added, 2026-07-18: stashed so _load_resuming_checkpoint() can skip
        # restoring the checkpoint's saved scaler state below -- see the comment
        # there for why that's necessary (checkpoints saved by an OLD run using
        # the default dynamic scaler carry its growth_factor/backoff_factor/scale,
        # and GradScaler.load_state_dict() restores ALL of those fields, silently
        # overwriting/undoing the pinned-scale construction below on resume).
        self._scaler_bypass_dynamic_scale = bypass_dynamic_scale
        if bypass_dynamic_scale:
            logging.info(
                "(Ha) amp.bypass_dynamic_scale_for_bf16=True: pinning GradScaler's "
                "scale at (effectively) a constant 1.0 instead of the default "
                "dynamic scale, since bfloat16 doesn't need loss scaling and its "
                "decay was confirmed to corrupt weights."
            )
            self.scaler = torch.amp.GradScaler(
                self.device,
                enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
                init_scale=1.0,
                growth_factor=1.0 + 1e-7,
                backoff_factor=1.0 - 1e-7,
                growth_interval=2**31 - 1,
            )
        else:
            self.scaler = torch.amp.GradScaler(
                self.device,
                enabled=self.optim_conf.amp.enabled if self.optim_conf else False,
            )

        self.gradient_clipper = (
            instantiate(self.optim_conf.gradient_clip) if self.optim_conf else None
        )
        self.gradient_logger = (
            instantiate(self.optim_conf.gradient_logger) if self.optim_conf else None
        )

        logging.info("Finished setting up components: Model, loss, optim, meters etc.")

    def _maybe_freeze_non_decoder_params(self):
        """(Ha) Added 2026-07-20: when `optim.train_full_model=False`, freeze
        every model parameter EXCEPT the ones matching
        `optim.decoder_only_trainable_patterns` (default: the decoder /
        segmentation-head params, same patterns as the `transformer.decoder.*`
        LR group in `options.lr`), so only the decoder head gets fine-tuned.

        Frozen params are left in place -- they are NOT removed from the
        model or from their optimizer param group -- only `requires_grad` is
        flipped off. This means `construct_optimizer`'s
        "param groups must cover every model parameter" validation still
        passes unchanged, and toggling `train_full_model` back to True later
        (e.g. to resume full fine-tuning from a decoder-only checkpoint)
        requires no other code/config change: autograd simply won't populate
        `.grad` for frozen params, and AdamW.step() already skips any param
        whose `.grad is None`.
        """
        if not self.optim_conf or self.optim_conf.train_full_model:
            return
        patterns = self.optim_conf.decoder_only_trainable_patterns
        num_frozen_params, num_trainable_params = 0, 0
        num_frozen, num_trainable = 0, 0
        for name, param in self.model.named_parameters():
            if any(fnmatch.fnmatch(name, pattern) for pattern in patterns):
                num_trainable += 1
                num_trainable_params += param.numel()
                continue
            param.requires_grad = False
            num_frozen += 1
            num_frozen_params += param.numel()
        logging.info(
            "(Ha) optim.train_full_model=False: froze %d/%d parameter tensors "
            "(%d params), kept %d/%d tensors (%d params) trainable -- patterns=%s",
            num_frozen,
            num_frozen + num_trainable,
            num_frozen_params,
            num_trainable,
            num_frozen + num_trainable,
            num_trainable_params,
            patterns,
        )

    def _construct_optimizers(self):
        self.optim = construct_optimizer(
            self.model,
            self.optim_conf.optimizer,
            self.optim_conf.options,
            self.optim_conf.param_group_modifiers,
        )

    def _log_loss_detailed_and_return_core_loss(self, loss, loss_str, step):
        core_loss = loss.pop(CORE_LOSS_KEY)
        if step % self.logging_conf.log_scalar_frequency == 0:
            for k in loss:
                log_str = os.path.join(loss_str, k)
                self.logger.log(log_str, loss[k], step)
        return core_loss


def print_model_summary(model: torch.nn.Module, log_dir: str = ""):
    """
    Prints the model and the number of parameters in the model.
    # Multiple packages provide this info in a nice table format
    # However, they need us to provide an `input` (as they also write down the output sizes)
    # Our models are complex, and a single input is restrictive.
    # https://github.com/sksq96/pytorch-summary
    # https://github.com/nmhkahn/torchsummaryX
    """
    if get_rank() != 0:
        return
    param_kwargs = {}
    trainable_parameters = sum(
        p.numel() for p in model.parameters(**param_kwargs) if p.requires_grad
    )
    total_parameters = sum(p.numel() for p in model.parameters(**param_kwargs))
    non_trainable_parameters = total_parameters - trainable_parameters
    logging.info("==" * 10)
    logging.info(f"Summary for model {type(model)}")
    #logging.info(f"Model is {model}")
    #logging.info(f"\tTotal parameters {get_human_readable_count(total_parameters)}")
    #logging.info(
    #    f"\tTrainable parameters {get_human_readable_count(trainable_parameters)}"
    #)
    #logging.info(
    #    f"\tNon-Trainable parameters {get_human_readable_count(non_trainable_parameters)}"
    #)
    #logging.info("==" * 10)

    if log_dir:
        output_fpath = os.path.join(log_dir, "model.txt")
        with g_pathmgr.open(output_fpath, "w") as f:
            print(model, file=f)


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.
    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'
    Args:
        number: a positive integer number
    Return:
        A string formatted according to the pattern described above.
    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"
    else:
        return f"{number:,.1f} {labels[index]}"
