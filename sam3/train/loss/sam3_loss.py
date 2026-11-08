# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

import logging
from typing import Optional

import torch

from sam3.model.model_misc import SAM3Output

from sam3.train.utils.distributed import get_world_size

from .loss_fns import CORE_LOSS_KEY, Det2TrkAssoc, Masks


def _log_non_finite_loss_terms(l_dict, loss_fn, suffix, is_aux, o2m=False):
    """(Sy) Added, 2026-07-17: name the exact offending loss term as soon as it goes
    non-finite, instead of only discovering "Loss is nan" much later at
    trainer.py's top-level `math.isfinite(loss.item())` crash check with no
    further detail. See .github/skills/loss-debugging/SKILL.md for context.

    Kept as a permanent (not one-off/temporary) diagnostic per explicit user
    decision, so future NaN/Inf investigations don't have to re-add this.
    """
    for key, value in l_dict.items():
        if not isinstance(value, torch.Tensor) or value.numel() == 0:
            continue
        if not torch.isfinite(value).all():
            logging.error(
                "Non-finite loss term detected: loss_fn=%s key=%s%s "
                "is_aux=%s o2m=%s value=%s",
                type(loss_fn).__name__,
                key,
                suffix,
                is_aux,
                o2m,
                value.detach().float().tolist() if value.numel() <= 8 else "<tensor too large to print>",
            )


def _count_matched_pairs(indices) -> int:
    """(Ha) Added, 2026-07-17: count total matched (prediction, target) pairs in a
    matcher's `indices` output, regardless of whether it's returned as a list of
    per-sample (idx_i, idx_j) tuples or a single (batch_idx, src_idx) tuple of
    concatenated tensors. Used only for the zero-loss diagnostic below.

    (Ha) Fixed, 2026-07-18: a per-sample entry can be `None` (negative sample,
    already guarded) OR a genuinely empty sequence (e.g. `()`/`[]`, not
    `None`) for a fully-negative sample -- indexing `pair[0]` on an empty
    `pair` raised `IndexError: index 0 is out of bounds for dimension 0 with
    size 0` and crashed training a second time. Now checks `len(pair) == 0`
    too, and wraps each per-sample count in a try/except so this diagnostic-
    only helper can never crash the training loop again regardless of what
    shape indices takes.
    """
    if indices is None:
        return 0
    if isinstance(indices, (list, tuple)) and len(indices) == 2 and all(
        hasattr(x, "numel") for x in indices
    ):
        # (batch_idx, src_idx) concatenated-tensor form
        return int(indices[0].numel())
    if isinstance(indices, (list, tuple)):
        # list of per-sample (idx_i, idx_j) tuples; a per-sample entry can be
        # None or empty for a negative sample with zero matched instances.
        total = 0
        for pair in indices:
            try:
                if pair is None or len(pair) == 0:
                    continue
                total += int(pair[0].numel()) if hasattr(pair[0], "numel") else len(pair[0])
            except Exception:
                continue
        return total
    return 0



class DummyLoss(torch.nn.Module):
    """A dummy loss that always returns 0 (as a placeholder for eval)"""

    def __init__(
        self,
        core_loss_key: str = CORE_LOSS_KEY,
        device: Optional[str] = None,  # default: cuda if available else cpu
        **kwargs,
    ):
        super().__init__()
        self.core_loss_key = core_loss_key
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def forward(self, *args, **kwargs):
        return {self.core_loss_key: torch.tensor(0.0, device=self.device)}

    def accumulate(self, out_dict):
        """
        Called by iterative losses.
        """
        if self.core_loss_key not in out_dict:
            out_dict[self.core_loss_key] = torch.tensor(0.0, device=self.device)
        return out_dict


class Sam3LossWrapper(torch.nn.Module):
    def __init__(
        self,
        loss_fns_find,
        normalization="global",
        matcher=None,
        o2m_matcher=None,
        o2m_weight=1.0,
        use_o2m_matcher_on_o2m_aux=True,
        loss_fn_semantic_seg=None,
        normalize_by_valid_object_num=False,
        normalize_by_stage_num=False,
        scale_by_find_batch_size=False,
    ):
        super().__init__()
        self.loss_fns_find = loss_fns_find
        assert normalization in ["global", "local", "none"]
        self.normalization = normalization
        self.normalize_by_valid_object_num = normalize_by_valid_object_num
        self.normalize_by_stage_num = normalize_by_stage_num
        self.matcher = matcher
        self.o2m_matcher = o2m_matcher
        self.o2m_weight = o2m_weight
        # whether to use the o2m matcher on the o2m queries in auxiliary outputs
        self.use_o2m_matcher_on_o2m_aux = use_o2m_matcher_on_o2m_aux
        self.loss_fn_semantic_seg = loss_fn_semantic_seg
        self.scale_by_find_batch_size = scale_by_find_batch_size
        # (Ha) Added, 2026-07-18: populated with the main stage's raw
        # pred_boxes/pred_logits/pred_masks/indices/target boxes whenever the
        # zero-loss diagnostic below fires, so trainer.py's anomalous-batch
        # dump can include what the MODEL actually output at the matched
        # indices -- not just the input images/targets it already captures.
        # This is the concrete next step to confirm/refute the "matched pairs
        # point at a zero-valued padding slot on both sides" hypothesis.
        self.last_zero_loss_debug = None

    def _get_num_boxes(self, targets):
        # the average number of target boxes for loss normalization
        if self.normalize_by_valid_object_num:
            # valid boxes are those with non-zero height and width
            # (while padded invisible boxes are )
            boxes_hw = targets["boxes"].view(-1, 4)  # cx, cy, w, h
            num_boxes = (boxes_hw[:, 2:] > 0).all(dim=-1).sum().float()
        else:
            num_boxes = targets["num_boxes"].sum().float()
        if self.normalization == "global":
            torch.distributed.all_reduce(num_boxes)
            num_boxes = torch.clamp(num_boxes / get_world_size(), min=1)
        elif self.normalization == "local":
            num_boxes = torch.clamp(num_boxes, min=1)
        elif self.normalization == "none":
            num_boxes = 1
        return num_boxes

    def compute_loss(self, nested_out, targets):
        num_boxes = self._get_num_boxes(targets)
        o2m_out_is_valid = nested_out.get("o2m_out_is_valid", None)
        o2m_target_is_valid_padded = nested_out.get("o2m_target_is_valid_padded", None)

        # Get a list of outputs, including auxiliary and first stage outputs
        output_list = [(nested_out, "", False)]  # (out, suffix, is_aux)
        if "aux_outputs" in nested_out:
            output_list.extend(
                (aux_out, f"_aux_{i}", True)
                for i, aux_out in enumerate(nested_out["aux_outputs"])
            )
        if "first_stage" in nested_out:
            output_list.append((nested_out["first_stage"], "_fs", True))

        # Compute all the requested losses
        losses = {}
        total_core_loss = 0.0
        # (Ha) Added, 2026-07-18: record each individual loss_fn's CORE_LOSS_KEY
        # contribution (per stage/is_aux/o2m), so the zero-loss diagnostic below
        # can report exactly WHICH term(s) are zero instead of only the total --
        # the total being non-negative-summed means the only way it can be
        # exactly 0.0 with real matched pairs is if every individual term is
        # independently 0.0, which this makes directly checkable.
        _core_contributions = []
        for out, suffix, is_aux in output_list:
            # o2o matcher indices need to be computed by the model (as the video model requires
            # a specific way of matching free and locked indices beyond just calling the matcher)
            indices = out["indices"]
            has_o2m_out = "pred_logits_o2m" in out
            if has_o2m_out:
                o2m_out = {
                    k[: -len("_o2m")]: v for k, v in out.items() if k.endswith("_o2m")
                }
                # o2m targets are the same as the o2o targets (assuming repeat=1)
                o2m_targets = targets
                if self.use_o2m_matcher_on_o2m_aux or not is_aux:
                    o2m_indices = self.o2m_matcher(
                        o2m_out,
                        o2m_targets,
                        out_is_valid=o2m_out_is_valid,
                        target_is_valid_padded=o2m_target_is_valid_padded,
                    )
                else:
                    o2m_indices = self.matcher(
                        o2m_out,
                        o2m_targets,
                        out_is_valid=o2m_out_is_valid,
                        target_is_valid_padded=o2m_target_is_valid_padded,
                    )

            for loss_fn in self.loss_fns_find:
                l_dict = loss_fn(
                    outputs=out,
                    targets=targets,
                    indices=indices,
                    num_boxes=num_boxes,
                    is_aux=is_aux,
                )
                _log_non_finite_loss_terms(l_dict, loss_fn, suffix, is_aux, o2m=False)
                _core_term = l_dict.pop(CORE_LOSS_KEY)
                _core_contributions.append(
                    (f"{type(loss_fn).__name__}{suffix}", "o2o",
                     _core_term.item() if isinstance(_core_term, torch.Tensor) else float(_core_term))
                )
                total_core_loss += _core_term
                losses.update({f"{k}{suffix}": v for k, v in l_dict.items()})

                compute_o2m_loss = has_o2m_out
                # a special handling to allow turning off mask loss in o2m
                # (to be compatible with the original implementation)
                if isinstance(loss_fn, Masks):
                    compute_o2m_loss = compute_o2m_loss and "pred_masks" in o2m_out
                if isinstance(loss_fn, Det2TrkAssoc):
                    compute_o2m_loss = False  # Det2TrkAssoc does not support o2m
                if compute_o2m_loss:
                    l_dict = loss_fn(
                        outputs=o2m_out,
                        targets=o2m_targets,
                        indices=o2m_indices,
                        num_boxes=num_boxes,
                        is_aux=is_aux,
                    )
                    _log_non_finite_loss_terms(l_dict, loss_fn, suffix, is_aux, o2m=True)
                    for k in l_dict:
                        l_dict[k] *= self.o2m_weight
                    _core_term = l_dict.pop(CORE_LOSS_KEY)
                    _core_contributions.append(
                        (f"{type(loss_fn).__name__}{suffix}", "o2m",
                         _core_term.item() if isinstance(_core_term, torch.Tensor) else float(_core_term))
                    )
                    total_core_loss += _core_term
                    losses.update({f"{k}{suffix}_o2m": v for k, v in l_dict.items()})

        # (Ha) Added, 2026-07-17: root-causing a silent training collapse where
        # total_core_loss was observed to be exactly 0.0 for an entire epoch
        # (see experiments/5cat_matrix_chip_v2, epoch 9 of job 237436875). Log
        # enough context on every zero-loss batch to tell apart the possible
        # causes: an all-negative batch (num_boxes==0, expected/benign given
        # remove_samples_with_0_gt=True on BinaryHungarianMatcherV2) vs. a
        # genuine matching/loss-weight bug (num_boxes>0 but 0 matched pairs, or
        # matched pairs>0 but still zero loss).
        _core_val = (
            total_core_loss.item()
            if isinstance(total_core_loss, torch.Tensor)
            else float(total_core_loss)
        )
        if _core_val == 0.0:
            _num_boxes_val = (
                num_boxes.item() if isinstance(num_boxes, torch.Tensor) else num_boxes
            )
            _targets_num_boxes_sum = (
                targets["num_boxes"].sum().item()
                if isinstance(targets, dict) and "num_boxes" in targets
                else "n/a"
            )
            _matched_by_stage = [
                _count_matched_pairs(out.get("indices")) for out, _, _ in output_list
            ]
            # (Ha) Added, 2026-07-18: break the total down per (loss_fn, o2o/o2m)
            # contribution, so we can see exactly which term(s) are the zero
            # ones instead of only knowing the sum is zero. Only report the
            # non-zero-value ones as "OK" implicitly by omission would hide
            # useful info, so log everything explicitly.
            _nonzero_terms = [(name, kind, val) for name, kind, val in _core_contributions if val != 0.0]
            _zero_terms = [(name, kind) for name, kind, val in _core_contributions if val == 0.0]
            logging.warning(
                "(Ha) Zero total_core_loss diagnostic: num_boxes(normalized)=%s "
                "targets_num_boxes_sum(raw)=%s n_stages=%s matched_pairs_per_stage=%s "
                "nonzero_contributions=%s all_zero_terms=%s",
                _num_boxes_val,
                _targets_num_boxes_sum,
                len(output_list),
                _matched_by_stage,
                _nonzero_terms if _nonzero_terms else "NONE -- every single term was exactly 0.0",
                _zero_terms,
            )

            # (Ha) Added, 2026-07-18: capture the main stage's raw model
            # output tensors AT THE MATCHED INDICES ONLY (not the full
            # per-query tensors), so trainer.py's anomalous-batch dump can
            # include what the model actually predicted -- confirms/refutes
            # whether matched pairs point at a zero-valued padding slot on
            # both sides (which would trivially make every loss term above
            # compute to exactly 0.0).
            #
            # (Ha) Fixed, 2026-07-18: the first version captured the FULL
            # pred_boxes/pred_logits/pred_masks tensors (all queries, not just
            # matched ones) via `.detach()` alone (no `.cpu()`), which kept
            # them GPU-resident and alive in `self.last_zero_loss_debug` until
            # trainer.py consumed them well after this step's normal tensors
            # would have been freed -- pred_masks in particular is
            # num_queries x H x W, which is large. This directly caused an
            # OOM crash (job 237913353, step 730) at the exact same moment as
            # a zero-loss event, on a config already at ~44/44.39GB with zero
            # headroom. Fix: slice down to just the matched (batch, query)
            # positions (a handful of rows, not the whole tensor) and move to
            # CPU immediately, right here, instead of leaving GPU references
            # alive across the boundary into trainer.py.
            try:
                main_out, _, _ = output_list[0]
                main_indices = main_out.get("indices")
                pairs = []
                if main_indices is not None:
                    if isinstance(main_indices, (list, tuple)) and len(main_indices) == 2 and all(
                        hasattr(x, "numel") for x in main_indices
                    ):
                        batch_idx, src_idx = main_indices
                        pairs = list(zip(batch_idx.tolist(), src_idx.tolist()))
                    elif isinstance(main_indices, (list, tuple)):
                        for b, pair in enumerate(main_indices):
                            if pair is None:
                                continue
                            idx_i = pair[0] if isinstance(pair, (list, tuple)) else pair
                            for q in (idx_i.tolist() if hasattr(idx_i, "tolist") else list(idx_i)):
                                pairs.append((b, q))
                pairs = pairs[:20]  # cap, this is a debug aid, not a full audit

                def _slice_matched(tensor):
                    if tensor is None or not pairs:
                        return None
                    return torch.stack(
                        [tensor[b, q].detach().cpu() for b, q in pairs if b < tensor.shape[0] and q < tensor.shape[1]]
                    )

                self.last_zero_loss_debug = {
                    "matched_pairs": pairs,
                    "matched_pred_boxes": _slice_matched(main_out.get("pred_boxes")),
                    "matched_pred_logits": _slice_matched(main_out.get("pred_logits")),
                    "matched_pred_masks": _slice_matched(main_out.get("pred_masks")),
                    "target_boxes": (
                        targets["boxes"].detach().cpu()
                        if isinstance(targets, dict) and targets.get("boxes") is not None
                        else None
                    ),
                }
            except Exception:
                logging.exception("(Ha) Failed to capture zero-loss debug tensors (non-fatal)")


        losses[CORE_LOSS_KEY] = total_core_loss
        return losses

    def forward(self, find_stages: SAM3Output, find_targets):
        if find_stages.loss_stages is not None:
            find_targets = [find_targets[i] for i in find_stages.loss_stages]
        with SAM3Output.iteration_mode(
            find_stages, iter_mode=SAM3Output.IterMode.ALL_STEPS_PER_STAGE
        ) as find_stages:
            assert len(find_stages) == len(find_targets)
            total_losses = {}
            for stage_outputs, stage_targets in zip(find_stages, find_targets):
                stage_targets = [stage_targets] * len(stage_outputs)
                # If there are multiple steps within a stage, compute the loss for all of them (e.g. interactivity)
                for outputs, targets in zip(stage_outputs, stage_targets):
                    cur_losses = self.compute_loss(outputs, targets)

                    if self.loss_fn_semantic_seg is not None:
                        cur_losses_semantic = self.loss_fn_semantic_seg(
                            outputs, targets
                        )
                        _log_non_finite_loss_terms(
                            cur_losses_semantic, self.loss_fn_semantic_seg, "", False
                        )
                        cur_losses[CORE_LOSS_KEY] += cur_losses_semantic.pop(
                            CORE_LOSS_KEY
                        )
                        # make sure the semantic losses don't overlap with the find losses
                        assert set(cur_losses).isdisjoint(set(cur_losses_semantic))
                        cur_losses.update(cur_losses_semantic)

                    # Optionally, normalize the loss by the number of find stages (training video frames) so that
                    # image batches and video batches have similar loss scales. (Otherwise video batches would
                    # have a much higher loss scale due to summing the losses over all the find stages.)
                    if self.normalize_by_stage_num:
                        cur_losses[CORE_LOSS_KEY] /= len(find_stages)

                    if self.scale_by_find_batch_size:
                        bs = targets["num_boxes"].shape[0]
                        # sqrt scaling based on the "effective" batch size
                        cur_losses[CORE_LOSS_KEY] *= bs**0.5

                    for k, v in cur_losses.items():
                        if k not in total_losses:
                            total_losses[k] = v
                        else:
                            total_losses[k] += v

        return total_losses
