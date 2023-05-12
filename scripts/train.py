"""Run this script with 'torchrun'."""

import logging
import math
import os
import random
import shutil
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from functools import partial
from itertools import islice
from pathlib import Path
from typing import Any, Deque, Dict, Generator, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import wandb
from packaging import version
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, StateDictType
from torch.distributed.fsdp.api import (
    FullOptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
)
from torch.utils.data import DataLoader, DistributedSampler
from torchmetrics import MeanMetric, Metric

from olmo.config import (
    CheckpointType,
    DataConfig,
    EvaluatorConfig,
    ModelConfig,
    OptimizerType,
    SchedulerType,
    SpeedMonitorConfig,
    TrainConfig,
)
from olmo.data import DataCollator, IterableDataset, MemMapDataset
from olmo.downstream_eval import ICLMetric, label_to_task_map
from olmo.exceptions import OlmoCliError, OlmoConfigurationError
from olmo.model import Olmo
from olmo.optim import LionW, get_param_groups
from olmo.tokenizer import Tokenizer
from olmo.util import (
    clean_opt,
    global_rank,
    local_rank,
    log_extra_field,
    move_to_device,
    peak_gpu_memory,
    prepare_cli_environment,
    seed_all,
)

log = logging.getLogger("train")


@dataclass
class SpeedMonitor:
    cfg: SpeedMonitorConfig
    device_batch_num_tokens: int
    start_times: Deque[float] = field(default_factory=lambda: deque([]))
    global_step: int = 0

    def batch_start(self, global_step: int, record: bool = True) -> None:
        self.global_step = global_step
        if record:
            if len(self.start_times) >= self.cfg.window_size:
                self.start_times.popleft()
            self.start_times.append(time.monotonic())

    def reset(self) -> None:
        self.start_times.clear()

    def check(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {
            "throughput/total_tokens": self.global_step * self.device_batch_num_tokens * dist.get_world_size()
        }
        if self.start_times:
            interval_seconds = time.monotonic() - self.start_times[0]
            interval_batches = len(self.start_times)
            interval_tokens = self.device_batch_num_tokens * interval_batches
            metrics["throughput/device/tokens_per_second"] = interval_tokens / interval_seconds
            metrics["throughput/device/batches_per_second"] = interval_batches / interval_seconds
        return metrics


@dataclass
class LRMonitor:
    optim: torch.optim.Optimizer

    def check(self) -> Dict[str, float]:
        lrs = [group["lr"] for group in self.optim.param_groups]
        return {f"optim/learning_rate_group{idx}": lr for idx, lr in enumerate(lrs)}


@dataclass
class Evaluator:
    cfg: EvaluatorConfig
    eval_loader: DataLoader
    eval_batches: Iterator[Dict[str, Any]]
    eval_metric: Metric

    def reset_metrics(self) -> None:
        self.eval_metric.reset()

    def compute_metrics(self) -> Dict[str, float]:
        metric_val = self.eval_metric.compute()
        if isinstance(self.eval_metric, ICLMetric):
            # downstream eval
            return {
                f"eval/downstream/{self.cfg.label}_{self.eval_metric.metric_type}": metric_val.item(),
            }
        else:
            # metric_val is cross entropy loss
            loss = metric_val
            return {
                f"eval/{self.cfg.label}/CrossEntropyLoss": loss.item(),
                f"eval/{self.cfg.label}/Perplexity": torch.exp(loss).item(),
            }

    def update_metrics(
        self,
        batch: Dict[str, Any],
        loss: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        if isinstance(self.eval_metric, ICLMetric):
            # downstream eval
            self.eval_metric.update(batch, logits)  # type: ignore
        else:
            self.eval_metric.update(loss)


@dataclass
class Trainer:
    cfg: TrainConfig
    model: Olmo
    fsdp_model: FSDP
    optim: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    train_loader: DataLoader
    device: torch.device
    evaluators: List[Evaluator]
    ce_train_loss_metric: MeanMetric
    z_train_loss_metric: Optional[MeanMetric] = None
    global_step: int = 0
    global_data_step: int = 0
    checkpoints: List[Path] = field(default_factory=list)
    unsharded_checkpoints: List[Path] = field(default_factory=list)
    min_train_loss: float = float("inf")

    def state_dict(self) -> Dict[str, Any]:
        state_dict = self.non_tensor_state_dict()
        state_dict["model"] = self.fsdp_model.state_dict()
        state_dict["optim"] = FSDP.optim_state_dict(self.fsdp_model, self.optim)
        return state_dict

    def non_tensor_state_dict(self) -> Dict[str, Any]:
        return {
            "scheduler": self.scheduler.state_dict(),
            "global_step": self.global_step,  # move forward one batch
            "global_data_step": self.global_data_step,  # move forward one batch
            "checkpoints": self.checkpoints,
            "unsharded_checkpoints": self.unsharded_checkpoints,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(),
            },
        }

    def save_sharded_checkpoint(self) -> Path:
        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}"
        checkpoint_dir_tmp = Path(self.cfg.save_folder) / f"step{self.global_step}-tmp"

        try:
            next(checkpoint_dir.glob("*"))
            if self.cfg.save_overwrite:
                if global_rank() == 0:
                    shutil.rmtree(checkpoint_dir)
            else:
                raise OlmoConfigurationError(
                    f"Checkpoint for step {self.global_step} already exists, use --save-overwrite to overwrite it"
                )
        except StopIteration:
            pass

        if global_rank() == 0:
            checkpoint_dir_tmp.mkdir(parents=True, exist_ok=True)

        self.checkpoints.append(checkpoint_dir)
        dist.barrier()

        # Write the checkpoint.
        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            # NOTE: Alternatively we could use the checkpointing method in this test
            # https://github.com/pytorch/pytorch/blob/main/test/distributed/checkpoint/test_fsdp_optim_state.py
            # but we've had issues with that on AMD GPUs. See
            # https://github.com/pytorch/pytorch/issues/100041
            #  checkpoint.save_state_dict(self.state_dict(), checkpoint.FileSystemWriter(checkpoint_dir))
            torch.save(self.state_dict(), checkpoint_dir_tmp / f"rank{global_rank()}.pt")
            # Save config too.
            if global_rank() == 0:
                self.cfg.save(checkpoint_dir_tmp / "config.yaml")
            dist.barrier()

        if global_rank() == 0:
            # Replace temp directory with target checkpoint directory.
            checkpoint_dir_tmp.replace(checkpoint_dir)

            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / "latest"
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)

        # Remove old checkpoints.
        if self.cfg.save_num_checkpoints_to_keep > 0:
            while len(self.checkpoints) > self.cfg.save_num_checkpoints_to_keep:
                self.remove_sharded_checkpoint(0)

        dist.barrier()

        return checkpoint_dir

    def remove_sharded_checkpoint(self, idx: int = 0):
        oldest_checkpoint = self.checkpoints.pop(idx)
        dist.barrier()
        if global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
        dist.barrier()

    def restore_sharded_checkpoint(self, load_path: Path):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(offload_to_cpu=True),
            optim_state_dict_config=ShardedOptimStateDictConfig(offload_to_cpu=True),
        ):
            # NOTE: Alternatively we could use the checkpointing method in this test
            # https://github.com/pytorch/pytorch/blob/main/test/distributed/checkpoint/test_fsdp_optim_state.py
            # but we've had issues with that on AMD GPUs. See
            # https://github.com/pytorch/pytorch/issues/100041
            # But basically it would look like this.
            # Load the serialized state dict in place.
            #  state_dict = self.state_dict()
            #  del state_dict["optim"]  # Can't load optimizer together with the model
            #  checkpoint.load_state_dict(state_dict, checkpoint.FileSystemReader(load_path))
            #  self.fsdp_model.load_state_dict(state_dict["model"])
            # Load other state...
            # Load optim state.
            #  optim_state = load_sharded_optimizer_state_dict(
            #      model_state_dict=state_dict["model"],
            #      optimizer_key="optim",
            #      storage_reader=checkpoint.FileSystemReader(load_path),
            #  )
            #  flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], self.fsdp_model, self.optim)
            #  self.optim.load_state_dict(flattened_osd)

            # Deserialize state dictionary.
            state_dict = torch.load(load_path / f"rank{global_rank()}.pt")

            # Load state.
            self.fsdp_model.load_state_dict(state_dict["model"])
            self.global_step = state_dict["global_step"]
            self.global_data_step = state_dict["global_data_step"]
            self.checkpoints = [
                path
                for path in state_dict["checkpoints"]
                if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
            ]
            self.unsharded_checkpoints = [
                path
                for path in state_dict["unsharded_checkpoints"]
                if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder).resolve()
            ]
            self.scheduler.load_state_dict(state_dict["scheduler"])
            # NOTE: careful, the order of these arguments has changed since the 2.0 release.
            if version.parse(torch.__version__) < version.parse("2.1.0"):
                #  flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], self.fsdp_model, self.optim)  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(state_dict["optim"], self.fsdp_model, self.optim)  # type: ignore
            else:
                #  flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, optim_state["optim"])  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, state_dict["optim"])  # type: ignore
            self.optim.load_state_dict(flattened_osd)

            rng_state = state_dict.pop("rng")
            del state_dict, flattened_osd

        dist.barrier()

        if not self.cfg.restore_dataloader:
            self.global_data_step = 0
        elif self.cfg.fast_forward_batches:
            self.global_data_step += self.cfg.fast_forward_batches

        # Fast-forward data loader.
        if not self.cfg.dry_run:
            self.fast_forward_batches()
            dist.barrier()

        # Set rng state.
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        torch.cuda.set_rng_state(rng_state["cuda"])

    def save_unsharded_checkpoint(self) -> Path:
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        checkpoint_dir = Path(self.cfg.save_folder) / f"step{self.global_step}-unsharded"
        checkpoint_dir_tmp = Path(self.cfg.save_folder) / f"step{self.global_step}-unsharded-tmp"

        try:
            next(checkpoint_dir.glob("*"))
            if self.cfg.save_overwrite:
                if global_rank() == 0:
                    shutil.rmtree(checkpoint_dir)
            else:
                raise OlmoConfigurationError(
                    f"Unsharded checkpoint for step {self.global_step} already exists, use --save-overwrite to overwrite it"
                )
        except StopIteration:
            pass

        if global_rank() == 0:
            checkpoint_dir_tmp.mkdir(parents=True, exist_ok=True)

        self.unsharded_checkpoints.append(checkpoint_dir)
        dist.barrier()

        # Write the checkpoint.
        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            # We'll write the model and optimizer state dicts individually to reduce (CPU) memory consumption.
            # First the model state.
            model_state_dict = self.fsdp_model.state_dict()
            if global_rank() == 0:
                torch.save(model_state_dict, checkpoint_dir_tmp / "model.pt")
            del model_state_dict

            # Then the optimizer state.
            optim_state_dict = FSDP.optim_state_dict(self.fsdp_model, self.optim)
            if global_rank() == 0:
                torch.save(optim_state_dict, checkpoint_dir_tmp / "optim.pt")
            del optim_state_dict

            # Then everything else.
            other_state_dict = self.non_tensor_state_dict()
            if global_rank() == 0:
                torch.save(other_state_dict, checkpoint_dir_tmp / "other.pt")
                self.cfg.save(checkpoint_dir_tmp / "config.yaml")
            dist.barrier()

        if global_rank() == 0:
            # Replace temp directory with target checkpoint directory.
            checkpoint_dir_tmp.replace(checkpoint_dir)

            # Link to 'latest'.
            latest_path = Path(self.cfg.save_folder) / "latest-unsharded"
            latest_path.unlink(missing_ok=True)
            latest_path.symlink_to(checkpoint_dir.name, target_is_directory=True)

        # Remove old checkpoints.
        if self.cfg.save_num_unsharded_checkpoints_to_keep > 0:
            while len(self.unsharded_checkpoints) > self.cfg.save_num_unsharded_checkpoints_to_keep:
                self.remove_unsharded_checkpoint(0)

        dist.barrier()
        return checkpoint_dir

    def remove_unsharded_checkpoint(self, idx: int = 0):
        dist.barrier()
        oldest_checkpoint = self.unsharded_checkpoints.pop(idx)
        if global_rank() == 0 and oldest_checkpoint.is_dir():
            shutil.rmtree(oldest_checkpoint, ignore_errors=True)
        dist.barrier()

    def restore_unsharded_checkpoint(self, load_path: Path):
        # Zero-gradients to avoid gathering them.
        self.optim.zero_grad(set_to_none=True)

        with FSDP.state_dict_type(
            self.fsdp_model,
            state_dict_type=StateDictType.FULL_STATE_DICT,
            state_dict_config=FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            optim_state_dict_config=FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            # Load model state.
            self.fsdp_model.load_state_dict(torch.load(load_path / "model.pt"))

            # Load optimizer state.
            optim_state_dict = torch.load(load_path / "optim.pt")
            # NOTE: careful, the order of these arguments has changed since the 2.0 release.
            if version.parse(torch.__version__) < version.parse("2.1.0"):
                #  flattened_osd = FSDP.optim_state_dict_to_load(optim_state["optim"], self.fsdp_model, self.optim)  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(optim_state_dict, self.fsdp_model, self.optim)  # type: ignore
            else:
                #  flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, optim_state["optim"])  # type: ignore
                flattened_osd = FSDP.optim_state_dict_to_load(self.fsdp_model, self.optim, optim_state_dict)  # type: ignore
            del optim_state_dict
            self.optim.load_state_dict(flattened_osd)
            del flattened_osd

            # Load other state.
            other_state_dict = torch.load(load_path / "other.pt")
            self.global_step = other_state_dict["global_step"]
            self.global_data_step = other_state_dict["global_data_step"]
            self.checkpoints = [
                path
                for path in other_state_dict["checkpoints"]
                if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder)
            ]
            self.unsharded_checkpoints = [
                path
                for path in other_state_dict["unsharded_checkpoints"]
                if path.is_dir() and path.resolve().parent == Path(self.cfg.save_folder)
            ]
            self.scheduler.load_state_dict(other_state_dict["scheduler"])

        dist.barrier()

        if not self.cfg.restore_dataloader:
            self.global_data_step = 0
        elif self.cfg.fast_forward_batches:
            self.global_data_step += self.cfg.fast_forward_batches

        # Fast-forward data loader.
        if not self.cfg.dry_run:
            self.fast_forward_batches()
            dist.barrier()

    def fast_forward_batches(self):
        if self.global_data_step > 0:
            if self.global_data_step > self.global_step:
                log.info(
                    f"Fast-forwarding data loader to {self.global_step}+{self.global_data_step-self.global_step}"
                )
            else:
                log.info(f"Fast-forwarding data loader to {self.global_data_step}")
            assert isinstance(self.train_loader.dataset, IterableDataset)
            self.train_loader.dataset.start_step = self.global_data_step

    def save_checkpoint(self, checkpoint_type: CheckpointType = CheckpointType.sharded) -> Path:
        if checkpoint_type == CheckpointType.sharded:
            return self.save_sharded_checkpoint()
        elif checkpoint_type == CheckpointType.unsharded:
            return self.save_unsharded_checkpoint()
        else:
            raise NotImplementedError(checkpoint_type)

    def restore_checkpoint(self, load_path: Path, checkpoint_type: Optional[CheckpointType] = None):
        if checkpoint_type == CheckpointType.unsharded or (
            checkpoint_type is None and load_path.name.endswith("-unsharded")
        ):
            self.restore_unsharded_checkpoint(load_path)
        elif checkpoint_type == CheckpointType.sharded or checkpoint_type is None:
            self.restore_sharded_checkpoint(load_path)
        elif checkpoint_type is not None:
            raise NotImplementedError(checkpoint_type)

    def remove_checkpoint(self, idx: int = 0, checkpoint_type: CheckpointType = CheckpointType.sharded):
        if checkpoint_type == CheckpointType.sharded:
            self.remove_sharded_checkpoint(idx=idx)
        elif checkpoint_type == CheckpointType.unsharded:
            self.remove_unsharded_checkpoint(idx=idx)
        else:
            raise NotImplementedError(checkpoint_type)

    def get_labels(self, batch: Dict[str, Any]) -> torch.Tensor:
        # Labels are just input IDs shifted to the left (first item is ignored).
        labels, attention_mask = batch["input_ids"], batch.get("attention_mask")
        if attention_mask is not None:
            labels = labels.masked_fill(attention_mask == 0.0, -100)
        return labels[..., 1:].contiguous()

    def model_forward(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        # shape: (batch_size, seq_len, vocab_size)
        logits = self.fsdp_model(
            input_ids=batch["input_ids"],
            attention_mask=batch.get("attention_mask"),
            attention_bias=batch.get("attention_bias"),
        ).logits
        logits_for_loss = logits[..., :-1, :].contiguous()
        # shape: (batch_size * seq_len, vocab_size)
        logits_for_loss = logits_for_loss.view(-1, logits_for_loss.size(-1))
        # shape: (batch_size, seq_len)
        labels = self.get_labels(batch)
        # shape: (batch_size,)
        labels = labels.view(-1)
        ce_loss = F.cross_entropy(logits_for_loss, labels, ignore_index=-100)
        return ce_loss, logits

    def train_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Split into micro-batches.
        micro_batches = self.split_batch(batch)

        # In case this helps with memory utilization.
        del batch

        ce_batch_loss = torch.tensor(0.0, device=self.device)
        z_batch_loss = None if not cfg.softmax_auxiliary_loss else torch.tensor(0.0, device=self.device)
        for micro_batch in micro_batches:
            with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
                # Run forward pass.
                ce_loss, logits = self.model_forward(micro_batch)
                ce_loss = ce_loss / len(micro_batches)

                # In case this helps with memory utilization.
                del micro_batch

                # Update overall CE batch loss.
                ce_batch_loss += ce_loss.detach()

                # Get loss to optimize for.
                if self.cfg.softmax_auxiliary_loss:
                    z_squared = logits.logsumexp(-1).pow(2).mean()
                    z_loss = 1e-4 * z_squared / len(micro_batches)
                    loss = ce_loss + z_loss

                    # Update overall Z batch loss.
                    z_batch_loss += z_loss.detach()
                else:
                    loss = ce_loss

                del logits

            # Check for nan.
            if torch.isnan(loss):
                raise ValueError("nan loss encountered")

            # Run backward pass.
            loss.backward()

        return ce_batch_loss, z_batch_loss

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        # Zero-gradients.
        self.optim.zero_grad(set_to_none=True)

        # Reset metrics.
        self.ce_train_loss_metric.reset()
        if self.z_train_loss_metric is not None:
            self.z_train_loss_metric.reset()

        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward-backward pass.
        ce_batch_loss, z_batch_loss = self.train_batch(batch)

        # Clip gradient norms.
        grad_norm: Optional[float] = None
        if self.cfg.max_grad_norm is not None:
            grad_norm = self.fsdp_model.clip_grad_norm_(self.cfg.max_grad_norm).item()

        # Optimizer step.
        self.optim.step()
        self.scheduler.step()

        # Reduce loss metrics across ranks.
        self.ce_train_loss_metric.update(ce_batch_loss)
        ce_batch_loss = self.ce_train_loss_metric.compute()
        metrics = {
            "train/CrossEntropyLoss": ce_batch_loss.item(),
            "train/Perplexity": torch.exp(ce_batch_loss).item(),
        }
        if z_batch_loss is not None and self.z_train_loss_metric is not None:
            self.z_train_loss_metric.update(z_batch_loss)
            z_batch_loss = self.z_train_loss_metric.compute()
            metrics["train/ZLoss"] = z_batch_loss.item()

        if grad_norm is not None:
            metrics["optim/grad_norm"] = grad_norm

        # Update min train loss and see if we should stop early.
        self.min_train_loss = min(self.min_train_loss, ce_batch_loss.item())  # type: ignore
        if self.global_step > self.cfg.scheduler.t_warmup and ce_batch_loss.item() > 1.2 * self.min_train_loss:
            raise ValueError("Stopping early because train loss has increased substantially")

        return metrics

    def eval_batch(self, batch: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast("cuda", enabled=True, dtype=self.cfg.autocast_precision):
            ce_loss, logits = self.model_forward(batch)
        return ce_loss, logits

    def eval_step(self, batch: Dict[str, Any], evaluator: Evaluator) -> Dict[str, float]:
        # Move tensors to the right device.
        batch = move_to_device(batch, self.device)

        # Run forward pass.
        with torch.no_grad():  # NOTE: 'torch.inference_mode()' doesn't work with 'torch.compile()'.
            ce_loss, logits = self.eval_batch(batch)

        # Update metrics.
        evaluator.update_metrics(
            batch, ce_loss, logits
        )  # batch includes all keys that the downstream evaluation needs

        return evaluator.compute_metrics()

    def split_batch(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        batch_size = batch["input_ids"].shape[0]
        if batch_size <= self.cfg.device_train_microbatch_size:
            return [batch]
        else:
            micro_batches = {}
            for key, tensor in batch.items():
                micro_batches[key] = tensor.split(self.cfg.device_train_microbatch_size, dim=0)  # type: ignore
            return [
                {key: tensor[i] for key, tensor in micro_batches.items()}  # type: ignore
                for i in range(len(micro_batches["input_ids"]))
            ]

    def system_metrics(self) -> Dict[str, float]:
        metrics = {}
        peak_gpu_mb = peak_gpu_memory()
        if peak_gpu_mb is not None:
            metrics["System/Peak GPU Memory (MB)"] = peak_gpu_mb
        return metrics

    def log_metrics_to_console(self, prefix: str, metrics: Dict[str, float]):
        def format_float(value: float) -> str:
            if value < 0.0001:
                return str(value)  # scientific notation
            elif value > 1000:
                return f"{int(value):,d}"
            elif value > 100:
                return f"{value:.1f}"
            elif value > 10:
                return f"{value:.2f}"
            elif value > 1:
                return f"{value:.3f}"
            else:
                return f"{value:.4f}"

        log.info(
            f"{prefix}\n" + "\n".join([f"    {name}={format_float(value)}" for name, value in metrics.items()])
        )

    def should_log_this_step(self) -> bool:
        if self.global_step % self.cfg.console_log_interval == 0:
            return True
        elif self.cfg.wandb is not None and self.global_step % self.cfg.wandb.log_interval == 0:
            return True
        else:
            return False

    def eval(self) -> Dict[str, Any]:
        # Zero gradients and set model to 'eval' mode.
        self.optim.zero_grad(set_to_none=True)
        self.fsdp_model.eval()

        eval_metrics = {}
        for evaluator in self.evaluators:
            log.info(f"Running evaluation for '{evaluator.cfg.label}'...")

            # Reset metrics.
            evaluator.reset_metrics()

            # Check how many batches to evaluate on.
            num_eval_batches = evaluator.cfg.subset_num_batches
            if num_eval_batches is None:
                num_eval_batches = self.cfg.eval_subset_num_batches
            if num_eval_batches <= 0:
                num_eval_batches = max(1, len(evaluator.eval_loader))

            # Run model over batches.
            for eval_step, eval_batch in enumerate(islice(evaluator.eval_batches, num_eval_batches)):
                step_eval_metrics = self.eval_step(eval_batch, evaluator)

                # Log to console.
                if eval_step + 1 == num_eval_batches or (eval_step + 1) % self.cfg.console_log_interval == 0:
                    self.log_metrics_to_console(
                        f"[eval_step={eval_step + 1}/{num_eval_batches}]", step_eval_metrics
                    )

            # Get final metrics.
            eval_metrics.update(evaluator.compute_metrics())

        return eval_metrics

    def fit(self):
        if self.cfg.load_path is not None and self.global_step > 0:
            # Evaluate right away if we're loading from a checkpoint.
            eval_metrics = self.eval()

            # Log metrics to W&B.
            if wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)

        # Set model to 'train' mode.
        self.fsdp_model.train()

        # Initialize monitors.
        assert self.cfg.device_train_batch_size is not None
        speed_monitor = SpeedMonitor(
            self.cfg.speed_monitor,
            device_batch_num_tokens=self.cfg.device_train_batch_size * self.cfg.model.max_sequence_length,
        )
        lr_monitor = LRMonitor(self.optim)

        # Log system metrics at the start of training.
        sys_metrics = self.system_metrics()
        if sys_metrics:
            self.log_metrics_to_console("Pre-train system metrics", sys_metrics)
            if wandb.run is not None:
                wandb.log(sys_metrics, step=0)

        # Train.
        first_batch: bool = True
        for batch in self.train_loader:
            self.global_step += 1
            self.global_data_step += 1

            speed_monitor.batch_start(
                self.global_step,
                # We start monitoring speed after the first batch since the first
                # batch might be an outlier due to compiling and other initialization overhead.
                record=not first_batch,
            )

            # Run train step on batch.
            metrics = self.train_step(batch)

            # Maybe collect other metrics.
            if self.should_log_this_step():
                # Speed metrics.
                metrics.update(speed_monitor.check())
                # System metrics.
                metrics.update(self.system_metrics())
                # Learning rate metrics.
                metrics.update(lr_monitor.check())

            # Log metrics to console.
            if self.global_step % self.cfg.console_log_interval == 0:
                self.log_metrics_to_console(f"[step={self.global_step}/{self.cfg.max_duration}]", metrics)

            # Log metrics to W&B.
            if (
                wandb.run is not None
                and self.cfg.wandb is not None
                and self.global_step % self.cfg.wandb.log_interval == 0
            ):
                wandb.log(metrics, step=self.global_step)

            # Maybe save sharded checkpoint.
            if self.global_step % self.cfg.save_interval == 0 and self.cfg.save_num_checkpoints_to_keep != 0:
                log.info("Saving checkpoint...")
                checkpoint_path = self.save_sharded_checkpoint()
                log.info(f"Checkpoint saved to {checkpoint_path}")

                # Reset speed monitor so that we don't count the time taken to save checkpoints.
                speed_monitor.reset()

            # Maybe save unsharded checkpoint.
            if (
                self.cfg.save_interval_unsharded is not None
                and self.global_step % self.cfg.save_interval_unsharded == 0
                and self.cfg.save_num_unsharded_checkpoints_to_keep != 0
            ):
                log.info("Saving unsharded checkpoint...")
                checkpoint_path = self.save_unsharded_checkpoint()
                log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

                # Reset speed monitor so that we don't count the time taken to save checkpoints.
                speed_monitor.reset()

            # Maybe run evaluations.
            if self.global_step % self.cfg.eval_interval == 0:
                eval_metrics = self.eval()

                # Log metrics to W&B.
                if wandb.run is not None:
                    wandb.log(eval_metrics, step=self.global_step)

                # Reset speed monitor so that we don't count the time taken to run evaluations.
                speed_monitor.reset()

                # Reset model to 'train' mode.
                self.fsdp_model.train()

            # End of batch.
            first_batch = False

        # Save final unsharded model-only checkpoint.
        log.info("Saving final unsharded model checkpoint...")
        checkpoint_path = self.save_unsharded_checkpoint()
        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

    def close(self) -> None:
        if wandb.run is not None:
            wandb.finish()


def main(cfg: TrainConfig) -> None:
    # Ensure run name set.
    if cfg.run_name is None:
        cfg.run_name = os.environ.get("COMPOSER_RUN_NAME", "train-llm")
    log_extra_field("run_name", cfg.run_name)

    # Initialize process group and set device.
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(f"cuda:{local_rank()}")
    device = torch.device("cuda")

    # Fill some configuration options.
    cfg.model.precision = cfg.precision
    cfg.device_train_batch_size = cfg.global_train_batch_size // dist.get_world_size()
    assert cfg.device_train_batch_size is not None  # for mypy
    cfg.device_train_grad_accum = cfg.device_train_batch_size // cfg.device_train_microbatch_size

    # Display and save configuration.
    if global_rank() == 0:
        log.info("Configuration:")
        log.info(cfg)
        if not cfg.dry_run and (cfg.load_path is None or Path(cfg.load_path).parent != Path(cfg.save_folder)):
            # Save config.
            save_path = Path(cfg.save_folder) / "config.yaml"
            if save_path.is_file() and not cfg.save_overwrite:
                raise OlmoConfigurationError(f"{save_path} already exists, use --save_overwrite to overwrite")
            else:
                log.info(f"Saving config to {save_path}")
                save_path.parent.mkdir(exist_ok=True, parents=True)
                cfg.save(save_path)
            del save_path

    dist.barrier()

    # Set seed.
    seed_all(cfg.seed)

    # Maybe start W&B run.
    if cfg.wandb is not None and (global_rank() == 0 or not cfg.wandb.rank_zero_only):
        wandb_dir = Path(cfg.save_folder) / "wandb"
        wandb_dir.mkdir(parents=True, exist_ok=True)
        wandb.init(
            dir=wandb_dir,
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            group=cfg.wandb.group,
            name=cfg.wandb.name,
            tags=cfg.wandb.tags,
            config=cfg.asdict(exclude=["wandb"]),
        )

    dist.barrier()

    # Initialize the model.
    log.info("Initializing model...")
    olmo_model = Olmo(cfg.model)
    log.info(f"Total number of parameters: {olmo_model.num_params():,d}")
    log.info(f"Number of non-embedding parameters: {olmo_model.num_params(include_embedding=False):,d}")

    # Wrap the model in FSDP.
    fsdp_model = FSDP(
        olmo_model,
        sharding_strategy=cfg.fsdp.sharding_strategy,
        mixed_precision=MixedPrecision(  # equivalent to MosaicML's "PURE"
            param_dtype=cfg.autocast_precision,
            reduce_dtype=cfg.autocast_precision,
            buffer_dtype=cfg.autocast_precision,
        ),
        auto_wrap_policy=olmo_model.fsdp_wrap_fn,
        use_orig_params=cfg.fsdp.use_orig_params,  # needed for compile
        limit_all_gathers=True,
        device_id=local_rank(),
    )

    if cfg.activation_checkpointing:
        # verify we have FSDP activation support ready by importing:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointImpl,
            apply_activation_checkpointing,
            checkpoint_wrapper,
        )

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            offload_to_cpu=False,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,  # type: ignore
            check_fn=olmo_model.activation_checkpointing_fn,  # type: ignore
        )

    log.info("Model:")
    log.info(fsdp_model)

    # Construct optimizer and learning rate scheduler.
    optim = build_optimizer(cfg, fsdp_model)
    scheduler = build_scheduler(cfg, optim)

    # Construct data loader.
    train_loader = build_train_dataloader(cfg.data, cfg.model, cfg.device_train_batch_size)

    # Construct evaluators.
    evaluators = []
    tokenizer = None
    for eval_cfg in cfg.evaluators:
        evaluator: Evaluator
        if eval_cfg.data.paths:
            # Language modeling evaluation.
            eval_loader = build_eval_dataloader(
                eval_cfg.data,
                cfg.model,
                eval_cfg.device_eval_batch_size or cfg.device_eval_batch_size,
            )
            evaluator = Evaluator(
                cfg=eval_cfg,
                eval_loader=eval_loader,
                eval_batches=cycle_through_epochs(eval_loader),
                eval_metric=MeanMetric(nan_strategy="error").to(device),
            )
        elif eval_cfg.label in label_to_task_map:
            # Downstream evaluation.
            if tokenizer is None:
                tokenizer = Tokenizer.from_train_config(cfg)
            evaluator = build_downstream_evaluator(eval_cfg, cfg, tokenizer, device)
        else:
            raise OlmoConfigurationError(f"Not sure how to build evaluator for {eval_cfg}")
        evaluators.append(evaluator)

    # Consolidate components into `Trainer` object.
    trainer = Trainer(
        cfg=cfg,
        model=olmo_model,
        fsdp_model=fsdp_model,
        optim=optim,
        scheduler=scheduler,
        train_loader=train_loader,
        device=device,
        ce_train_loss_metric=MeanMetric(nan_strategy="error").to(device),
        z_train_loss_metric=None
        if not cfg.softmax_auxiliary_loss
        else MeanMetric(nan_strategy="error").to(device),
        evaluators=evaluators,
    )

    if not cfg.dry_run and cfg.load_path is None:
        checkpoint_type = (
            CheckpointType.sharded if cfg.save_num_checkpoints_to_keep != 0 else CheckpointType.unsharded
        )

        # We save a checkpoint up-front to make sure this won't fail (due to disk space or whatever).
        log.info("Saving pre-train checkpoint...")
        checkpoint_path = trainer.save_checkpoint(checkpoint_type=checkpoint_type)
        log.info(f"Checkpoint saved to {checkpoint_path}")

        # And they we verify that we can load it.
        log.info("Attempting to load pre-train checkpoint...")
        trainer.restore_checkpoint(checkpoint_path, checkpoint_type=checkpoint_type)
        log.info("Checkpoint successfully loaded")

        # But now we can remove it so we don't take up unnecessary space.
        log.info("Removing pre-train checkpoint...")
        trainer.remove_checkpoint(checkpoint_type=checkpoint_type)
        log.info("Successfully removed checkpoint")

    if cfg.load_path is not None:
        log.info(f"Loading checkpoint from {cfg.load_path}...")
        trainer.restore_checkpoint(Path(cfg.load_path))
        log.info("Checkpoint successfully loaded")

    if cfg.force_save_unsharded:
        log.info("Saving unsharded checkpoint...")
        checkpoint_path = trainer.save_unsharded_checkpoint()
        log.info(f"Unsharded checkpoint saved to {checkpoint_path}")

    if cfg.compile is not None:
        # NOTE: trying to compile the whole train step results in a compile-time error from within
        # the optimizer. We should investigate this further at some point.
        #  trainer.train_step = torch.compile(trainer.train_step, **cfg.compile.asdict())
        trainer.train_batch = torch.compile(trainer.train_batch, **cfg.compile.asdict())  # type: ignore
        trainer.eval_batch = torch.compile(trainer.eval_batch, **cfg.compile.asdict())  # type: ignore
        # Alternatively, could just do this:
        #  trainer.fsdp_model = torch.compile(trainer.fsdp_model, **cfg.compile.asdict())

    if not cfg.dry_run:
        log.info("Starting training...")
        trainer.fit()
    else:
        log.info("Dry run complete")

    trainer.close()


def build_eval_dataloader(
    data_config: DataConfig, model_config: ModelConfig, batch_size: int, shuffle: bool = True
) -> DataLoader:
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=model_config.pad_token_id)
    dataset = MemMapDataset(*data_config.paths, chunk_size=model_config.max_sequence_length)
    sampler = DistributedSampler(
        dataset,
        drop_last=data_config.drop_last,
        shuffle=shuffle,
        num_replicas=dist.get_world_size(),
        rank=global_rank(),
        seed=cfg.seed,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        sampler=sampler,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=cfg.data.persistent_workers,
        timeout=cfg.data.timeout,
    )


def build_train_dataloader(data_config: DataConfig, model_config: ModelConfig, batch_size: int) -> DataLoader:
    collator = DataCollator(pad_direction=data_config.pad_direction, pad_token_id=model_config.pad_token_id)
    dataset = MemMapDataset(*data_config.paths, chunk_size=model_config.max_sequence_length)
    return DataLoader(
        IterableDataset(
            dataset,  # type: ignore
            seed=cfg.seed,
            shuffle=True,
            drop_last=data_config.drop_last,
            max_steps=cfg.max_duration,
        ),
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        prefetch_factor=cfg.data.prefetch_factor,
        persistent_workers=cfg.data.persistent_workers,
        timeout=cfg.data.timeout,
    )


def build_downstream_evaluator(
    eval_cfg: EvaluatorConfig,
    train_config: TrainConfig,
    tokenizer: Tokenizer,
    device: torch.device,
    is_unit_test=False,
) -> Evaluator:
    task_class = label_to_task_map[eval_cfg.label]
    ds_eval_dataset = task_class(tokenizer=tokenizer)  # type: ignore
    data_config = eval_cfg.data
    if is_unit_test:
        ds_eval_sampler = None
    else:
        ds_eval_sampler = DistributedSampler(
            ds_eval_dataset,
            drop_last=data_config.drop_last,
            shuffle=False,
            num_replicas=dist.get_world_size(),
            rank=global_rank(),
            seed=cfg.seed,
        )
    ds_eval_dataloader = DataLoader(
        ds_eval_dataset,
        batch_size=eval_cfg.device_eval_batch_size or train_config.device_eval_batch_size,
        collate_fn=ds_eval_dataset.collate_fn,
        num_workers=data_config.num_workers,
        sampler=ds_eval_sampler,
        pin_memory=data_config.pin_memory,
        prefetch_factor=data_config.prefetch_factor,
        persistent_workers=data_config.persistent_workers,
        timeout=data_config.timeout,
    )
    metric = ICLMetric(metric_type=ds_eval_dataset.metric_type)

    evaluator = Evaluator(
        cfg=eval_cfg,
        eval_loader=ds_eval_dataloader,
        eval_batches=cycle_through_epochs(ds_eval_dataloader),
        eval_metric=metric.to(device),
    )
    return evaluator


def build_optimizer(cfg: TrainConfig, model: nn.Module) -> torch.optim.Optimizer:
    params = (
        get_param_groups(model)
        if (cfg.optimizer.no_decay_norm_and_bias and cfg.optimizer.weight_decay > 0.0)
        else model.parameters()
    )
    if cfg.optimizer.name == OptimizerType.lionw:
        return LionW(
            params,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == OptimizerType.adam:
        return torch.optim.Adam(
            params,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    elif cfg.optimizer.name == OptimizerType.adamw:
        return torch.optim.AdamW(
            params,
            lr=cfg.optimizer.learning_rate,
            betas=cfg.optimizer.betas,
            weight_decay=cfg.optimizer.weight_decay,
        )
    else:
        raise NotImplementedError


def build_scheduler(cfg: TrainConfig, optim: torch.optim.Optimizer) -> torch.optim.lr_scheduler.LRScheduler:
    schedulers: List[torch.optim.lr_scheduler.LRScheduler] = []
    if cfg.scheduler.name == SchedulerType.cosine_with_warmup:
        milestones = [cfg.scheduler.t_warmup]
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=cfg.scheduler.alpha_f, end_factor=1.0, total_iters=cfg.scheduler.t_warmup
            )
        ]
        if cfg.scheduler.t_max is None:
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                cfg.max_duration - cfg.scheduler.t_warmup,
                eta_min=cfg.optimizer.learning_rate * cfg.scheduler.alpha_f,
            )
            schedulers.append(cosine)
        else:
            milestones.append(cfg.scheduler.t_max)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optim,
                cfg.scheduler.t_max - cfg.scheduler.t_warmup,
                eta_min=cfg.optimizer.learning_rate * cfg.scheduler.alpha_f,
            )
            linear = torch.optim.lr_scheduler.LinearLR(
                optim,
                start_factor=cfg.scheduler.alpha_f,
                end_factor=cfg.scheduler.alpha_f**2,
                total_iters=cfg.max_duration - cfg.scheduler.t_max,
            )
            schedulers.append(cosine)
            schedulers.append(linear)
        return torch.optim.lr_scheduler.SequentialLR(optim, schedulers, milestones)
    elif cfg.scheduler.name == SchedulerType.inverse_sqrt_with_warmup:
        milestones = [cfg.scheduler.t_warmup]
        schedulers = [
            torch.optim.lr_scheduler.LinearLR(
                optim, start_factor=cfg.scheduler.alpha_f, end_factor=1.0, total_iters=cfg.scheduler.t_warmup
            ),
            torch.optim.lr_scheduler.LambdaLR(optim, lambda step: 1.0 if step <= 0 else 1.0 / math.sqrt(step)),
        ]
        return torch.optim.lr_scheduler.SequentialLR(optim, schedulers, milestones)
    else:
        raise NotImplementedError


def cycle_through_epochs(dataloader: DataLoader) -> Generator[Dict[str, Any], None, None]:
    while True:
        for batch in dataloader:
            yield batch

        if isinstance(dataloader.sampler, DistributedSampler):
            epoch = dataloader.sampler.epoch + 1
            dataloader.sampler.set_epoch(epoch)


if __name__ == "__main__":
    prepare_cli_environment()

    try:
        yaml_path, args_list = sys.argv[1], sys.argv[2:]
    except IndexError:
        raise OlmoCliError(f"Usage: {sys.argv[0]} [CONFIG_PATH] [OPTIONS]")

    cfg = TrainConfig.load(yaml_path, [clean_opt(s) for s in args_list])
    main(cfg)
