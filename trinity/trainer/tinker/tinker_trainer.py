import asyncio
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import ray
import tinker
import torch
from tinker import types

from trinity.algorithm import ALGORITHM_TYPE
from trinity.algorithm.advantage_fn import ADVANTAGE_FN
from trinity.algorithm.entropy_loss_fn import ENTROPY_LOSS_FN
from trinity.algorithm.entropy_loss_fn.entropy_loss_fn import DummyEntropyLossFn
from trinity.algorithm.kl_fn import KL_FN
from trinity.algorithm.kl_fn.kl_fn import DummyKLFn
from trinity.algorithm.policy_loss_fn import POLICY_LOSS_FN
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import Config
from trinity.common.experience import Experience
from trinity.manager.synchronizer import Synchronizer
from trinity.trainer.tinker.utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    to_tinker_input,
)
from trinity.trainer.trainer import TrainEngineWrapper
from trinity.utils.log import get_logger
from trinity.utils.timer import Timer

# Upper bound on concurrent reference-logprob requests per train step. The Tinker
# service persists each pending future's payload to Redis, so an unbounded gather
# over a full rollout batch spikes Redis and host memory without adding throughput
# once the sampling engines are saturated. Each item is one sampling request
# (max_tokens=1 + include_prompt_logprobs); override via
# TINKER_REF_LOGPROB_CONCURRENCY (e.g. 1536) to overlap more per-request latency.
REF_LOGPROB_CONCURRENCY = int(os.getenv("TINKER_REF_LOGPROB_CONCURRENCY", "512"))

# Mirrors tinker/lib/public_interfaces/training_client.py, which greedily packs each
# call's datums into sub-requests of at most MAX_CHUNK_BYTES / MAX_CHUNK_LEN and then
# yields whatever remains -- as few as a single datum. A backend that shards data
# across ranks has to reject that: with fewer datums than ranks, some ranks idle while
# the others block forever in NCCL collectives. Packing to a budget below the SDK's
# limit makes every chunk exactly one sub-request, so Trinity controls each shard size.
TINKER_MAX_CHUNK_BYTES = 5_000_000
TINKER_MAX_CHUNK_LEN = 1024
# to_tinker_input builds each Datum from model_input (total_length tokens) plus
# weights and target_tokens (total_length - 1 entries each), and the SDK estimates 10
# bytes per element of each of those three.
TINKER_BYTES_PER_TOKEN = 30
# The server-loss datum carries six loss_fn_inputs arrays (target_tokens, weights,
# logprobs, advantages, ref_logprobs, mask) instead of two, so the SDK's byte
# estimator (10 bytes per element per array + model input) sees ~70 bytes per token.
SERVER_LOSS_BYTES_PER_TOKEN = 70
MIN_CHUNK_DATUMS = 4
# Garbage-collecting the previous sampler checkpoint must never stall training: the
# SDK retries 5xx responses for up to two hours, which wedges the loop if the server
# answers one of these deletes with 500.
STALE_SAMPLER_DELETE_TIMEOUT = 60.0


def _run_id_from_tinker_path(path: str) -> str:
    """Extract the run id from ``tinker://<run_id>:train:<n>/...``."""
    return path.split("://", 1)[1].split(":", 1)[0]


class TinkerTrainerWrapper(TrainEngineWrapper):
    def __init__(self, config: Config):
        self.config = config
        self.logger = get_logger("tinker_trainer")
        self._init_algorithm()
        self.synchronizer = Synchronizer.get_actor(namespace=self.config.synchronizer.ray_namespace)
        # When set, _loss_func divides by this instead of the current call's datum
        # count. Required for chunked accumulation: each chunk's gradient must be
        # scaled by 1/total_datums so the sum equals the full-batch mean gradient.
        self._loss_scale_denominator: Optional[int] = None
        tinker_cfg = self.config.model.tinker
        self._bytes_per_token = (
            SERVER_LOSS_BYTES_PER_TOKEN if tinker_cfg.server_loss_fn else TINKER_BYTES_PER_TOKEN
        )
        self._max_sdk_chunk_bytes = TINKER_MAX_CHUNK_BYTES
        if tinker_cfg.sdk_chunk_bytes_cap:
            # Raise the SDK's client-side sub-request byte limit so a pre-packed chunk
            # can travel as one request. Requires a server with no body-size cap.
            from tinker.lib.public_interfaces import training_client as sdk_training_client

            sdk_training_client.MAX_CHUNK_BYTES_COUNT = tinker_cfg.sdk_chunk_bytes_cap
            self._max_sdk_chunk_bytes = tinker_cfg.sdk_chunk_bytes_cap
            self.logger.info(
                f"Raised Tinker SDK chunk byte cap to {tinker_cfg.sdk_chunk_bytes_cap}"
            )

    def _init_algorithm(self):
        self.algorithm = ALGORITHM_TYPE.get(self.config.algorithm.algorithm_type)
        self.algorithm_config = algorithm_config = self.config.algorithm
        if self.algorithm.compute_advantage_in_trainer:
            self.advantage_fn = ADVANTAGE_FN.get(algorithm_config.advantage_fn)(
                **algorithm_config.advantage_fn_args
            )
            self.kl_fn = KL_FN.get(algorithm_config.kl_penalty_fn)(
                **algorithm_config.kl_penalty_fn_args
            )
            # TODO
            raise NotImplementedError(
                "`compute_advantage_in_trainer` is not implemented yet in tinker"
            )
        self.loss_agg_mode = algorithm_config.loss_agg_mode
        self.policy_loss_fn = POLICY_LOSS_FN.get(algorithm_config.policy_loss_fn)(
            backend="tinker", **algorithm_config.policy_loss_fn_args
        )
        self.kl_loss_fn = KL_FN.get(algorithm_config.kl_loss_fn)(**algorithm_config.kl_loss_fn_args)
        # Reference logprobs feed ONLY the KL loss term. When that term is provably
        # zero, fetching them is pure waste (measured ~18% of step time, plus it
        # holds a backend sampler slot). This does not affect the policy gradient:
        # GRPO's importance ratio uses old_logprob, which arrives free with each
        # rollout experience. The kl_penalty_fn needs no consideration here because
        # it is only built in the compute_advantage_in_trainer branch above, which
        # raises NotImplementedError for this backend.
        self._reference_needed = self.algorithm.use_reference and not (
            isinstance(self.kl_loss_fn, DummyKLFn) or self.kl_loss_fn.kl_coef == 0.0
        )
        if self.algorithm.use_reference and not self._reference_needed:
            self.logger.info(
                "Skipping reference logprobs: the KL loss term is identically zero "
                f"(kl_loss_fn={algorithm_config.kl_loss_fn}, "
                f"kl_coef={getattr(self.kl_loss_fn, 'kl_coef', None)})."
            )
        self.entropy_loss_fn = ENTROPY_LOSS_FN.get(algorithm_config.entropy_loss_fn)(
            **algorithm_config.entropy_loss_fn_args
        )

        # EXPERIMENTAL: apply loss scale fix
        self.do_fix_actor_microbatch_loss_scale = (
            self.config.trainer.fix_actor_microbatch_loss_scale
            and (self.loss_agg_mode == "token-mean")
        )

        self.lr_scheduler_type = algorithm_config.optimizer.lr_scheduler_type
        self.total_steps = self.config.trainer.total_steps or sys.maxsize
        self.num_warmup_steps = algorithm_config.optimizer.lr_warmup_steps
        if self.num_warmup_steps < 0:
            self.num_warmup_steps = int(
                algorithm_config.optimizer.lr_warmup_steps_ratio * self.total_steps
            )
        self.min_lr_ratio = algorithm_config.optimizer.min_lr_ratio
        assert 0.0 <= self.min_lr_ratio <= 1.0
        self.logger.info(
            f"Total steps: {self.total_steps if self.total_steps != sys.maxsize else 'unlimited'},"
            f" num_warmup_steps: {self.num_warmup_steps}"
        )

        if self.lr_scheduler_type not in {"constant", "cosine"}:
            raise NotImplementedError(
                f"LR scheduler type {self.lr_scheduler_type} is not supported"
            )

    @property
    def _current_lr_factor(self):
        train_step_num = self._train_step_num
        # warmup
        if train_step_num < self.num_warmup_steps:
            factor = float(train_step_num) / float(max(1.0, self.num_warmup_steps))
            factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * factor
            return factor

        # decay
        if train_step_num >= self.total_steps:
            progress = 1.0
        else:
            progress = float(train_step_num - self.num_warmup_steps) / float(
                max(1.0, self.total_steps - self.num_warmup_steps)
            )
        if self.lr_scheduler_type == "constant":
            factor = 1.0
        elif self.lr_scheduler_type == "cosine":
            num_cycles = 0.5  # TODO: may add to config
            factor = 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
        factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * factor
        return max(self.min_lr_ratio, factor)

    @property
    def current_learning_rate(self):
        return self._current_lr_factor * self.algorithm_config.optimizer.lr

    @property
    def adam_params(self):
        return types.AdamParams(
            learning_rate=self.current_learning_rate,
            beta1=self.algorithm_config.optimizer.betas[0],
            beta2=self.algorithm_config.optimizer.betas[1],
            # eps is currently not in config
            weight_decay=self.algorithm_config.optimizer.weight_decay,
            grad_clip_norm=self.config.trainer.grad_clip,
        )

    async def prepare(self):
        # Full-param save_weights_for_sampler / ref-client calls can block on the ~115s
        # sampler redeploy, exceeding the tinker SDK's default 60s read timeout.
        self.service_client = tinker.ServiceClient(timeout=float(os.getenv("TINKER_TIMEOUT", "600")))
        self.checkpoint_manager = self.service_client.create_rest_client()
        name_prefix_list = [self.config.project, self.config.group, self.config.name]
        self.tinker_checkpoint_name_prefix = "-".join(
            [prefix for prefix in name_prefix_list if prefix]
        )
        self.default_local_dir = self.config.checkpoint_job_dir

        self.local_latest_checkpointed_iteration = os.path.join(
            self.config.checkpoint_job_dir, "latest_checkpointed_iteration.txt"
        )
        self.local_latest_state_dict_iteration = os.path.join(
            self.config.checkpoint_job_dir, "latest_state_dict_iteration.txt"
        )

        if os.path.exists(self.local_latest_checkpointed_iteration):
            with open(self.local_latest_checkpointed_iteration, "r") as f:
                self._train_step_num = self.latest_remote_checkpoint_step = int(f.read().strip())
            checkpoint_file_path = os.path.join(
                self.default_local_dir,
                f"global_step_{self._train_step_num}",
                "remote_checkpoint_path.txt",
            )
            with open(checkpoint_file_path, "r") as f:
                self.latest_remote_checkpoint_path = f.read().strip()
            tinker_cfg = self.config.model.tinker
            if tinker_cfg.full_param:
                # The SDK's create_training_client_from_state_with_optimizer_async asserts
                # weights_info.is_lora (LoRA-only), so it cannot resume a full-param run.
                # Create the full-param client directly (same as a fresh start) and load the
                # checkpoint state + optimizer from the tinker:// path, which encodes the
                # source run id, so a new run can load a previous run's checkpoint.
                self.logger.info(
                    f"Resuming full-param training from {self.latest_remote_checkpoint_path}"
                )
                self.actor_client = await self.service_client.create_lora_training_client_async(
                    base_model=self.config.model.model_path,
                    rank=tinker_cfg.rank,
                    seed=tinker_cfg.seed,
                    train_mlp=tinker_cfg.train_mlp,
                    train_attn=tinker_cfg.train_attn,
                    train_unembed=tinker_cfg.train_unembed,
                    user_metadata={"training_mode": "full_param"},
                )
                load_future = await self.actor_client.load_state_with_optimizer_async(
                    self.latest_remote_checkpoint_path
                )
                await load_future.result_async()
            else:
                self.actor_client = (
                    await self.service_client.create_training_client_from_state_with_optimizer_async(
                        path=self.latest_remote_checkpoint_path,
                    )
                )
        else:
            tinker_cfg = self.config.model.tinker
            user_metadata = {"training_mode": "full_param"} if tinker_cfg.full_param else None
            self.logger.info(
                f"Tinker training mode: {'full_param' if tinker_cfg.full_param else f'lora(rank={tinker_cfg.rank})'}"
            )
            self.actor_client = await self.service_client.create_lora_training_client_async(
                base_model=self.config.model.model_path,
                rank=tinker_cfg.rank,
                seed=tinker_cfg.seed,
                train_mlp=tinker_cfg.train_mlp,
                train_attn=tinker_cfg.train_attn,
                train_unembed=tinker_cfg.train_unembed,
                user_metadata=user_metadata,
            )
            self.latest_remote_checkpoint_step = 0
            self.latest_remote_checkpoint_path = None
            self._train_step_num = 0
        self.model_info = await self.actor_client.get_info_async()

        if os.path.exists(self.local_latest_state_dict_iteration):
            with open(self.local_latest_state_dict_iteration, "r") as f:
                self.latest_remote_sampler_step = int(f.read().strip())
            sampler_file_path = os.path.join(
                self.default_local_dir,
                f"global_step_{self.latest_remote_sampler_step}",
                "remote_sampler_path.txt",
            )
            with open(sampler_file_path, "r") as f:
                self.latest_remote_sampler_path = f.read().strip()
        else:
            self.latest_remote_sampler_step = None
            self.latest_remote_sampler_path = None

        if self._reference_needed:
            self.ref_client = await self.service_client.create_sampling_client_async(
                base_model=self.config.model.model_path,
            )
        else:
            self.ref_client = None

    @property
    def train_step_num(self) -> int:
        """Get the current training step number."""
        return self._train_step_num

    def _loss_func(
        self, batch: list[types.Datum], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        return self._loss_func_impl(self.model_inputs_list, logprobs)

    def _loss_func_impl(
        self, model_inputs_list: List[dict], logprobs: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = 0.0
        metrics = {}
        assert len(model_inputs_list) == len(
            logprobs
        ), "len(model_inputs_list) must equal to len(logprobs)"
        for model_inputs, logprob in zip(model_inputs_list, logprobs):
            micro_batch_metrics = {}
            response_mask = model_inputs["action_mask"]
            logprob = logprob[-response_mask.shape[0] :]

            pg_loss, pg_loss_metrics = self.policy_loss_fn(logprob=logprob, **model_inputs)
            prefix_metrics(
                src_metrics=pg_loss_metrics, prefix="actor", dst_metrics=micro_batch_metrics
            )

            if self.entropy_loss_fn != DummyEntropyLossFn:
                entropy = -(logprob * logprob.exp())
            else:
                entropy = None
            # compute entropy loss from entropy
            entropy_loss, entropy_loss_metrics = self.entropy_loss_fn(  # type: ignore
                entropy=entropy,
                **model_inputs,
                loss_agg_mode=self.loss_agg_mode,
            )
            prefix_metrics(
                src_metrics=entropy_loss_metrics,
                prefix="actor",
                dst_metrics=micro_batch_metrics,
            )

            # compute kl loss
            if "ref_logprob" in model_inputs:
                kl_loss, kl_loss_metrics = self.kl_loss_fn.calculate_kl_loss(
                    logprob=logprob,
                    ref_logprob=model_inputs["ref_logprob"],
                    response_mask=response_mask,
                    loss_agg_mode=self.loss_agg_mode,
                    old_logprob=model_inputs["old_logprob"],
                )
            else:
                # Reference logprobs were not fetched because the KL term is
                # identically zero (see _reference_needed), so this substitutes
                # exactly what calculate_kl_loss would have returned.
                kl_loss = torch.zeros_like(pg_loss)
                kl_loss_metrics = {}
            prefix_metrics(
                src_metrics=kl_loss_metrics,
                prefix="actor",
                dst_metrics=micro_batch_metrics,
            )

            # compute policy loss
            policy_loss = pg_loss - entropy_loss + kl_loss
            loss_scale = 1.0
            if not self.do_fix_actor_microbatch_loss_scale:
                loss_scale /= self._loss_scale_denominator or len(logprobs)
            loss = policy_loss * loss_scale
            total_loss = total_loss + loss
            micro_batch_metrics["actor/final_loss"] = loss.detach().item()

            # update metrics
            for key, val in micro_batch_metrics.items():
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(val)

        avg_metrics = {k: sum(v) / len(v) for k, v in metrics.items()}
        return total_loss, avg_metrics

    def _datum_request_bytes(self, total_length: int) -> int:
        """Estimate a datum's serialized size the same way the Tinker SDK does."""
        return self._bytes_per_token * total_length - 20

    def _pack_chunk_bounds(self, lengths: List[int], budget: int) -> List[Tuple[int, int]]:
        """Pack ascending-sorted datum lengths into ``(start, end)`` chunk bounds.

        Each chunk stays under the SDK's own byte limit so the SDK emits it as a single
        sub-request instead of re-splitting it and leaving a too-small remainder.
        """
        largest = self._datum_request_bytes(lengths[-1])
        # Reserve room to fold an undersized tail into the preceding chunk without
        # pushing that chunk over the SDK limit.
        usable = min(budget, self._max_sdk_chunk_bytes - (MIN_CHUNK_DATUMS - 1) * largest)
        usable = max(usable, largest)

        bounds: List[Tuple[int, int]] = []
        start = 0
        chunk_bytes = 0
        for i, length in enumerate(lengths):
            datum_bytes = self._datum_request_bytes(length)
            if i > start and (
                chunk_bytes + datum_bytes > usable or i - start == TINKER_MAX_CHUNK_LEN
            ):
                bounds.append((start, i))
                start = i
                chunk_bytes = 0
            chunk_bytes += datum_bytes
        bounds.append((start, len(lengths)))

        if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < MIN_CHUNK_DATUMS:
            tail_end = bounds.pop()[1]
            bounds.append((bounds.pop()[0], tail_end))
        return bounds

    def _with_server_loss_inputs(self, datum: types.Datum, model_inputs: dict) -> types.Datum:
        """Attach per-token old logprobs / advantages / ref logprobs for a server-side loss.

        Arrays are padded to the datum length (one shorter than the full sequence) with
        zeros; values occupy the trailing ``response_length`` positions, which is the
        same alignment the client-side loss uses (``logprob[-response_length:]``).
        """
        total_length = model_inputs["total_length"]
        response_length = model_inputs["action_mask"].shape[0]

        def padded(value) -> torch.Tensor:
            out = torch.zeros(total_length - 1, dtype=torch.float32)
            if value is not None:
                out[-response_length:] = torch.as_tensor(value, dtype=torch.float32)
            return out

        loss_fn_inputs = dict(datum.loss_fn_inputs)
        loss_fn_inputs["logprobs"] = padded(model_inputs.get("old_logprob"))
        loss_fn_inputs["advantages"] = padded(model_inputs.get("advantages"))
        if self.config.model.tinker.server_loss_send_mask:
            # Response mask so the backend reproduces the client-side masked token-mean
            # exactly (per-datum denominator = number of response tokens, not full length).
            loss_fn_inputs["mask"] = padded(model_inputs["action_mask"])
        if "ref_logprob" in model_inputs:
            loss_fn_inputs["ref_logprobs"] = padded(model_inputs["ref_logprob"])
        return types.Datum(model_input=datum.model_input, loss_fn_inputs=loss_fn_inputs)

    def _server_loss_fn_config(self, num_total_datums: int) -> dict:
        config = dict(self.config.model.tinker.server_loss_fn_config or {})
        policy_args = self.algorithm_config.policy_loss_fn_args or {}
        config.setdefault("clip_range", policy_args.get("clip_range", 0.2))
        config.setdefault("clip_ratio_c", policy_args.get("clip_ratio_c", 3.0))
        config.setdefault("kl_coef", self.kl_loss_fn.kl_coef)
        # The backend normalizes like the client-side path: sum over datums of each
        # datum's masked token-mean, divided by the total datum count of the batch.
        config["num_total_datums"] = num_total_datums
        return config

    async def _update_actor_with_server_loss(
        self,
        batch: List[types.Datum],
        model_inputs_list: List[dict],
        chunk_budget: Optional[int],
        metrics: Dict,
    ) -> None:
        """Chunked update where the loss is computed by the backend (``server_loss_fn``).

        Removes the client-side custom-loss protocol, whose first forward pass exists
        only to fetch logprobs for the client; here old logprobs travel inside each
        datum and the backend computes the GRPO loss during its own forward.
        """
        assert len(batch) == len(model_inputs_list)
        batch = [
            self._with_server_loss_inputs(datum, model_inputs)
            for datum, model_inputs in zip(batch, model_inputs_list)
        ]
        loss_fn = self.config.model.tinker.server_loss_fn
        num_total_datums = len(batch)

        if chunk_budget is None:
            fwdbwd_future = await self.actor_client.forward_backward_async(
                batch, loss_fn, self._server_loss_fn_config(num_total_datums)
            )
            optim_future = await self.actor_client.optim_step_async(self.adam_params)
            fwdbwd_result = await fwdbwd_future
            optim_result = await optim_future
            metrics.update(fwdbwd_result.metrics or {})
            if optim_result.metrics:
                metrics.update(optim_result.metrics)
            return

        # Sort by token length so each chunk is length-homogeneous, which keeps the
        # backend's per-micro-batch padding close to zero.
        order = sorted(range(len(batch)), key=lambda i: model_inputs_list[i]["total_length"])
        batch = [batch[i] for i in order]
        model_inputs_list = [model_inputs_list[i] for i in order]
        self.model_inputs_list = model_inputs_list

        lengths = [m["total_length"] for m in model_inputs_list]
        bounds = self._pack_chunk_bounds(lengths, chunk_budget)
        accumulated: Dict[str, List[float]] = {}
        try:
            async def run_chunk(start: int, end: int):
                future = await self.actor_client.forward_backward_async(
                    batch[start:end], loss_fn, self._server_loss_fn_config(num_total_datums)
                )
                return await future

            inflight = max(1, self.config.model.tinker.forward_backward_max_inflight)
            if inflight == 1 or len(bounds) == 1:
                chunk_results = [await run_chunk(s, e) for s, e in bounds]
            else:
                sem = asyncio.Semaphore(inflight)

                async def run_one(start: int, end: int):
                    async with sem:
                        return await run_chunk(start, end)

                chunk_results = await asyncio.gather(*[run_one(s, e) for s, e in bounds])
            for chunk_result in chunk_results:
                for key, val in (chunk_result.metrics or {}).items():
                    accumulated.setdefault(key, []).append(val)
        finally:
            self.model_inputs_list = model_inputs_list

        optim_future = await self.actor_client.optim_step_async(self.adam_params)
        optim_result = await optim_future
        for key, vals in accumulated.items():
            metrics[key] = sum(vals) / len(vals)
        if optim_result.metrics:
            metrics.update(optim_result.metrics)
        metrics["actor/num_forward_backward_chunks"] = float(len(bounds))

    async def train_step(self, batch_exps: List[Experience]) -> Dict:
        """Training one step.

        Args:
            batch (List[Experience]): A batch of experiences to train.

        Returns:
            Dict: Metrics of the training step.
        """
        batch, batch_input_tokens, model_inputs_list = to_tinker_input(batch_exps, self.logger)
        self.model_inputs_list = model_inputs_list
        timing_raw = {}
        metrics = {}
        self._train_step_num += 1

        with Timer(timing_raw, "step"):
            if self._reference_needed:
                with Timer(timing_raw, "ref"):
                    ref_logprobs: List = []
                    for start in range(0, len(batch_input_tokens), REF_LOGPROB_CONCURRENCY):
                        chunk = batch_input_tokens[start : start + REF_LOGPROB_CONCURRENCY]
                        ref_logprobs.extend(
                            await asyncio.gather(
                                *[self.ref_client.compute_logprobs_async(t) for t in chunk]
                            )
                        )
                for model_inputs, ref_logprob in zip(model_inputs_list, ref_logprobs):
                    response_length = model_inputs["action_mask"].shape[0]
                    model_inputs["ref_logprob"] = torch.tensor(ref_logprob[-response_length:])

            if self.algorithm.compute_advantage_in_trainer:
                # TODO: following is verl format, which is not compatible with tinker
                raise NotImplementedError(
                    "`compute_advantage_in_trainer` is not implemented yet in tinker"
                )
            else:
                # skip token_level_scores for sft/dpo
                for model_inputs in model_inputs_list:
                    if "token_level_scores" in model_inputs:
                        assert "token_level_rewards" not in model_inputs
                        model_inputs["token_level_rewards"] = model_inputs["token_level_scores"]

            # update actor
            with Timer(timing_raw, "update_actor"):
                chunk_budget = self.config.model.tinker.forward_backward_chunk_bytes
                if self.config.model.tinker.server_loss_fn:
                    await self._update_actor_with_server_loss(
                        batch, model_inputs_list, chunk_budget, metrics
                    )
                elif chunk_budget is None:
                    fwdbwd_future = await self.actor_client.forward_backward_custom_async(
                        batch, self._loss_func
                    )
                    optim_future = await self.actor_client.optim_step_async(self.adam_params)
                    fwdbwd_result = await fwdbwd_future
                    optim_result = await optim_future
                    metrics.update(fwdbwd_result.metrics)
                    if optim_result.metrics:
                        metrics.update(optim_result.metrics)
                else:
                    # Sort by token length so each chunk is length-homogeneous, which
                    # keeps the backend's per-micro-batch padding close to zero.
                    order = sorted(
                        range(len(batch)), key=lambda i: model_inputs_list[i]["total_length"]
                    )
                    batch = [batch[i] for i in order]
                    model_inputs_list = [model_inputs_list[i] for i in order]
                    self.model_inputs_list = model_inputs_list

                    # Gradients accumulate across forward_backward calls and are only
                    # zeroed by optim_step, so N chunks followed by ONE optim_step is a
                    # single full-batch update. _loss_scale_denominator makes each chunk
                    # normalize by the total datum count rather than its own length.
                    lengths = [m["total_length"] for m in model_inputs_list]
                    bounds = self._pack_chunk_bounds(lengths, chunk_budget)
                    accumulated: Dict[str, List[float]] = {}
                    self._loss_scale_denominator = len(batch)
                    try:
                        # Each chunk gets its own loss closure bound to its slice of
                        # model_inputs, so chunks can be in flight concurrently without
                        # racing on self.model_inputs_list.
                        def make_chunk_loss(inputs_slice: List[dict]):
                            def _chunk_loss(_batch, logprobs):
                                return self._loss_func_impl(inputs_slice, logprobs)

                            return _chunk_loss

                        async def run_chunk(start: int, end: int):
                            future = await self.actor_client.forward_backward_custom_async(
                                batch[start:end], make_chunk_loss(model_inputs_list[start:end])
                            )
                            return await future

                        inflight = max(1, self.config.model.tinker.forward_backward_max_inflight)
                        if inflight == 1 or len(bounds) == 1:
                            chunk_results = [await run_chunk(s, e) for s, e in bounds]
                        else:
                            sem = asyncio.Semaphore(inflight)

                            async def run_one(start: int, end: int):
                                async with sem:
                                    return await run_chunk(start, end)

                            chunk_results = await asyncio.gather(
                                *[run_one(s, e) for s, e in bounds]
                            )
                        for chunk_result in chunk_results:
                            for key, val in (chunk_result.metrics or {}).items():
                                accumulated.setdefault(key, []).append(val)
                    finally:
                        self._loss_scale_denominator = None
                        self.model_inputs_list = model_inputs_list

                    optim_future = await self.actor_client.optim_step_async(self.adam_params)
                    optim_result = await optim_future
                    for key, vals in accumulated.items():
                        metrics[key] = sum(vals) / len(vals)
                    if optim_result.metrics:
                        metrics.update(optim_result.metrics)
                    metrics["actor/num_forward_backward_chunks"] = float(len(bounds))

        # collect metrics
        metrics.update(compute_data_metrics(batch=self.model_inputs_list))
        timing_metrics = compute_timing_metrics(batch=self.model_inputs_list, timing_raw=timing_raw)
        metrics.update({k.replace("timing_s/", "time/"): v for k, v in timing_metrics.items()})
        metrics.update(
            compute_throughout_metrics(batch=self.model_inputs_list, timing_raw=timing_raw)
        )

        return metrics

    async def save_checkpoint(
        self, block_until_saved: bool = False, save_as_hf: bool = False
    ) -> None:
        """Save the checkpoint."""
        if self.train_step_num == self.latest_remote_checkpoint_step:
            return
        self.latest_remote_checkpoint_step = self.train_step_num
        checkpoint_name = f"{self.tinker_checkpoint_name_prefix}-state-{self.train_step_num}"
        save_state_future = await self.actor_client.save_state_async(checkpoint_name)
        self.latest_remote_checkpoint_path = (await save_state_future).path
        local_path = os.path.join(
            self.default_local_dir,
            f"global_step_{self.train_step_num}",
        )
        os.makedirs(local_path, exist_ok=True)

        # save a flag to indicate this is a full checkpoint dir
        # make sure this flag is created before notifying the synchronizer
        # to avoid the synchronizer recognizing it as a state_dict-only checkpoint
        # TODO: use a better way to indicate full checkpoint
        flag_path = os.path.join(local_path, ".full_checkpoint")
        with open(flag_path, "w") as f:
            f.write("")

        remote_checkpoint_path = os.path.join(local_path, "remote_checkpoint_path.txt")
        with open(remote_checkpoint_path, "w") as f:
            f.write(self.latest_remote_checkpoint_path)

        with open(self.local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.train_step_num))

    def sync_weight_nccl(self) -> None:
        """Sync the model weight."""
        raise NotImplementedError("Tinker trainer does not support NCCL sync")

    async def get_weight_sync_info(self):
        return None, None, []

    async def setup_weight_sync_group(
        self, master_address, master_port, world_size, group_name, timeout
    ):
        pass

    async def teardown_weight_sync_group(self):
        pass

    async def upload_state_dict(self) -> None:
        """Upload the state dict to Synchronizer."""
        await self.save_state_dict()
        ray.get(
            self.synchronizer.set_model_state_dict.remote(
                self.latest_remote_sampler_path, self.train_step_num
            )
        )

    async def save_state_dict(self) -> None:
        """Only save the model state dict for Synchronizer."""
        if self.train_step_num == self.latest_remote_sampler_step:
            return
        self.stale_remote_sampler_step = self.latest_remote_sampler_step
        self.latest_remote_sampler_step = self.train_step_num
        stale_sampler_path = self.latest_remote_sampler_path
        current_checkpoint_name = (
            f"{self.tinker_checkpoint_name_prefix}-sampler-{self.train_step_num}"
        )
        save_weights_future = await self.actor_client.save_weights_for_sampler_async(
            current_checkpoint_name
        )
        self.latest_remote_sampler_path = (await save_weights_future).path
        if self.stale_remote_sampler_step is not None:
            stale_checkpoint_name = (
                f"{self.tinker_checkpoint_name_prefix}-sampler-{self.stale_remote_sampler_step}"
            )
            # After resuming from another run's checkpoint the recorded sampler path
            # belongs to that source run; this run cannot delete it, and retrying the
            # cross-run delete forever wedges training (server answers 500).
            if (
                stale_sampler_path is not None
                and _run_id_from_tinker_path(stale_sampler_path) != self.model_info.model_id
            ):
                self.logger.info(
                    f"Skipping stale sampler checkpoint of another run: {stale_sampler_path}"
                )
            else:
                try:
                    await asyncio.wait_for(
                        self.checkpoint_manager.delete_checkpoint_async(
                            self.model_info.model_id, stale_checkpoint_name
                        ),
                        timeout=STALE_SAMPLER_DELETE_TIMEOUT,
                    )
                except Exception:
                    self.logger.warning(
                        f"Failed to remove stale state_dict {stale_checkpoint_name}"
                    )
        local_path = os.path.join(
            self.default_local_dir,
            f"global_step_{self.train_step_num}",
        )
        os.makedirs(local_path, exist_ok=True)
        remote_sampler_path = os.path.join(local_path, "remote_sampler_path.txt")
        with open(remote_sampler_path, "w") as f:
            f.write(self.latest_remote_sampler_path)

        with open(self.local_latest_state_dict_iteration, "w") as f:
            f.write(str(self.train_step_num))
