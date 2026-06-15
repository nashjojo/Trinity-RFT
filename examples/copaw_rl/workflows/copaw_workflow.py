import time
from typing import List, Optional

import torch

from trinity.common.experience import Experience
from trinity.common.models.model import ModelWrapper
from trinity.common.workflows import WORKFLOWS
from trinity.common.workflows.workflow import MultiTurnWorkflow, Task


@WORKFLOWS.register_module("copaw_rl_workflow")
class CoPawRLWorkflow(MultiTurnWorkflow):
    def __init__(
        self,
        *,
        task: Task,
        model: ModelWrapper,
        auxiliary_models: Optional[List[ModelWrapper]] = None,
    ):
        super().__init__(
            task=task,
            model=model,
            auxiliary_models=auxiliary_models,
        )

    def run(self):
        from examples.copaw_rl.workflows.sandbox_utils import (
            get_or_create_sandbox,
            run_workflow,
        )

        start_time = time.time()
        sandbox_id = self.task.workflow_args.get("sandbox_id", None)
        token = self.task.workflow_args["token"]
        domain = self.task.workflow_args["domain"]
        template = self.task.workflow_args["template"]

        sandbox, created = get_or_create_sandbox(sandbox_id, token, domain, template, self.logger)
        sandbox_id = sandbox.sandbox_id

        oss_config = self.task.workflow_args["oss"]
        otel_config = self.task.workflow_args["otel"]
        dashscope_api_key = self.task.workflow_args["dashscope_api_key"]
        # When ``text_only`` is true (e.g. text-only models like
        # Qwen3-4B-Thinking served via tinker/TuFT), skip multimodal
        # processing so we don't need vllm / HF AutoProcessor at runtime.
        text_only = bool(self.task.workflow_args.get("text_only", False))
        task_id = self.task.raw_task["task_id"]
        api_server_url = f"{self.model.api_address}/v1"
        # For tinker/TuFT engines, ``model_path`` is a dynamic ``tinker://...``
        # sampler path (updated every weight sync) that TuFT's OAI router can
        # resolve to the latest LoRA adapter. For vLLM it falls back to
        # ``model_name`` (the HF model id), keeping previous behaviour.
        provider_model_id = self.model.model_path or self.model.model_name
        # The HF base model name is still needed for tokenizer / multi-modal
        # processor loading -- ``tinker://`` paths are not loadable by HF.
        hf_model_name = self.model.model_name
        try:
            dataset = run_workflow(
                sandbox,
                task_id,
                oss_config,
                otel_config,
                dashscope_api_key,
                api_server_url,
                provider_model_id,
                self.logger,
            )
        except Exception as e:
            self.logger.error(f"Error running workflow (ID: {sandbox_id}): {e}")
            raise e
        finally:
            sandbox.kill()

        exps = []
        processor = None
        vllm_processor = None
        if not text_only:
            # Lazy import to avoid pulling in vllm / transformers when
            # running text-only with tinker/TuFT.
            import transformers  # noqa: WPS433  (local import on purpose)

            from trinity.common.models.mm_utils import (  # noqa: WPS433
                ClientMultiModalProcessor,
            )

            vllm_processor = ClientMultiModalProcessor(model_name=hf_model_name)
        for data in dataset:
            prompt_token_ids = torch.tensor(data["prompt_token_ids"])
            response_token_ids = torch.tensor(data["token_ids"])
            token_ids = torch.cat([prompt_token_ids, response_token_ids])
            logprobs = torch.tensor(data["logprobs"])
            prompt_length = len(prompt_token_ids)
            action_mask = torch.tensor(data["response_mask"], dtype=torch.int)
            reward = float(data.get("reward", 0.0))
            metrics = {
                "reward": reward,
            }

            if text_only:
                multi_modal_inputs = None
            else:
                messages = data["messages"]
                _, mm_data, _ = vllm_processor.process_messages(messages)
                if mm_data is not None:
                    if processor is None:
                        processor = transformers.AutoProcessor.from_pretrained(hf_model_name)
                    multi_modal_inputs = {}
                    if images := mm_data.get("image", None):
                        images = [img.media for img in images]
                        image_inputs = processor.image_processor(
                            images=images, return_tensors="pt"
                        )
                        multi_modal_inputs.update(image_inputs)
                    if videos := mm_data.get("video", None):
                        videos = [vid.media for vid in videos]
                        video_inputs = processor.video_processor(
                            videos=videos, return_tensors="pt"
                        )
                        multi_modal_inputs.update(video_inputs)
                else:
                    multi_modal_inputs = None

            exp = Experience(
                tokens=token_ids,
                logprobs=logprobs,
                prompt_length=prompt_length,
                action_mask=action_mask,
                reward=reward,
                metrics=metrics,
                multi_modal_inputs=multi_modal_inputs,
            )
            exps.append(exp)
        del processor, vllm_processor

        self.logger.info(
            f"Workflow finished in {time.time() - start_time:.2f} seconds. Sandbox {'created' if created else 'connected'} "
            f"(ID: {sandbox_id}). Reward = {reward}. Collected {len(exps)} experiences."
        )

        return exps
