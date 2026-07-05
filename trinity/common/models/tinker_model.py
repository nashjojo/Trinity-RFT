import time
from os import getenv
from typing import List, Optional, Sequence

import ray
import tinker
import torch
from tinker import types
from torch import Tensor

from trinity.common.config import InferenceModelConfig
from trinity.common.constants import SyncMethod
from trinity.common.experience import Experience
from trinity.common.models.model import BaseInferenceModel
from trinity.manager.synchronizer import Synchronizer


class TinkerModel(BaseInferenceModel):
    def __init__(
        self,
        config: InferenceModelConfig,
    ) -> None:
        super().__init__(config)
        self.model_version = -1
        # Lazily resolve the Synchronizer actor: in ``trinity debug
        # --module inference_model`` (and other partial-launch flows) the
        # Synchronizer is not started, but TinkerModel itself only needs it
        # in ``sync_model_weights``. Looking it up at __init__ time would
        # crash the actor before it can serve any inference request.
        self.synchronizer = None
        self.model = None
        self.model_path = config.model_path

    def _get_synchronizer(self):
        if self.synchronizer is None:
            self.synchronizer = Synchronizer.get_actor(
                namespace=ray.get_runtime_context().namespace
            )
        return self.synchronizer

    async def _initialize_tokenizer(self) -> None:
        """Initialize the tokenizer."""
        if hasattr(self.model, 'get_tokenizer'):
            self.tokenizer = self.model.get_tokenizer()
        else:
            # Fallback: SamplingClient (newer Tinker SDK) doesn't expose
            # get_tokenizer(). Load directly via transformers.
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, trust_remote_code=True
            )


    async def _generate_internal(self, prompt: dict, **kwargs) -> types.SampleResponse:
        assert self.model is not None
        sampling_params = {
            "max_tokens": kwargs.get("max_tokens", self.config.max_response_tokens),
            "seed": kwargs.get("seed", self.config.seed),
            "temperature": kwargs.get("temperature", 1.0),
            "top_k": kwargs.get("top_k", -1),
            "top_p": kwargs.get("top_p", 1),
        }

        return await self.model.sample_async(
            prompt=types.ModelInput.from_ints(prompt["prompt_token_ids"]),
            sampling_params=sampling_params,
            num_samples=kwargs.get("n", 1),
            include_prompt_logprobs=kwargs.get("include_prompt_logprobs", False),
            topk_prompt_logprobs=kwargs.get("topk_prompt_logprobs", self.config.logprobs),
        )

    async def generate(self, prompt: str, **kwargs) -> Sequence[Experience]:
        """Generate a responses from a prompt in async."""
        if self.tokenizer is None:
            await self._initialize_tokenizer()

        returned_seq, is_valid = self._handle_prompt_truncation(prompt, **kwargs)
        if not is_valid:
            return returned_seq  # is_valid is False: returned_seq is a list of dummy experiences
        token_ids = returned_seq  # is_valid is True: returned_seq is prompt's token_ids

        with_chat_completion = kwargs.get("with_chat_completion", False)
        if with_chat_completion:
            create_time = int(time.time())
        output = await self._generate_internal(prompt={"prompt_token_ids": token_ids}, **kwargs)
        logprobs = kwargs.get("logprobs", self.config.logprobs)
        return_logprobs = logprobs is not None and logprobs is not False
        experiences = [
            Experience(
                tokens=torch.tensor(token_ids + sequence.tokens, dtype=torch.int32),
                logprobs=(
                    torch.tensor(sequence.logprobs, dtype=torch.float32)
                    if return_logprobs
                    else torch.tensor([], dtype=torch.float32)
                ),
                prompt_length=len(token_ids),
                prompt_text=self.tokenizer.decode(token_ids),
                response_text=self.tokenizer.decode(sequence.tokens),
            )
            for sequence in output.sequences
        ]
        if with_chat_completion:
            from openai.types.chat.chat_completion import (
                ChatCompletion,
                ChatCompletionMessage,
                ChatCompletionTokenLogprob,
                Choice,
                ChoiceLogprobs,
            )

            return_token_ids = kwargs.get("return_token_ids", False)
            chat_completion = ChatCompletion(
                id="",
                choices=[
                    Choice(
                        finish_reason=sequence.stop_reason,
                        index=i,
                        logprobs=ChoiceLogprobs(
                            content=[
                                ChatCompletionTokenLogprob(
                                    token=self.tokenizer.decode(token_id),
                                    logprob=logprob,
                                    top_logprobs=[],
                                )
                                for token_id, logprob in zip(sequence.tokens, sequence.logprobs)
                            ]
                        ),
                        message=ChatCompletionMessage(
                            content=self.tokenizer.decode(sequence.tokens), role="assistant"
                        ),
                        token_ids=(sequence.tokens if return_token_ids else None),
                    )
                    for i, sequence in enumerate(output.sequences)
                ],
                created=create_time,
                model=self.model_path,
                object="chat.completion",
                prompt_token_ids=token_ids,
            )
            experiences.append(chat_completion)

        return experiences

    async def chat(self, messages: List[dict], **kwargs) -> Sequence[Experience]:
        """Generate experiences from a list of history chat messages in async."""
        if self.tokenizer is None:
            await self._initialize_tokenizer()

        # TODO: this is a hack to support openai chat messages, which only supports text
        for msg in messages:
            if isinstance(msg["content"], list):
                text_parts = [item["text"] for item in msg["content"] if item["type"] == "text"]
                content_str = "".join(text_parts)
            else:
                content_str = msg["content"]
            msg["content"] = content_str

        prompt = self.apply_chat_template(self.tokenizer, messages)
        return await self.generate(prompt=prompt, **kwargs)

    async def logprobs(self, token_ids: List[int], **kwargs) -> Tensor:
        """Generate logprobs for a list of tokens in async."""
        logprobs = await self.model.compute_logprobs_async(types.ModelInput.from_ints(token_ids))
        return torch.tensor(logprobs[1:], dtype=torch.float32)

    async def prepare(self) -> None:
        """Prepare the model before inference."""
        self.service_client = tinker.ServiceClient()
        self.model = await self.service_client.create_sampling_client_async(
            base_model=self.config.model_path,
        )
        await self._initialize_tokenizer()

    async def sync_model_weights(
        self, model_version: int, sync_method: SyncMethod, timeout: float = 1200
    ) -> int:
        self.model_version = model_version
        synchronizer = self._get_synchronizer()
        remote_sampler_path, _ = await synchronizer.get_model_state_dict.remote()
        self.model = await self.service_client.create_sampling_client_async(
            model_path=remote_sampler_path,
        )
        self.model_path = remote_sampler_path
        return model_version

    def get_model_version(self) -> int:
        """Get the checkpoint version."""
        return self.model_version

    def get_api_server_url(self) -> Optional[str]:
        """
        Get the OpenAI-compatible API URL exposed by the Tinker-protocol service.

        - When a local TuFT server is used (``TINKER_BASE_URL`` is set, e.g.
          ``http://localhost:10610``), TuFT exposes the OpenAI endpoints under
          ``/oai/api/v1/{completions, chat/completions, models}`` (see TuFT
          ``src/tuft/oai/router.py``). Trinity workflows typically append ``/v1``
          to ``api_address``, so we return ``<base>/oai/api`` here so that the
          final URL becomes ``<base>/oai/api/v1`` and matches TuFT's router
          prefix. TuFT additionally resolves ``tinker://...`` model ids to the
          right LoRA adapter via dynamic LoRA loading on the underlying vLLM
          backend.
        - Otherwise (legacy public Tinker), fall back to the official URL,
          which is currently kept for documentation/back-compat only.

        Documentation: https://tinker-docs.thinkingmachines.ai/compatible-apis/openai
        """
        # Prefer TINKER_PUBLIC_URL for sandbox access (public IP),
        # fall back to TINKER_BASE_URL (may be internal IP for local use).
        public_url = getenv("TINKER_PUBLIC_URL")
        if public_url:
            return public_url.rstrip("/") + "/oai/api"
        base_url = getenv("TINKER_BASE_URL")
        if base_url:
            return base_url.rstrip("/") + "/oai/api"
        return "https://tinker.thinkingmachines.dev/services/tinker-prod/oai/api/"

    def get_api_key(self):
        return getenv("TINKER_API_KEY")

    def get_model_path(self) -> Optional[str]:
        """Get the latest sampler weight path."""
        return self.model_path
