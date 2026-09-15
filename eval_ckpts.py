"""Offline test-set eval of tinker sampler checkpoints, replicating
StepWiseAlfworldWorkflow's protocol exactly (system prompt, format_observation,
parse_action, max 30 env steps). Reports per-checkpoint episode success rate on
the 140-game test split, 2 stochastic repeats per game (temperature 1.0, matching
the training rollout so numbers are comparable to the train-rollout trend).
"""
import asyncio
import json
import os
import sys
import time

import textworld
import textworld.gym
from alfworld.agents.environment.alfred_tw_env import (
    AlfredDemangler,
    AlfredExpert,
    AlfredExpertType,
)
from tinker import types
from transformers import AutoTokenizer

import tinker
from trinity.common.workflows.envs.alfworld.alfworld_workflow import (
    AlfWORLD_SYSTEM_PROMPT,
    format_observation,
    parse_action,
)

BASE = os.path.dirname(os.path.abspath(__file__))
CK = os.environ.get(
    "EVAL_CK_DIR",
    os.path.join(BASE, "checkpoints/ALFWORLD/Step_Wise_Alfworld_TuFT_8GPU_v3"),
)
TEST = os.path.join(BASE, "examples/grpo_alfworld/alfworld_data/test.jsonl")
MAX_STEPS = 30
# Keep prompts under the 8192 context minus the 512 response budget; the training
# chain applies the same cap (model.max_prompt_tokens), so eval and training
# truncate overlong histories identically instead of failing vLLM's length check.
MAX_PROMPT = int(os.environ.get("EVAL_MAX_PROMPT", "7679"))
_SYS_TOKENS = None
REPEATS = int(os.environ.get("EVAL_REPEATS", "2"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "32"))


def make_env(game):
    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
    ri = textworld.EnvInfos(description=True, inventory=True, admissible_commands=True)
    eid = textworld.gym.register_game(game, ri, wrappers=[AlfredDemangler(), expert])
    return textworld.gym.make(eid)


def _render_prompt(tok, mem):
    global _SYS_TOKENS
    prompt = tok.apply_chat_template(
        mem, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok.encode(prompt)
    if _SYS_TOKENS is None:
        _SYS_TOKENS = len(
            tok.encode(
                tok.apply_chat_template(
                    [mem[0]], tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
            )
        )
    if len(ids) > MAX_PROMPT:
        # keep the pinned system prompt + the most recent turns
        ids = ids[:_SYS_TOKENS] + ids[-(MAX_PROMPT - _SYS_TOKENS):]
    return ids


# textworld's tatsu PDDL parser holds shared module-level state, so env.reset /
# env.step must not run concurrently across episodes.
ENV_LOCK = asyncio.Lock()


async def run_one(sampler, tok, game, seed, sem):
    try:
        return await _run_one_inner(sampler, tok, game, seed, sem)
    except Exception as e:  # one broken episode must not kill the whole eval
        print(f"[warn] episode failed: {type(e).__name__}: {str(e)[:160]}", flush=True)
        return 0.0


async def _run_one_inner(sampler, tok, game, seed, sem):
    async with sem:
        async with ENV_LOCK:
            env = await asyncio.to_thread(make_env, game)
            obs, _ = await asyncio.to_thread(env.reset)
        mem = [{"role": "system", "content": AlfWORLD_SYSTEM_PROMPT}]
        done = False
        rew = 0.0
        try:
            for _ in range(MAX_STEPS):
                mem.append({"role": "user", "content": format_observation(obs)})
                ids = await asyncio.to_thread(_render_prompt, tok, mem)
                r = await sampler.sample_async(
                    prompt=types.ModelInput.from_ints(ids),
                    sampling_params={
                        "max_tokens": 512,
                        "temperature": 1.0,
                        "top_k": -1,
                        "top_p": 1,
                        "seed": seed,
                    },
                    num_samples=1,
                )
                txt = await asyncio.to_thread(tok.decode, r.sequences[0].tokens)
                mem.append({"role": "assistant", "content": txt})
                async with ENV_LOCK:
                    obs, rew, done, _ = await asyncio.to_thread(env.step, parse_action(txt))
                if done:
                    break
        finally:
            async with ENV_LOCK:
                await asyncio.to_thread(env.close)
        return 1.0 if rew > 0 else 0.0


async def main():
    games = [json.loads(l)["game_file"] for l in open(TEST)]
    tok = AutoTokenizer.from_pretrained("/mnt/workspace/kaixiang/models/Qwen3-1.7B")
    client = tinker.ServiceClient()
    sem = asyncio.Semaphore(CONCURRENCY)
    print(f"test games={len(games)} repeats={REPEATS} specs={sys.argv[1:]}", flush=True)
    for idx, spec in enumerate(sys.argv[1:] or ["10", "15", "20", "25", "30", "40"]):
        tclient = None
        t0 = time.time()
        if spec.startswith("tinker://"):
            # Pre-materialized sampler: needs no FSDP training slot.
            label = spec.rstrip("/").split("-")[-1]
            sampler = await client.create_sampling_client_async(model_path=spec)
        else:
            # Materialize a sampler from a training checkpoint. Each ephemeral client
            # occupies an FSDP slot, so it is unloaded right after its eval (below).
            # Note: unload_model also DELETES the ephemeral run's records in Redis;
            # never unload a run whose checkpoints must stay loadable (this branch's
            # run is disposable, so it is safe).
            step = int(spec)
            label = f"step{step}"
            path = open(
                os.path.join(CK, f"global_step_{step}", "remote_checkpoint_path.txt")
            ).read().strip()
            # create_training_client_from_state_async asserts is_lora, so full-param
            # checkpoints must use the trainer's own resume flow.
            tclient = await client.create_lora_training_client_async(
                base_model=os.environ.get("TRINITY_MODEL_PATH", "Qwen/Qwen3-1.7B"),
                rank=16,
                user_metadata={"training_mode": "full_param"},
            )
            _load = await tclient.load_state_with_optimizer_async(path)
            await _load.result_async()
            _sw = await tclient.save_weights_for_sampler_async(f"eval-step{step}")
            sw = await _sw
            sampler = await client.create_sampling_client_async(model_path=sw.path)
        tasks = [
            run_one(sampler, tok, g, 1000 + idx * 97 + i * 7 + rep, sem)
            for rep in range(REPEATS)
            for i, g in enumerate(games)
        ]
        res = await asyncio.gather(*tasks)
        n = len(res)
        s = sum(res)
        print(
            f"ckpt {label:>6}: success {s}/{n} = {100*s/n:.2f}%  ({time.time()-t0:.0f}s)",
            flush=True,
        )
        if tclient is not None:
            try:
                import httpx

                async with httpx.AsyncClient() as hc:
                    r = await hc.post(
                        f"{os.environ.get('TINKER_BASE_URL', 'http://127.0.0.1:10610')}/api/v1/unload_model",
                        json={"model_id": tclient.model_id},
                        headers={"X-API-Key": os.environ["TINKER_API_KEY"]},
                        timeout=900,
                    )
                print(f"[unload {label}] {tclient.model_id} -> {r.status_code}", flush=True)
            except Exception as e:
                print(f"[unload {label}] failed: {e}", flush=True)
            tclient = None


asyncio.run(main())
