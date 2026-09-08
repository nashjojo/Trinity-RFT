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
CK = os.path.join(
    BASE, "checkpoints/ALFWORLD/Step_Wise_Alfworld_TuFT_lora32_lr5e5"
)
TEST = os.path.join(BASE, "examples/grpo_alfworld/alfworld_data/test.jsonl")
MAX_STEPS = 30
REPEATS = int(os.environ.get("EVAL_REPEATS", "2"))
CONCURRENCY = int(os.environ.get("EVAL_CONCURRENCY", "32"))


def make_env(game):
    expert = AlfredExpert(expert_type=AlfredExpertType.HANDCODED)
    ri = textworld.EnvInfos(description=True, inventory=True, admissible_commands=True)
    eid = textworld.gym.register_game(game, ri, wrappers=[AlfredDemangler(), expert])
    return textworld.gym.make(eid)


def _render_prompt(tok, mem):
    prompt = tok.apply_chat_template(
        mem, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return tok.encode(prompt)


# textworld's tatsu PDDL parser holds shared module-level state, so env.reset /
# env.step must not run concurrently across episodes.
ENV_LOCK = asyncio.Lock()


async def run_one(sampler, tok, game, seed, sem):
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
        t0 = time.time()
        if spec.startswith("tinker://"):
            # Pre-materialized sampler: needs no FSDP training slot.
            label = spec.rstrip("/").split("-")[-1]
            sampler = await client.create_sampling_client_async(model_path=spec)
        else:
            # SLOT LEAK WARNING: each create_training_client_from_state_async occupies
            # an FSDP slot and the tinker SDK (0.24.1) exposes no close/release, so
            # looping over checkpoints leaks one slot per checkpoint (only 4 exist for
            # rank 32). Prefer the tinker:// sampler-path mode above, which uses no
            # slot. If this branch must be used, release the ephemeral client after
            # its sampler has been consumed via
            #   POST /api/v1/unload_model {"model_id": "<uuid>"}
            # and ONLY then: unload_model also DELETES the run and checkpoint records
            # in Redis (disk files survive but become unreferencable), so never call
            # it on a run whose checkpoints you still want to load.
            step = int(spec)
            label = f"step{step}"
            path = open(
                os.path.join(CK, f"global_step_{step}", "remote_checkpoint_path.txt")
            ).read().strip()
            # Sampler checkpoints are pruned by the trainer (only the latest survives),
            # so materialize a fresh sampler from the surviving training checkpoint.
            tclient = await client.create_training_client_from_state_async(path)
            sw = await tclient.save_weights_for_sampler_async(f"eval-step{step}")
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


asyncio.run(main())
