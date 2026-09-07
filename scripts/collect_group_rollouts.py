"""Debug group sampling; requires rollout_rema.py in this directory.

No parameter updates. GSM8K train is a smoke-test dataset, not a claim
about the paper's training recipe. Token log probabilities are not saved.
"""

import argparse
import json
import math
import re
from decimal import Decimal
from pathlib import Path


def numeric_answer(text):
    if text is None:
        return None
    text = text.strip()
    pattern = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
    if not re.fullmatch(pattern, text):
        return None
    return Decimal(text.replace(",", ""))


def group_advantages(rewards):
    mean = sum(rewards) / len(rewards)
    std = math.sqrt(sum((r - mean) ** 2 for r in rewards) / len(rewards))
    return [(r - mean) / (std + 1e-8) for r in rewards], std


def generate(model, tokenizer, messages, budget, temperature):
    import torch

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(
        prompt, return_tensors="pt", add_special_tokens=False,
    ).to(model.device)
    prompt_ids = inputs["input_ids"][0].tolist()
    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    with torch.inference_mode():
        output = model.generate(
            **inputs, max_new_tokens=budget, do_sample=True,
            temperature=temperature, top_p=1.0, top_k=0,
            repetition_penalty=1.0, num_beams=1,
            eos_token_id=eos, pad_token_id=tokenizer.pad_token_id,
        )
    completion = output[0, len(prompt_ids):].tolist()
    ended = bool(completion) and completion[-1] in eos_ids
    return {
        "messages": messages,
        "prompt_token_ids": prompt_ids,
        "completion_token_ids": completion,
        "completion_mask": [1] * len(completion),
        "text": tokenizer.decode(completion, skip_special_tokens=True).strip(),
        "stop_reason": "eos" if ended else (
            "max_new_tokens" if len(completion) >= budget else "other"
        ),
    }


def rollout(model, tokenizer, problem, args, protocol):
    trajectory = []
    calls = []
    finished = False
    for round_number in range(1, args.max_rounds + 1):
        meta = generate(
            model, tokenizer,
            protocol.build_meta_messages(problem, trajectory),
            args.meta_max_new_tokens, args.temperature,
        )
        finished = "[FINISH]" in meta["text"]
        reasoning = generate(
            model, tokenizer,
            protocol.build_reasoning_messages(problem, meta["text"], trajectory),
            args.reasoning_max_new_tokens, args.temperature,
        )
        trajectory.append({
            "round": round_number, "meta_output": meta["text"],
            "reasoning_output": reasoning["text"],
        })
        calls.extend([
            {"round": round_number, "agent": "meta", **meta},
            {"round": round_number, "agent": "reasoning", **reasoning},
        ])
        print(f"    round={round_number} finish={finished} "
              f"meta_stop={meta['stop_reason']} reasoning_stop={reasoning['stop_reason']}",
              flush=True)
        if finished:
            break

    boxes = protocol.extract_boxed_all(trajectory[-1]["reasoning_output"])
    # GSM8K requests a single number; ambiguous multi-box outputs need review.
    prediction = boxes[0] if len(boxes) == 1 else None
    return {
        "trajectory": trajectory, "calls": calls,
        "finished_by_meta": finished, "forced_finish": False,
        "termination": "meta_finish" if finished else "max_rounds",
        "final_response": trajectory[-1]["reasoning_output"],
        "boxed_answers": boxes, "prediction": prediction,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument("--meta-max-new-tokens", type=int, default=512)
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/group_rollouts_gsm8k_train.jsonl")
    args = parser.parse_args()
    if min(args.limit, args.start_index, args.max_rounds,
           args.meta_max_new_tokens, args.reasoning_max_new_tokens) < 1:
        parser.error("Counts and indices must be positive")
    if args.group_size < 2 or not math.isfinite(args.temperature) or args.temperature <= 0:
        parser.error("group-size must be >= 2 and temperature must be finite and > 0")
    path = Path(args.output)
    if path.exists():
        parser.error("Output already exists; choose a new --output path")

    import torch
    import transformers
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    import rollout_rema as protocol

    ds = load_dataset("openai/gsm8k", "main", split="train")
    start = args.start_index - 1
    if start + args.limit > len(ds):
        parser.error("Requested range exceeds the training dataset")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype="auto", device_map="auto",
    ).eval()
    metadata = {
        "config": vars(args), "dataset": "openai/gsm8k",
        "subset": "main", "split": "train", "dataset_fingerprint": ds._fingerprint,
        "torch_version": torch.__version__, "transformers_version": transformers.__version__,
        "model_commit": getattr(model.config, "_commit_hash", None),
        "enable_thinking": False, "top_p": 1.0, "top_k": 0,
        "reward_rule": "meta finish AND one complete numeric box equal to gold",
        "advantage_rule": "population std, epsilon=1e-8",
        "purpose": "debug collection only; not ready for GRPO updates",
    }
    path.parent.mkdir(parents=True, exist_ok=True)    
    with path.open("x", encoding="utf-8") as output:
        output.write(json.dumps({"type": "metadata", **metadata}, ensure_ascii=False) + "\n")
        output.flush()
        for position in range(start, start + args.limit):
            item = ds[position]
            if "####" not in item["answer"]:
                raise ValueError("Missing GSM8K answer delimiter")
            gold = item["answer"].rsplit("####", 1)[1].strip()
            gold_number = numeric_answer(gold)
            if gold_number is None:
                raise ValueError(f"Unsupported gold answer: {gold}")
            records = []
            for sample in range(args.group_size):
                seed = args.seed + position * args.group_size + sample
                set_seed(seed)
                print(f"Question {position + 1}, rollout {sample + 1}/{args.group_size}", flush=True)
                record = rollout(model, tokenizer, item["question"], args, protocol)
                record["seed"] = seed
                record["sample_index"] = sample + 1
                record["reward"] = float(
                    record["finished_by_meta"] and numeric_answer(record["prediction"]) == gold_number
                )
                records.append(record)
                output.write(json.dumps({
                    "type": "rollout", "question_index": position + 1,
                    "problem": item["question"], "gold": gold, **record,
                }, ensure_ascii=False) + "\n")
                output.flush()
                print(f"    pred={record['prediction']} gold={gold} reward={record['reward']}", flush=True)
            rewards = [r["reward"] for r in records]
            advantages, std = group_advantages(rewards)
            output.write(json.dumps({
                "type": "group_summary", "question_index": position + 1,
                "sample_indices": list(range(1, args.group_size + 1)),
                "rewards": rewards, "advantages": advantages,
                "zero_reward_variance": std == 0,
            }) + "\n")
            output.flush()
            print(f"Rewards: {rewards}\nAdvantages: {advantages}", flush=True)
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
