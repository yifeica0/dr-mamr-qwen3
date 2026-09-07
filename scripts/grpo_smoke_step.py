"""One diagnostic GRPO-style LoRA update, not the Dr. MAMR training recipe.

Requires collect_group_rollouts.py and rollout_rema.py beside this file.
Uses fresh GSM8K train rollouts, temperature=1, no KL, and uniform means
over trajectories, agent calls, then completion tokens. No forced FINISH.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path


def completion_bounds(prompt_length, completion_length):
    if prompt_length < 1 or completion_length < 1:
        raise ValueError("Prompt and completion must both contain tokens")
    return prompt_length - 1, prompt_length + completion_length - 1


def token_logps(model, call):
    import torch
    import torch.nn.functional as F

    prompt = call["prompt_token_ids"]
    completion = call["completion_token_ids"]
    start, end = completion_bounds(len(prompt), len(completion))
    ids = torch.tensor([prompt + completion], device=model.device)
    logits = model(
        input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
    ).logits[0, start:end]
    targets = ids[0, len(prompt):]
    # FP32 normalization in chunks limits temporary vocabulary-sized tensors.
    return torch.cat([
        -F.cross_entropy(logits[i:i + 128].float(), targets[i:i + 128], reduction="none")
        for i in range(0, len(completion), 128)
    ])


def clipped_loss(new_logps, old_logps, advantage, mask, clip):
    import torch

    ratio = (new_logps - old_logps).exp()
    surrogate = torch.minimum(
        ratio * advantage, ratio.clamp(1 - clip, 1 + clip) * advantage,
    )
    return -(surrogate * mask).sum() / mask.sum()


def self_test():
    import torch

    # A tiny causal model makes token alignment and prompt exclusion observable.
    class Toy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.table = torch.nn.Parameter(torch.arange(16.).reshape(4, 4) / 9)
            self.device = torch.device("cpu")

        def forward(self, input_ids, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(logits=self.table[input_ids])

    toy = Toy()
    call = {"prompt_token_ids": [0, 1], "completion_token_ids": [2, 3]}
    values = token_logps(toy, call)
    expected = toy.table.log_softmax(-1)[[1, 2], [2, 3]]
    torch.testing.assert_close(values, expected)
    (-values.mean()).backward()
    assert toy.table.grad[0].abs().sum() == 0  # Prompt-only prediction.
    assert toy.table.grad[1].abs().sum() > 0  # First completion token.
    assert toy.table.grad[2].abs().sum() > 0
    assert toy.table.grad[3].abs().sum() == 0  # No target after completion.
    assert completion_bounds(3, 2) == (2, 4)

    new = torch.zeros(2, requires_grad=True)
    old = torch.zeros(2)
    clipped_loss(new, old, 1.0, torch.tensor([1., 0.]), 0.2).backward()
    torch.testing.assert_close(new.grad, torch.tensor([-1., 0.]))
    for advantage, ratio in [(1., 1.5), (-1., 0.5)]:
        new = torch.tensor([math.log(ratio)], requires_grad=True)
        loss = clipped_loss(new, torch.zeros(1), advantage, torch.ones(1), 0.2)
        loss.backward()
        assert new.grad.item() == 0  # Clipped side must not keep pushing.

    from collect_group_rollouts import group_advantages
    assert group_advantages([1., 1., 1., 1.])[0] == [0.] * 4
    a, _ = group_advantages([1., 1., 1., 0.])
    assert abs(sum(a)) < 1e-6 and a[-1] < 0 < a[0]
    print("PASS: causal alignment, prompt/completion masks, clipping, group advantage")


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--index", type=int, default=2)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--max-groups", type=int, default=3)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--meta-max-new-tokens", type=int, default=512)
    parser.add_argument("--reasoning-max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/grpo_smoke_step_01")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if min(args.index, args.max_groups, args.max_rounds, args.meta_max_new_tokens,
           args.reasoning_max_new_tokens, args.max_seq_length) < 1 or args.group_size < 2:
        parser.error("Counts must be positive; group-size must be >= 2")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be positive and finite")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        parser.error("Output directory exists; choose a new --output-dir")

    import torch
    import peft
    import transformers
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
    import collect_group_rollouts as collector
    import rollout_rema as protocol

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("This smoke run requires a CUDA GPU supporting BF16")
    set_seed(args.seed)
    ds = load_dataset("openai/gsm8k", "main", split="train")
    if args.index > len(ds):
        parser.error("index exceeds GSM8K train length")
    item = ds[args.index - 1]
    if "####" not in item["answer"]:
        raise ValueError("Missing GSM8K gold delimiter")
    gold = item["answer"].rsplit("####", 1)[1].strip()
    gold_number = collector.numeric_answer(gold)
    if gold_number is None:
        raise ValueError(f"Unsupported gold: {gold}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16,
    ).to("cuda:0")
    # Remove model-specific sampling processors so sampled and scored policies agree.
    eos = base.generation_config.eos_token_id
    if eos is None:
        eos = tokenizer.eos_token_id
    base.generation_config = GenerationConfig(
        bos_token_id=tokenizer.bos_token_id, eos_token_id=eos,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = get_peft_model(base, LoraConfig(
        task_type="CAUSAL_LM", r=8, lora_alpha=16,
        lora_dropout=0.0, bias="none", target_modules=["q_proj", "v_proj"],
    ))
    # eval disables dropout but does not disable autograd during the update.
    model.eval()
    model.print_trainable_parameters()
    args.temperature = 1.0
    output_dir.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(output_dir / "adapter_before")
    write_json(output_dir / "config.json", {
        "args": vars(args), "dataset": "openai/gsm8k", "split": "train",
        "dataset_fingerprint": ds._fingerprint,
        "model_commit": getattr(base.config, "_commit_hash", None),
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                     "peft": peft.__version__},
        "source_sha256": {
            Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in [collector, protocol]
        },
        "kl_beta": 0.0, "clip_epsilon": 0.2,
        "loss_reduction": "mean trajectories / mean agent calls / mean completion tokens",
        "purpose": "One-step diagnostic, not paper reproduction or accuracy evidence",
        "sampling": "temperature=1, top_p=1, top_k=0, repetition_penalty=1",
        "group_selection": "bounded retries on same question, skip uniform rewards or truncation",
    })

    selected = None
    for group_index in range(args.max_groups):
        records = []
        for sample in range(args.group_size):
            seed = args.seed + group_index * args.group_size + sample
            set_seed(seed)
            print(f"Group {group_index + 1}, rollout {sample + 1}/{args.group_size}", flush=True)
            record = collector.rollout(model, tokenizer, item["question"], args, protocol)
            record["seed"] = seed
            record["reward"] = float(record["finished_by_meta"] and
                                      collector.numeric_answer(record["prediction"]) == gold_number)
            records.append(record)
        rewards = [r["reward"] for r in records]
        advantages, std = collector.group_advantages(rewards)
        invalid_calls = any(
            c["stop_reason"] != "eos" or not c["completion_token_ids"] or
            len(c["prompt_token_ids"]) + len(c["completion_token_ids"]) > args.max_seq_length
            for r in records for c in r["calls"]
        )
        write_json(output_dir / f"group_{group_index + 1}.json", {
            "question": item["question"], "gold": gold, "records": records,
            "rewards": rewards, "advantages": advantages,
            "invalid_calls": invalid_calls,
        })
        print(f"Rewards: {rewards}\nAdvantages: {advantages}", flush=True)
        if std > 0 and not invalid_calls:
            selected = records, advantages
            break
        print("Skipping group: uniform rewards or truncated/oversized calls", flush=True)
    if selected is None:
        write_json(output_dir / "metrics.json", {
            "status": "no_update", "reason": "No eligible mixed-reward group within budget",
        })
        print("NO UPDATE: collection budget reached. See group logs; parameters were not updated.")
        return

    records, advantages = selected
    print("Computing old policy log probabilities before any update...", flush=True)
    with torch.no_grad():
        for record in records:
            for call in record["calls"]:
                values = token_logps(model, call)
                if not torch.isfinite(values).all():
                    raise RuntimeError("Non-finite old policy log probabilities")
                call["old_logps"] = values.cpu().tolist()
    write_json(output_dir / "training_group.json", {
        "problem": item["question"], "gold": gold,
        "records": records, "advantages": advantages,
    })

    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    before = {name: p.detach().float().cpu().clone() for name, p in parameters.items()}
    optimizer = torch.optim.AdamW(parameters.values(), lr=args.learning_rate, weight_decay=0.0)
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    max_log_ratio = 0.0
    for record, advantage in zip(records, advantages):
        for call in record["calls"]:
            new = token_logps(model, call)
            old = torch.tensor(call["old_logps"], device=new.device)
            mask = torch.tensor(call["completion_mask"], device=new.device, dtype=torch.float32)
            if mask.shape != new.shape or mask.sum().item() <= 0:
                raise RuntimeError("Invalid completion mask")
            delta = (new.detach() - old).abs().max().item()
            max_log_ratio = max(max_log_ratio, delta)
            if not math.isfinite(delta) or delta > 0.05:
                raise RuntimeError("Old/new log probabilities disagree before optimizer step")
            loss = clipped_loss(new, old, advantage, mask, 0.2)
            loss = loss / (len(records) * len(record["calls"]))
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite loss")
            total_loss += loss.detach().item()
            loss.backward()
            del new, old, mask, loss
    grad_norm = torch.nn.utils.clip_grad_norm_(
        list(parameters.values()), 1.0, error_if_nonfinite=True,
    ).item()
    if grad_norm <= 0:
        raise RuntimeError("No nonzero gradient; optimizer step aborted")
    optimizer.step()
    changes = {
        name: (p.detach().float().cpu() - before[name]).abs().max().item()
        for name, p in parameters.items()
    }
    max_change = max(changes.values())
    if not all(math.isfinite(x) for x in changes.values()) or max_change <= 0:
        raise RuntimeError("Parameters did not change finitely after optimizer step")
    model.save_pretrained(output_dir / "adapter_after")
    tokenizer.save_pretrained(output_dir / "adapter_after")
    metrics = {
        "status": "updated", "optimizer_steps": 1, "loss_before_update": total_loss,
        "gradient_norm_before_clip": grad_norm, "max_parameter_change": max_change,
        "changed_parameter_tensors": sum(x > 0 for x in changes.values()),
        "max_abs_log_ratio_before_step": max_log_ratio,
        "sampled_round_counts": [len(r["trajectory"]) for r in records],
        "multi_round_observed": any(len(r["trajectory"]) > 1 for r in records),
        "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
    }
    write_json(output_dir / "metrics.json", metrics)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Saved LoRA adapter to {output_dir / 'adapter_after'}")


if __name__ == "__main__":
    main()
