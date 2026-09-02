import argparse
import re

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)


class StopAfterBoxed(StoppingCriteria):
    def __init__(self, tokenizer, prompt_len):
        self.tokenizer = tokenizer
        self.prompt_len = prompt_len
        self.pattern = re.compile(r"\\boxed\{[^{}]+\}")

    def __call__(self, input_ids, scores, **kwargs):
        generated_ids = input_ids[0][self.prompt_len:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return self.pattern.search(text) is not None


def extract_boxed(text):
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if not matches:
        return None
    return matches[0].strip()


def extract_number(text):
    if text is None:
        return None
    text = str(text).replace(",", "")
    matches = re.findall(r"-?\d+(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1]


def normalize_answer(x):
    if x is None:
        return None
    x = str(x).strip()
    x = x.replace(",", "")
    x = x.replace("$", "")
    x = x.replace(" ", "")
    return x


def extract_gsm8k_gold(answer_text):
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip().replace(",", "")
    return extract_number(answer_text)


def build_prompt(tokenizer, problem):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful math solver. "
                "Solve the problem briefly. "
                "Your final answer must be written exactly once using LaTeX boxed notation."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Problem:\n{problem}\n\n"
                "Solve step by step, then end with one line containing only the final answer in \\boxed{}."
            ),
        },
    ]

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    return (
        "System: You are a careful math solver. "
        "Your final answer must be written exactly once using LaTeX boxed notation.\n"
        f"User: Problem:\n{problem}\n\n"
        "Solve step by step, then end with one line containing only the final answer in \\boxed{}.\n"
        "Assistant:"
    )


def is_correct(pred, gold):
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    pred_num = extract_number(pred_norm)
    gold_num = extract_number(gold_norm)

    return pred_num is not None and pred_num == gold_num


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Loading dataset: openai/gsm8k main {args.split}")
    ds = load_dataset("openai/gsm8k", "main", split=args.split)
    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    correct = 0
    total = 0

    for idx, item in enumerate(ds, start=1):
        problem = item["question"]
        gold = extract_gsm8k_gold(item["answer"])
        prompt = build_prompt(tokenizer, problem)

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        do_sample = args.temperature > 0.0

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
                stopping_criteria=StoppingCriteriaList(
                    [StopAfterBoxed(tokenizer, prompt_len)]
                ),
            )

        generated_ids = output_ids[0][prompt_len:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

        pred_boxed = extract_boxed(generated_text)
        pred = pred_boxed if pred_boxed is not None else extract_number(generated_text)
        ok = is_correct(pred, gold)

        correct += int(ok)
        total += 1

        print("=" * 80)
        print(f"Question {idx}/{len(ds)}")
        print("Problem:", problem)
        print("Gold:", gold)
        print("Pred:", pred)
        print("Correct:", ok)
        print("-" * 80)
        print(generated_text.strip())

    print("=" * 80)
    print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")


if __name__ == "__main__":
    main()
