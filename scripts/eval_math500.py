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
    marker = r"\boxed{"
    start = text.find(marker)
    if start == -1:
        return None

    i = start + len(marker)
    depth = 1
    chars = []

    while i < len(text):
        ch = text[i]

        if ch == "{":
            depth += 1
            chars.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(ch)
        else:
            chars.append(ch)

        i += 1

    return None

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
    replacements = {
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\!": "",
        r"\ ": "",
        "$": "",
    }
    for old, new in replacements.items():
        x = x.replace(old, new)
    x = x.replace(" ", "")
    x = x.replace("\n", "")
    x = x.replace("\t", "")
    return x

print(normalize_answer(r"\left( 3, \frac{\pi}{2} \right)"))
print(normalize_answer(r"(3, \frac{\pi}{2})"))

def build_prompt(tokenizer, problem):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful math solver. "
                "Solve the problem briefly and correctly. "
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


def get_problem(item):
    for key in ["problem", "question"]:
        if key in item:
            return item[key]
    raise KeyError(f"Cannot find problem field. Available keys: {list(item.keys())}")

def get_gold(item):
    if "answer" in item:
        return str(item["answer"]).strip()

    if "final_answer" in item:
        return str(item["final_answer"]).strip()

    if "solution" in item:
        boxed = extract_boxed(str(item["solution"]))
        if boxed is not None:
            return boxed
        return str(item["solution"]).strip()

    raise KeyError(f"Cannot find answer field. Available keys: {list(item.keys())}")

def is_correct(pred, gold):
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    print(f"Loading dataset: {args.dataset} {args.split}")
    ds = load_dataset(args.dataset, split=args.split)

    if args.limit > 0:
        ds = ds.select(range(min(args.limit, len(ds))))

    print(f"Dataset size: {len(ds)}")
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
        problem = get_problem(item)
        gold = get_gold(item)
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
        print("Gold raw:", gold)
        print("Gold normalized:", normalize_answer(gold))
        print("Pred raw:", pred)
        print("Pred normalized:", normalize_answer(pred))
        print("Correct:", ok)
        print("-" * 80)
        print(generated_text.strip())

    print("=" * 80)
    print(f"Accuracy: {correct}/{total} = {correct / total:.2%}")


if __name__ == "__main__":
    main()
