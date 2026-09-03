import argparse
import re
import sympy as sp
import torch
import html

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

def latex_to_sympy_expr(x):
    if x is None:
        return None

    x = normalize_answer(x)

    replacements = {
        r"\pi": "pi",
        r"\sqrt": "sqrt",
    }

    for old, new in replacements.items():
        x = x.replace(old, new)

    x = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", x)
    x = x.replace("^", "**")

    try:
        return sp.sympify(x)
    except Exception:
        return None


def numeric_equal(pred, gold, tolerance=1e-3):
    pred_expr = latex_to_sympy_expr(pred)
    gold_expr = latex_to_sympy_expr(gold)

    if pred_expr is None or gold_expr is None:
        return False

    try:
        diff = abs(float(sp.N(pred_expr)) - float(sp.N(gold_expr)))
        return diff <= tolerance
    except Exception:
        return False

def normalize_answer(x):
    if x is None:
        return None

    x = html.unescape(str(x).strip())

    x = re.sub(r"\\text\{([^{}]+)\}", r"\1", x)
    x = re.sub(r"\\mathrm\{([^{}]+)\}", r"\1", x)
    x = re.sub(r"\\operatorname\{([^{}]+)\}", r"\1", x)

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

def build_prompt(tokenizer, problem):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a careful math solver. "
                "Solve the problem accurately and concisely. "
                "Before giving the final answer, identify what the problem is asking for. "
                "Your final boxed answer must directly answer what the problem asks for. "
                "If the problem asks for a person, place, option, label, or text, put that text in the box. "
                "If the problem asks for a number, expression, coordinate, interval, set, or equation, put that object in the box. "
                "Do not put an intermediate value in the final box."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Solve the following math problem efficiently and clearly.\n\n"
                f"{problem}\n\n"
                "Think step by step before answering. "
                "Your last line must be exactly:\n"
                "Therefore, the final answer is: $\\boxed{ANSWER}$.\n"
                "Replace ANSWER with only the requested final answer."
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
        "Your final boxed answer must directly answer what the problem asks for. "
        "Do not put an intermediate value in the final box.\n"
        f"User: Solve the following math problem efficiently and clearly.\n\n{problem}\n\n"
        "Think step by step before answering. "
        "Your last line must be exactly:\n"
        "Therefore, the final answer is: $\\boxed{ANSWER}$.\n"
        "Replace ANSWER with only the requested final answer.\n"
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

def strip_degree_for_compare(x):
    if x is None:
        return None

    x = normalize_answer(x)
    x = x.replace(r"^\circ", "")
    x = x.replace(r"^{\circ}", "")
    x = x.replace(r"\degree", "")
    x = x.replace(r"^\degree", "")

    return x


def is_correct(pred, gold):
    pred_norm = normalize_answer(pred)
    gold_norm = normalize_answer(gold)

    if pred_norm == gold_norm:
        return True

    pred_degree_stripped = strip_degree_for_compare(pred)
    gold_degree_stripped = strip_degree_for_compare(gold)

    if pred_degree_stripped == gold_degree_stripped:
        return True

    return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--dataset", default="HuggingFaceH4/MATH-500")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
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
