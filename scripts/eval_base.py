import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import StoppingCriteria, StoppingCriteriaList

MODEL_ID = "Qwen/Qwen3-1.7B"

QUESTIONS = [
    {
        "problem": "What is 12 * 13?",
        "answer": "156",
    },
    {
        "problem": "If x + 5 = 17, what is x?",
        "answer": "12",
    },
    {
        "problem": "Compute 25% of 80.",
        "answer": "20",
    },
]

PROMPT_TEMPLATE = """You are solving a math problem.

Problem:
{problem}

Solve briefly.
Your last line must contain the final answer using LaTeX boxed notation.
Do not write any boxed expression until the last line."""

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


def normalize_answer(x):
    if x is None:
        return None
    return str(x).strip().replace(" ", "")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    correct = 0

    for idx, item in enumerate(QUESTIONS, start=1):
        prompt = PROMPT_TEMPLATE.format(problem=item["problem"])
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=tok.eos_token_id,
		pad_token_id=tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([
                StopAfterBoxed(tok, prompt_len)]),
)
        generated_ids = out[0][inputs["input_ids"].shape[1]:]
        text = tok.decode(generated_ids, skip_special_tokens=True)
        pred = extract_boxed(text)
        ok = normalize_answer(pred) == normalize_answer(item["answer"])

        correct += int(ok)

        print("=" * 80)
        print(f"Question {idx}")
        print("Problem:", item["problem"])
        print("Gold:", item["answer"])
        print("Pred:", pred)
        print("Correct:", ok)
        print("-" * 80)
        print(text)

    print("=" * 80)
    print(f"Accuracy: {correct}/{len(QUESTIONS)} = {correct / len(QUESTIONS):.2%}")


if __name__ == "__main__":
    main()
