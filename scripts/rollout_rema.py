import argparse
import json
import os
import re

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


META_SYSTEM_PROMPT = r"""
You are a meta-think agent that represents human high-level think process. When solving a question,
you will have a discussion with the human, and each time you will think about what to do next. For
example:
• Exploring multiple angles and approaches
• Breaking down the solution into clear steps
• Continuously reflecting on intermediate results honestly and adapting your strategy as you
progress
• Backtracking when necessary
• Requesting exploration of multiple solutions individually
• Finally, confirm the answer with the tag [FINISH].
Please do not focus on completing the task by calculating the final answer; that step will be handled
by a separate reasoning agent.
""".strip()


REASONING_SYSTEM_PROMPT = r"""
You are a reasoning agent that follows structured problem-solving instructions step by step. Your goals
are:
• Follow the given instruction precisely.
• Reason step by step toward a solution.
• Avoid producing empty or blank outputs at any step.
• If uncertain, provide your best reasoning and partial answer rather than outputting nothing.
• Always provide a meaningful and non-empty response, even during intermediate steps.
• When you receive the signal [FINISH], finalize your answer and place it within
\boxed{}.
• If unable to finalize, explain why and still output your best available answer within
\boxed{}.
Remember: You must never produce trivial outputs.
""".strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inference-only ReMA rollout on MATH-500."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-1.7B",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/rema_math500.jsonl",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="Run one MATH-500 question using a 1-based index.",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum number of Meta/Reasoning discussion rounds.",
    )
    parser.add_argument(
        "--meta-max-new-tokens",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--reasoning-max-new-tokens",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )

    return parser.parse_args()


def apply_chat_template(tokenizer, messages):
    if (
        hasattr(tokenizer, "apply_chat_template")
        and tokenizer.chat_template is not None
    ):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    parts = []

    for message in messages:
        role = message["role"].capitalize()
        parts.append(f"{role}: {message['content']}")

    parts.append("Assistant:")

    return "\n".join(parts)


def generate_text(
    model,
    tokenizer,
    messages,
    max_new_tokens,
    temperature,
):
    prompt = apply_chat_template(tokenizer, messages)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
    ).to(model.device)

    prompt_length = inputs["input_ids"].shape[1]
    do_sample = temperature > 0.0

    generation_arguments = {
        **inputs,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }

    if do_sample:
        generation_arguments["temperature"] = temperature

    with torch.no_grad():
        output_ids = model.generate(**generation_arguments)

    generated_ids = output_ids[0, prompt_length:]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


def extract_boxed_all(text):
    results = []
    marker = r"\boxed{"
    search_start = 0

    while True:
        position = text.find(marker, search_start)

        if position == -1:
            break

        index = position + len(marker)
        depth = 1
        characters = []

        while index < len(text):
            character = text[index]

            if character == "{":
                depth += 1
                characters.append(character)

            elif character == "}":
                depth -= 1

                if depth == 0:
                    results.append(
                        "".join(characters).strip()
                    )
                    break

                characters.append(character)

            else:
                characters.append(character)

            index += 1

        search_start = index + 1

    return results


def extract_boxed(text):
    answers = extract_boxed_all(text)

    if not answers:
        return None

    return answers[-1]

def extract_prediction(text):
    answers = extract_boxed_all(text)

    if not answers:
        return None

    if len(answers) == 1:
        return answers[0]

    return ", ".join(answers)

def normalize_answer(answer):
    if answer is None:
        return None

    answer = str(answer).strip()

    answer = re.sub(
        r"\\text\{([^{}]+)\}",
        r"\1",
        answer,
    )
    answer = re.sub(
        r"\\mathrm\{([^{}]+)\}",
        r"\1",
        answer,
    )
    answer = re.sub(
        r"\\operatorname\{([^{}]+)\}",
        r"\1",
        answer,
    )
    answer = re.sub(
        r"\\sqrt([0-9a-zA-Z]+)",
        r"\\sqrt{\1}",
        answer,
    )

    replacements = {
        r"\left": "",
        r"\right": "",
        r"\,": "",
        r"\!": "",
        r"\ ": "",
        "$": "",
    }

    for old, new in replacements.items():
        answer = answer.replace(old, new)

    answer = answer.replace(" ", "")
    answer = answer.replace("\n", "")
    answer = answer.replace("\t", "")

    return answer


def strip_single_variable_assignment(answer):
    answer = normalize_answer(answer)

    if answer is None:
        return None

    match = re.fullmatch(r"[a-zA-Z]=(.+)", answer)

    if match:
        return match.group(1)

    return answer


def strip_degree_for_comparison(answer):
    answer = normalize_answer(answer)

    if answer is None:
        return None

    answer = answer.replace(r"^{\circ}", "")
    answer = answer.replace(r"^\circ", "")
    answer = answer.replace(r"^\degree", "")
    answer = answer.replace(r"\degree", "")

    return answer


def answers_equal(prediction, gold):
    prediction_normalized = normalize_answer(prediction)
    gold_normalized = normalize_answer(gold)

    if prediction_normalized is None:
        return False

    if prediction_normalized == gold_normalized:
        return True

    if (
        strip_single_variable_assignment(prediction)
        == strip_single_variable_assignment(gold)
    ):
        return True

    if (
        strip_degree_for_comparison(prediction)
        == strip_degree_for_comparison(gold)
    ):
        return True

    return False


def get_problem(item):
    for field in ["problem", "question", "prompt"]:
        if field in item:
            return str(item[field]).strip()

    raise KeyError(
        f"Cannot find problem field. Available keys: "
        f"{list(item.keys())}"
    )


def get_gold(item):
    for field in ["answer", "final_answer"]:
        if field in item:
            return str(item[field]).strip()

    if "solution" in item:
        boxed_answer = extract_boxed(
            str(item["solution"])
        )

        if boxed_answer is not None:
            return boxed_answer

    raise KeyError(
        f"Cannot find answer field. Available keys: "
        f"{list(item.keys())}"
    )


def format_discussion(trajectory):
    if not trajectory:
        return "No discussion has occurred yet."

    parts = []

    for turn in trajectory:
        parts.append(
            f"Meta-think agent, round {turn['round']}:\n"
            f"{turn['meta_output']}"
        )
        parts.append(
            f"Reasoning agent, round {turn['round']}:\n"
            f"{turn['reasoning_output']}"
        )

    return "\n\n".join(parts)


def build_meta_messages(problem, trajectory):
    discussion = format_discussion(trajectory)

    user_message = f"""
Question:
{problem}

Discussion so far:
{discussion}

What should the reasoning agent do next?
""".strip()

    return [
        {
            "role": "system",
            "content": META_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def build_reasoning_messages(
    problem,
    meta_instruction,
    trajectory,
):
    if trajectory:
        previous_reasoning = trajectory[-1][
            "reasoning_output"
        ]
    else:
        previous_reasoning = (
            "No previous reasoning response."
        )

    user_message = f"""
Question:
{problem}

Instruction from the meta-think agent:
{meta_instruction}

Previous response from the reasoning agent:
{previous_reasoning}
""".strip()

    return [
        {
            "role": "system",
            "content": REASONING_SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": user_message,
        },
    ]


def run_rollout(
    model,
    tokenizer,
    problem,
    max_rounds,
    meta_max_new_tokens,
    reasoning_max_new_tokens,
    temperature,
):
    trajectory = []
    finished_by_meta = False
    forced_finish = False

    for round_number in range(1, max_rounds + 1):
        meta_messages = build_meta_messages(
            problem=problem,
            trajectory=trajectory,
        )

        meta_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=meta_messages,
            max_new_tokens=meta_max_new_tokens,
            temperature=temperature,
        )

        finished_by_meta = "[FINISH]" in meta_output

        instruction_for_reasoning = meta_output

        if (
            round_number == max_rounds
            and not finished_by_meta
        ):
            instruction_for_reasoning = (
                meta_output + "\n\n[FINISH]"
            )
            forced_finish = True

        reasoning_messages = build_reasoning_messages(
            problem=problem,
            meta_instruction=instruction_for_reasoning,
            trajectory=trajectory,
        )

        reasoning_output = generate_text(
            model=model,
            tokenizer=tokenizer,
            messages=reasoning_messages,
            max_new_tokens=reasoning_max_new_tokens,
            temperature=temperature,
        )

        trajectory.append(
            {
                "round": round_number,
                "meta_output": meta_output,
                "instruction_for_reasoning": (
                    instruction_for_reasoning
                ),
                "reasoning_output": reasoning_output,
                "meta_requested_finish": finished_by_meta,
                "forced_finish": forced_finish,
            }
        )

        if finished_by_meta or forced_finish:
            break

    final_response = trajectory[-1]["reasoning_output"]
    prediction = extract_prediction(final_response)
    return {
        "trajectory": trajectory,
        "finished_by_meta": finished_by_meta,
        "forced_finish": forced_finish,
        "final_response": final_response,
        "predicted_answer": prediction,
    }


def select_items(dataset, index, limit):
    if index is not None:
        if not 1 <= index <= len(dataset):
            raise ValueError(
                f"--index must be between 1 and "
                f"{len(dataset)}"
            )

        return [(index, dataset[index - 1])]

    actual_limit = min(limit, len(dataset))

    return [
        (position + 1, dataset[position])
        for position in range(actual_limit)
    ]


def main():
    args = parse_args()

    if args.max_rounds < 1:
        raise ValueError("--max-rounds must be at least 1")

    print(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )

    model.eval()

    dataset = load_dataset(
        "HuggingFaceH4/MATH-500",
        split="test",
    )

    evaluation_items = select_items(
        dataset=dataset,
        index=args.index,
        limit=args.limit,
    )

    output_directory = os.path.dirname(args.output)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    correct_count = 0
    total_count = 0

    with open(
        args.output,
        "w",
        encoding="utf-8",
    ) as output_file:
        for question_number, item in evaluation_items:
            problem = get_problem(item)
            gold = get_gold(item)

            print("=" * 80)
            print(
                f"Question {question_number}/"
                f"{len(dataset)}"
            )
            print(f"Problem: {problem}")

            rollout = run_rollout(
                model=model,
                tokenizer=tokenizer,
                problem=problem,
                max_rounds=args.max_rounds,
                meta_max_new_tokens=(
                    args.meta_max_new_tokens
                ),
                reasoning_max_new_tokens=(
                    args.reasoning_max_new_tokens
                ),
                temperature=args.temperature,
            )

            prediction = rollout["predicted_answer"]
            correct = answers_equal(prediction, gold)

            total_count += 1
            correct_count += int(correct)

            result = {
                "question_number": question_number,
                "problem": problem,
                "gold_raw": gold,
                "gold_normalized": normalize_answer(gold),
                "prediction_raw": prediction,
                "prediction_normalized": (
                    normalize_answer(prediction)
                ),
                "correct": correct,
                **rollout,
            }

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False,
                )
                + "\n"
            )
            output_file.flush()

            for turn in rollout["trajectory"]:
                print("-" * 80)
                print(
                    f"Round {turn['round']} - "
                    f"Meta-think Agent"
                )
                print(turn["meta_output"])
                print()

                if turn["forced_finish"]:
                    print(
                        "[Framework forced FINISH because "
                        "max rounds were reached]"
                    )
                    print()

                print(
                    f"Round {turn['round']} - "
                    f"Reasoning Agent"
                )
                print(turn["reasoning_output"])
                print()

            print("-" * 80)
            print(f"Gold raw: {gold}")
            print(
                f"Gold normalized: "
                f"{normalize_answer(gold)}"
            )
            print(f"Pred raw: {prediction}")
            print(
                f"Pred normalized: "
                f"{normalize_answer(prediction)}"
            )
            print(f"Correct: {correct}")
            print(
                f"Finished by Meta: "
                f"{rollout['finished_by_meta']}"
            )
            print(
                f"Forced finish: "
                f"{rollout['forced_finish']}"
            )

    accuracy = (
        correct_count / total_count
        if total_count
        else 0.0
    )

    print("=" * 80)
    print(
        f"Accuracy: {correct_count}/{total_count} "
        f"= {accuracy:.2%}"
    )
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
