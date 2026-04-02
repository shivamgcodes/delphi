from .prompts import example, system, system_single_token


def build_examples(
    *,
    n_few_shot_pairs: int = 3,
    activations: bool = False,
    cot: bool = False,
):
    examples = []
    n = max(0, min(int(n_few_shot_pairs), 3))
    for i in range(1, n + 1):
        prompt, response = example(i, activations=activations, cot=cot)

        messages = [
            {
                "role": "user",
                "content": prompt,
            },
            {
                "role": "assistant",
                "content": response,
            },
        ]

        examples.extend(messages)

    return examples


def build_prompt(
    examples: str,
    activations: bool = False,
    cot: bool = False,
    n_few_shot_pairs: int = 3,
) -> list[dict]:
    messages = system(
        cot=cot,
    )

    few_shot_examples = build_examples(
        n_few_shot_pairs=n_few_shot_pairs,
        activations=activations,
        cot=cot,
    )

    messages.extend(few_shot_examples)

    user_start = f"\n{examples}\n"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages


def build_single_token_prompt(
    examples,
):
    messages = system_single_token()

    user_start = f"WORDS: {examples}"

    messages.append(
        {
            "role": "user",
            "content": user_start,
        }
    )

    return messages
