# sft_formatting.py

def format_mathinstruct(examples):
    texts = []
    for instruction, output in zip(examples["instruction"], examples["output"]):
        text = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n{output}"
        )
        texts.append(text)
    return texts


def format_python_alpaca(examples):
    texts = []
    for instruction, input_text, output in zip(
        examples["instruction"], examples["input"], examples["output"]
    ):
        if input_text.strip():
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{input_text}\n\n"
                f"### Response:\n{output}"
            )
        else:
            text = (
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n{output}"
            )
        texts.append(text)
    return texts


def format_aya_dataset(examples):
    texts = []
    for input_text, target in zip(examples["inputs"], examples["targets"]):
        text = (
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{target}"
        )
        texts.append(text)
    return texts


def format_aya_dataset_with_language(examples):
    texts = []
    for input_text, target, lang in zip(
        examples["inputs"], examples["targets"], examples["language"]
    ):
        text = (
            f"### Language: {lang}\n\n"
            f"### Input:\n{input_text}\n\n"
            f"### Response:\n{target}"
        )
        texts.append(text)
    return texts


def format_alpaca(examples):
    return examples["text"]


def format_copa(examples):
    texts = []
    for premise, choice1, choice2, question, label in zip(
        examples["premise"],
        examples["choice1"],
        examples["choice2"],
        examples["question"],
        examples["label"],
    ):

        if question.lower() == "cause":
            q_text = "What was the cause?"
        else:
            q_text = "What was the effect?"

        if label == 1:
            answer = choice1
        else:
            answer = choice2

        text = (
            f"### Premise:\n{premise}\n\n"
            f"### Question:\n{q_text}\n\n"
            f"### Choices:\n1. {choice1}\n2. {choice2}\n\n"
            f"### Answer:\n{answer}"
        )
        texts.append(text)
    return texts


def format_socialiqa(examples):
    texts = []
    for context, question, a, b, c, label in zip(
        examples["context"],
        examples["question"],
        examples["answerA"],
        examples["answerB"],
        examples["answerC"],
        examples["label"],
    ):
        if label == "1":
            answer_text = a
        elif label == "2":
            answer_text = b
        elif label == "3":
            answer_text = c
        else:
            raise ValueError(f"Unexpected label: {label}")

        text = (
            f"### Context:\n{context}\n\n"
            f"### Question:\n{question}\n\n"
            f"### Choices:\n"
            f"A. {a}\n"
            f"B. {b}\n"
            f"C. {c}\n\n"
            f"### Answer:\n{answer_text}"
        )
        texts.append(text)
    return texts


def format_pubmedqa(examples):
    texts = []
    for question, context, long_answer, final_decision in zip(
        examples["question"], examples["context"], examples["long_answer"], examples["final_decision"]
    ):
        context_text = "\n\n".join(context["contexts"])
        text = (
            f"### Question:\n{question}\n\n"
            f"### Context:\n{context_text}\n\n"  # dropped MeSH and labels
            f"### Long Answer:\n{long_answer}\n\n"
            f"### Final Decision:\n{final_decision}"
        )
        texts.append(text)
    return texts
