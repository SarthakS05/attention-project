import random
import re

import torch
from datasets import Dataset
from torch.optim import Adam
from transformers import AutoModelForCausalLM, AutoTokenizer


def configure_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    return device


def build_s1_synthetic(n=1000):
    samples = []
    for _ in range(n):
        a = random.randint(20, 99)
        b = random.randint(20, 99)
        question = f"Compute {a} + {b}. Provide the final answer as an integer."
        answer = a + b

        reasoning = (
            f"Let's solve this carefully.\n"
            f"{a} is composed of tens and ones: {a // 10}*10 + {a % 10}.\n"
            f"{b} is composed of tens and ones: {b // 10}*10 + {b % 10}.\n"
            f"Adding them: {a} + {b} = {answer}.\n"
        )

        full_trace = (
            f"Question: {question}\n"
            f"Reasoning:\n{reasoning}"
            f"Final Answer: {answer}"
        )

        samples.append({
            "question": question,
            "reasoning_trace": full_trace,
            "answer": str(answer),
        })
    return Dataset.from_list(samples)


def load_model(base_model: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    return tokenizer, model, device


def encode_batch(tokenizer, texts, device, max_length=512):
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def collate_supervised_batch(tokenizer, ds, idxs, device):
    batch = ds.select(idxs)
    texts = batch["reasoning_trace"]
    input_ids, attention_mask = encode_batch(tokenizer, texts, device)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def extract_final_answer(text: str) -> str:
    match = re.search(r"Final Answer:\s*([-+]?\d+)", text)
    if match:
        return match.group(1)
    nums = re.findall(r"-?\d+", text)
    return nums[-1] if nums else ""


def generate_with_budget(model, tokenizer, question: str, device, base_budget: int = 64, extra_waits: int = 0) -> str:
    prompt = f"Question: {question}\nReasoning:\nLet's think step by step.\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=base_budget,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    if extra_waits > 0:
        wait_suffix = "\n" + "Wait.\n" * extra_waits
        prompt2 = text + wait_suffix
        inputs2 = tokenizer(prompt2, return_tensors="pt").to(device)
        with torch.no_grad():
            out2 = model.generate(
                **inputs2,
                max_new_tokens=base_budget,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        text = tokenizer.decode(out2[0], skip_special_tokens=True)

    return text


def evaluate_with_budget(model, tokenizer, ds, device, n_samples: int = 100, base_budget: int = 64, extra_waits: int = 0):
    correct = 0
    total = min(len(ds), n_samples)
    for i in range(total):
        item = ds[i]
        q = item["question"]
        gt = item["answer"]
        traj = generate_with_budget(model, tokenizer, q, device, base_budget=base_budget, extra_waits=extra_waits)
        pred = extract_final_answer(traj)
        if pred == gt:
            correct += 1

    acc = correct / total
    print(f"Budget eval - base={base_budget}, extra_waits={extra_waits} -> accuracy {acc:.3f} on {total} examples")
    return acc


def main():
    device = configure_device()
    tokenizer, model, device = load_model("Qwen/Qwen2.5-7B-Instruct")

    train_ds = build_s1_synthetic(800)
    eval_ds = build_s1_synthetic(200)
    print("Train size:", len(train_ds), "Eval size:", len(eval_ds))

    optimizer = Adam(model.parameters(), lr=1e-5)
    epochs = 2
    batch_size = 4

    for epoch in range(epochs):
        indices = list(range(len(train_ds)))
        random.shuffle(indices)

        total_loss = 0.0
        steps = 0

        for i in range(0, len(indices), batch_size):
            batch_idxs = indices[i:i + batch_size]
            if len(batch_idxs) < batch_size:
                continue

            batch = collate_supervised_batch(tokenizer, train_ds, batch_idxs, device)
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["input_ids"],
            )
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            steps += 1

        print(f"[SFT] Epoch {epoch + 1}/{epochs} — Loss {total_loss / max(steps, 1):.4f}")

    model.eval()
    evaluate_with_budget(model, tokenizer, eval_ds, device, n_samples=50, base_budget=64, extra_waits=0)
    evaluate_with_budget(model, tokenizer, eval_ds, device, n_samples=50, base_budget=64, extra_waits=3)
    evaluate_with_budget(model, tokenizer, eval_ds, device, n_samples=50, base_budget=64, extra_waits=6)


if __name__ == "__main__":
    main()
