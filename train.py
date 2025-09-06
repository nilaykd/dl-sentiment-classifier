# train.py — IMDb fine-tune with HF (works with transformers 4.56.x)

import os
import numpy as np
from datasets import load_dataset
from packaging import version

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
)
from transformers.training_args import TrainingArguments
import evaluate

def main():
    model_name = "prajjwal1/bert-tiny"
    output_dir = "outputs"

    # 1) Data
    raw = load_dataset("imdb")
    split = raw["train"].train_test_split(test_size=0.1, seed=42)
    train_ds, val_ds = split["train"], split["test"]
    test_ds = raw["test"]

    # 2) Tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=256)
    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=["text"])
    val_ds   = val_ds.map(tok_fn,   batched=True, remove_columns=["text"])
    test_ds  = test_ds.map(tok_fn,  batched=True, remove_columns=["text"])

    # 3) Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    # 4) Metrics
    metric = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return metric.compute(predictions=preds, references=labels)

    # 5) Training args — use eval_strategy for transformers 4.56.x
    common = dict(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_steps=50,
        load_best_model_at_end=True,
        report_to=[],   # set to ["tensorboard"] if you want TB logs
        seed=42,
    )

    # 4.56.x expects eval_strategy, not evaluation_strategy
    args = TrainingArguments(
        eval_strategy="epoch",   # <— this is the key change
        save_strategy="epoch",
        **common,
    )

    # 6) Trainer
    collator = DataCollatorWithPadding(tokenizer=tok)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print("Final test:", trainer.evaluate(test_ds))

    trainer.save_model(os.path.join(output_dir, "best"))
    tok.save_pretrained(os.path.join(output_dir, "best"))

if __name__ == "__main__":
    main()
