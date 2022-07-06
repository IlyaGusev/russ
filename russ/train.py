import argparse
from transformers import BertConfig, AutoModelForTokenClassification, TrainingArguments, Trainer, DebertaV2Config


from russ.tokenizer import CharTokenizer
from russ.dataset import StressDataset


def train(
    train_path,
    val_path,
    batch_size,
    gradient_accumulation_steps,
    logging_steps,
    eval_steps,
    save_steps,
    learning_rate,
    warmup_steps,
    num_train_epochs,
    output_dir,
    sample_rate,
    checkpoint
):
    tokenizer = CharTokenizer()
    tokenizer.train(train_path)

    train_dataset = StressDataset(train_path, tokenizer, sample_rate=sample_rate)
    val_dataset = StressDataset(val_path, tokenizer, sample_rate=sample_rate)

    for item in train_dataset:
        print(item)
        print(tokenizer.decode(item["input_ids"], skip_special_tokens=True))
        break

    id2label = {
        0: "NO",
        1: "PRIMARY",
        2: "SECONDARY"
    }
    configuration = DebertaV2Config(
        vocab_size=tokenizer.vocab_size,
        hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        max_position_embeddings=64,
        intermediate_size=512,
        max_length=40,
        num_labels=len(id2label),
        id2label=id2label,
        label2id={label: i for i, label in id2label.items()},
        pad_token_id=tokenizer.pad_token_id
    )
    model = AutoModelForTokenClassification.from_config(configuration)
    print(model)
    params_num = sum(p.numel() for p in model.parameters())
    print(f"Params num: {params_num}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=logging_steps,
        eval_steps=eval_steps,
        evaluation_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_total_limit=1,
        load_best_model_at_end=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset
    )
    trainer.train(checkpoint)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--val-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--sample-rate", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--logging-steps", type=int, default=128)
    parser.add_argument("--eval-steps", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--warmup-steps", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=int, default=20)
    args = parser.parse_args()
    train(**vars(args))
