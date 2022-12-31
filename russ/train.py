import argparse
import json
from transformers import BertConfig, AutoModelForTokenClassification, TrainingArguments, Trainer, DebertaV2Config, AutoConfig

from russ.lstm import LstmModelConfig, LstmModelForTokenClassification
from russ.tokenizer import CharTokenizer
from russ.dataset import StressDataset


def train(
    train_path,
    val_path,
    config_path,
    output_dir,
    sample_rate,
    checkpoint
):
    with open(config_path) as r:
        config = json.load(r)

    tokenizer = CharTokenizer(do_lower_case=config["do_lower_case"])
    tokenizer.train(train_path)

    max_length = 40
    dataset_params = {
        "tokenizer": tokenizer,
        "sample_rate": sample_rate,
        "max_length": max_length,
        "skip_secondary": config.pop("skip_secondary", False),
        "convert_secondary": config.pop("convert_secondary", False)
    }
    train_dataset = StressDataset(train_path, **dataset_params)
    val_dataset = StressDataset(val_path, **dataset_params)

    for item in train_dataset:
        print(item)
        print(tokenizer.decode(item["input_ids"], skip_special_tokens=True))
        break

    id2label = config.pop("id2label", {
        0: "NO",
        1: "PRIMARY",
        2: "SECONDARY"
    })
    model_params = config["model"]
    model_type = model_params.pop("model_type")
    configuration = AutoConfig.for_model(
        model_type,
        **model_params,
        vocab_size=tokenizer.vocab_size,
        max_length=max_length,
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
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        logging_steps=config["logging_steps"],
        eval_steps=config["eval_steps"],
        evaluation_strategy="steps",
        save_steps=config["save_steps"],
        learning_rate=config["learning_rate"],
        lr_scheduler_type="constant",
        warmup_steps=config["warmup_steps"],
        num_train_epochs=config["num_train_epochs"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
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
    parser.add_argument("--config-path", required=True)
    args = parser.parse_args()
    train(**vars(args))
