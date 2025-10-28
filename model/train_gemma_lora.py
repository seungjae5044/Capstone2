import argparse
import math
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from transformers import Trainer as HFTrainer
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Gemma-3-270m")
    parser.add_argument(
        "--dataset",
        type=str,
        default=str(Path(__file__).resolve().parent / "gemma_lora_dataset.jsonl"),
        help="JSONL dataset with prompt/response fields.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Path(__file__).resolve().parent / "gemma3_lora"),
        help="Directory to store LoRA adapters and checkpoints.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Per-device train batch size.",
    )
    parser.add_argument(
        "--grad_accum_steps",
        type=int,
        default=2,
        help="Gradient accumulation steps.",
    )
    parser.add_argument(
        "--epochs",
        type=float,
        default=3.0,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,
        help="Learning rate.",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Enable QLoRA 4-bit loading (saves VRAM).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Checkpoint directory to resume from (optional).",
    )
    return parser.parse_args()


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": tokenizer.eos_token})
    tokenizer.padding_side = "right"
    return tokenizer


def make_lora_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )


def main():
    args = parse_args()
    model_name = "google/gemma-3-270m"

    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    data_files = {"train": str(dataset_path)}
    raw_dataset = load_dataset("json", data_files=data_files)["train"]
    split_dataset = raw_dataset.train_test_split(test_size=0.05, seed=42)

    tokenizer = load_tokenizer(model_name)

    quant_config = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    if args.use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
    )
    if quant_config is None and torch.cuda.is_available():
        model.to(torch.device("cuda"))

    if tokenizer.pad_token_id != tokenizer.eos_token_id:
        model.resize_token_embeddings(len(tokenizer))

    sft_config = SFTConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_strategy="steps",
        logging_steps=20,
        eval_strategy="steps",
        eval_steps=200,
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=torch_dtype == torch.bfloat16,
        fp16=torch_dtype == torch.float16,
        report_to=[],
        do_eval=True,
        max_length=2048,
        dataset_text_field="text",
        resume_from_checkpoint=args.resume,
    )

    def to_text(example):
        return {"text": f"{example['prompt']}{example['response']}"}

    train_columns = split_dataset["train"].column_names
    eval_columns = split_dataset["test"].column_names
    train_dataset = split_dataset["train"].map(to_text, remove_columns=train_columns)
    eval_dataset = split_dataset["test"].map(to_text, remove_columns=eval_columns)

    class MySFTTrainer(SFTTrainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            return HFTrainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)

    trainer = MySFTTrainer(
        model=model,
        args=sft_config,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=make_lora_config(),
    )

    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model()
    trainer.save_state()

    adapter_path = Path(args.output_dir) / "adapter_model"
    adapter_path.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    total_steps = math.ceil(len(split_dataset["train"]) / (args.batch_size))
    print(f"✅ LoRA fine-tuning complete. Checkpoints saved at {args.output_dir}.")
    print(f"   Total train samples: {len(split_dataset['train'])}, steps ≈ {total_steps}")


if __name__ == "__main__":
    main()
