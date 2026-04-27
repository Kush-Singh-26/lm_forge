import os
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments,
    default_data_collator
)
from datasets import load_dataset
from forge import ForgeTrainer

def train():
    model_name = "gpt2"
    output_dir = "./outputs"
    
    # 1. Initialize Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # 2. Load Streaming Dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train", streaming=True)

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # 3. Setup Training Arguments
    # Note: ForgeTrainer will override per_device_train_batch_size and gradient_accumulation_steps
    # based on the detected environment (Colab vs. Modal).
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4, # Placeholder, overridden by forge.yaml
        max_steps=500,
        save_steps=100,
        logging_steps=10,
        learning_rate=5e-5,
        weight_decay=0.01,
        fp16=True,
        report_to="none"
    )

    # 4. Initialize ForgeTrainer
    # Automatically handles: 
    # - ForgeCallback (Hub sync, profiling)
    # - Streaming state (sample skipping)
    # - Profile-based arg overrides
    trainer = ForgeTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=default_data_collator,
        config_path="forge.yaml"
    )

    # 5. Start Training
    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    train()
