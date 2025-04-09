from utils.timer import Timer as timer

print("📦 Importing modules/libraries...")
with timer("Module/Library importing"):
    import os
    import warnings
    import logging
    import gc
    import torch
    import time
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from datasets import Dataset, DatasetDict, load_from_disk
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainingArguments,
        Trainer
    )
    from transformers.utils import logging




# --- Suppress warnings and logs ---
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["WANDB_DISABLED"] = "true"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")
logging.get_logger("tensorflow").setLevel(logging.ERROR)

# --- Hyperparams ---
model_name = "facebook/bart-base"  # Change this to any supported model (e.g., "t5-small", etc.)
model_path = f"./fine-tuned-{model_name.split('/')[-1]}"
max_input_len = 512
max_target_len = 64



def clear_cuda_mem():
    gc.collect()
    torch.cuda.empty_cache()

def load_csv_as_dataset(file_path):
    df = pd.read_csv(file_path, low_memory=False)
    
    # Example: predict mpg (comb08) from car features. Modify as needed.
    df = df[["make", "model", "year", "cylinders", "displ", "drive", "comb08"]].dropna()

    df["text"] = df.apply(lambda row: f"{row['year']} {row['make']} {row['model']} | Cylinders: {row['cylinders']}, Displacement: {row['displ']}, Drive: {row['drive']}", axis=1)
    df["summary"] = df["comb08"].astype(str)

    train_df, test_df = train_test_split(df[["text", "summary"]], test_size=0.1)
    return DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

def preprocess(batch, tokenizer):
    inputs = tokenizer(batch["text"], max_length=max_input_len, padding="max_length", truncation=True)
    targets = tokenizer(batch["summary"], max_length=max_target_len, padding="max_length", truncation=True)

    labels = np.where(np.array(targets["input_ids"]) == tokenizer.pad_token_id, -100, targets["input_ids"])
    batch = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "labels": labels
    }
    return batch

def main():
    clear_cuda_mem()

    # ===== Load model and tokenizer =====
    print("\nLoading model...")
    try:
        with timer("🧠 Model and Tokenizer loading"):
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print("✅ Model and Tokenizer loaded!")
    except Exception as e:
        print(f"Error while loading model and tokenizer: {e}")


    # ===== Load dataset =====
    print("🗂️ Loading dataset...")
    try:
        with timer("Dataset loading"):
            dataset = load_csv_as_dataset(os.path.join(os.path.dirname(__file__), "datasets", "1984–2026-vehicles.csv"))
            tokenized_path = os.path.join(os.path.dirname(__file__), "datasets", f"tokenized_{model_name.split('/')[-1]}")
            if os.path.exists(tokenized_path):
                print(f"📂 Found cached tokenized dataset: {tokenized_path}")
                dataset = load_from_disk(tokenized_path)
            else:
                print("⚡ Tokenizing dataset...")
                with timer("Dataset Tokenization"):
                    dataset = dataset.map(lambda x: preprocess(x, tokenizer), batched=True)
                    dataset.save_to_disk(tokenized_path)
                    print("💾 Tokenized dataset saved!")
            print("\n✅ Dataset loaded!")
    except Exception as e:
        print(f"Error while loading dataset: {e}")


    # ===== Define trainer =====
    training_args = Seq2SeqTrainingArguments(
        output_dir=model_path,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        learning_rate=5e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator
    )


    # ===== Train =====
    print("\n🏁 Starting training...")
    try:
        with timer("Training"):
                trainer.train()
    except Exception as e:
        print(f"Error while training model: {e}")


    # ===== Save =====
    print("\n💾 Saving model and tokenizer...")
    try:
        trainer.save_model(model_path)
        tokenizer.save_pretrained(model_path)
        print("\n✅ Model and Tokenizer Saved!")
    except Exception as e:
        print(f"Error while saving model and tokenizer: {e}")
    

    # ===== Evaluate =====
    print("\n🧪 Evaluating...")
    try:
        with timer():
            results = trainer.evaluate()
        print(f"📊 Results: {results}")
    except Exception as e:
        print(f"Error while evaluating: {e}")
    

if __name__ == "__main__":
    print("✅ CUDA available:", torch.cuda.is_available())
    print("💎 CUDA device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
    if not torch.cuda.is_available():
        print("If you have a GPU but it shows false and none, you need to do this:")
        print("\t--->\tpip uninstall torch torchvision torchaudio -y")
        print("\tTHEN")
        print("\t--->\tpip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print("\tTHEN")
        print("\t--->\tRESTART TRAINER")
        print("If you DO NOT have a GPU you can ignore this!\n\n")

    from multiprocessing import freeze_support
    freeze_support()
    main()