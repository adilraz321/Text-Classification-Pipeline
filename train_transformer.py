# src/train_transformer.py
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# ✅ Detect device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔥 Using device: {device}")

def df_to_hf_dataset(df):
    """Convert pandas dataframe to HuggingFace dataset"""
    return Dataset.from_pandas(df[['text_clean', 'label']].rename(columns={'text_clean': 'text'}))

def compute_metrics(pred):
    """Compute accuracy and F1"""
    labels = pred.label_ids
    preds = np.argmax(pred.predictions, axis=1)
    acc = (preds == labels).mean()
    f1 = f1_score(labels, preds)
    return {'accuracy': acc, 'f1': f1}

def main():
    # ✅ Load preprocessed CSVs
    train_df = pd.read_csv('data/train.csv')
    val_df = pd.read_csv('data/val.csv')
    test_df = pd.read_csv('data/test.csv')

    train_ds = df_to_hf_dataset(train_df)
    val_ds = df_to_hf_dataset(val_df)
    test_ds = df_to_hf_dataset(test_df)

    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)

    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # ✅ Load model and move to GPU if available
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    # ✅ Optimized training arguments for ~5 min training
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        save_strategy='epoch',
        per_device_train_batch_size=16 if torch.cuda.is_available() else 4,  # larger batch for GPU
        per_device_eval_batch_size=32 if torch.cuda.is_available() else 8,
        num_train_epochs=1,  # only 1 epoch for quick training
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model='accuracy',
        fp16=torch.cuda.is_available(),  # use mixed precision on GPU
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    print("\n🚀 Starting training...")
    trainer.train()

    print("\n📊 Evaluating on test set...")
    eval_res = trainer.evaluate(eval_dataset=test_ds)
    print(eval_res)

    print("\n💾 Saving model to 'models/bert_sentiment' ...")
    trainer.save_model('models/bert_sentiment')

if __name__ == '__main__':
    main()
