import pandas as pd
import json
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)
from datasets import Dataset

# --- Configuration ---
JSON_FILE_PATH = 'F:/GOKULRAJ FILE/WOXSEN/Sem 4/NLP/Final/safety_website/safety_incidents.json'
MODEL_NAME = 'distilbert-base-uncased'
SAVE_PATH = './saved_model'

# --- 1. Load and Prepare Data ---
print(f"--- Loading data from {JSON_FILE_PATH} ---")
try:
    with open(JSON_FILE_PATH, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
except Exception as e:
    print(f"Error loading data: {e}")
    exit()

# Drop rows where 'incident_report' or 'risk_level' is missing
df = df.dropna(subset=['incident_report', 'risk_level'])

print(f"Loaded {len(df)} records.")

# --- 2. Create Label Mappings ---
# Transformer models need integer labels, not strings.
labels = df['risk_level'].unique()
labels.sort() # Ensure consistent order: ['Critical', 'High', 'Low', 'Medium']
# We'll re-map for better ordinality, though the model doesn't require it
label_map = {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3}
id_to_label_map = {v: k for k, v in label_map.items()}

# Apply mapping
df['label'] = df['risk_level'].map(label_map)

# Handle any labels that might not be in our map (if data is messy)
df = df.dropna(subset=['label'])
df['label'] = df['label'].astype(int)

print(f"\nLabel mapping created: {label_map}")

# --- 3. Split Data ---
print("--- Splitting data into training and validation sets ---")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['incident_report'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label'].tolist() # Ensure class balance
)

# --- 4. Load Tokenizer and Model ---
print(f"--- Loading pre-trained model: {MODEL_NAME} ---")

# Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

# Load model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_map),
    id2label=id_to_label_map, # Attach mappings to model config
    label2id=label_map
)

# --- 5. Tokenize Data ---
print("--- Tokenizing datasets ---")

# Tokenization function
def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

# Create Dataset objects
train_dataset = Dataset.from_dict({'text': train_texts, 'label': train_labels})
val_dataset = Dataset.from_dict({'text': val_texts, 'label': val_labels})

# Apply tokenization
tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

print("Tokenization complete.")

# --- 6. Define Metrics ---
def compute_metrics(eval_pred):
    """Calculates accuracy and F1 score for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='macro')
    
    return {
        'accuracy': acc,
        'f1_macro': f1,
    }

# --- 7. Set Up Trainer ---
print("--- Configuring model trainer ---")

training_args = TrainingArguments(
    output_dir='./training_checkpoints', # Directory to save checkpoints
    num_train_epochs=3,               # Total number of training epochs
    per_device_train_batch_size=8,    # Batch size for training
    per_device_eval_batch_size=8,     # Batch size for evaluation
    warmup_steps=50,                  # Number of warmup steps
    weight_decay=0.01,                # Strength of weight decay
    logging_dir='./logs',             # Directory for logs
    logging_steps=50,
    evaluation_strategy="epoch",      # Evaluate at the end of each epoch
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    compute_metrics=compute_metrics,
)

# --- 8. Train the Model ---
print("\n--- !!! STARTING MODEL TRAINING !!! ---")
trainer.train()
print("--- !!! TRAINING COMPLETE !!! ---")

# --- 9. Save the Final Model & Tokenizer ---
print(f"--- Saving fine-tuned model and tokenizer to {SAVE_PATH} ---")
model.save_pretrained(SAVE_PATH)
tokenizer.save_pretrained(SAVE_PATH)

print("\n--- Project setup complete! ---")
print(f"Your new model is saved in '{SAVE_PATH}'.")
print("You can now run 'python server.py' to start the application.")
