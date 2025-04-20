import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from datasets import Dataset
from sklearn.model_selection import train_test_split

print("Starting self-training process...")

# Handle directory structure
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Step 1: Load high confidence predictions from the balanced dataset
print("Step 1: Loading high confidence predictions from balanced dataset...")
high_conf_path = os.path.join(parent_dir, "reddit_stress_high_confidence_balanced.csv")

if not os.path.exists(high_conf_path):
    # Try the old path as fallback
    old_path = os.path.join(parent_dir, "reddit_stress_high_confidence.csv")
    if os.path.exists(old_path):
        print(f"Warning: Balanced high confidence file not found. Using original high confidence file instead.")
        high_conf_path = old_path
    else:
        raise ValueError(f"High confidence predictions file not found! Run reddit_preprocess.py first.")

high_conf_df = pd.read_csv(high_conf_path)
print(f"Loaded {len(high_conf_df)} high confidence predictions")

# Step 2: Split the high confidence data for validation
print("Step 2: Splitting high confidence data for validation...")
train_df, val_df = train_test_split(
    high_conf_df, 
    test_size=0.2,
    stratify=high_conf_df["predicted_stress"],
    random_state=42
)

print(f"Training set: {len(train_df)} examples")
print(f"Validation set: {len(val_df)} examples")

# Step 3: Load the original model
print("Step 3: Loading the original model...")
if os.path.exists("./deberta-stress-model-retrain-global"):
    model_path = "./deberta-stress-model-retrain-global"
elif os.path.exists(os.path.join(parent_dir, "deberta-stress-model-retrain-global")):
    model_path = os.path.join(parent_dir, "deberta-stress-model-retrain-global")
else:
    raise ValueError("Model directory not found! Please train the model first.")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Step 4: Prepare datasets for self-training
print("Step 4: Preparing datasets for self-training...")

# Create Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

# Tokenize datasets
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# Use predicted_stress as the label
tokenized_train = tokenized_train.rename_column("predicted_stress", "label")
tokenized_val = tokenized_val.rename_column("predicted_stress", "label")

# Step 5: Define training arguments
print("Step 5: Setting up training arguments...")

# Create a new directory for the self-trained model
self_trained_model_dir = os.path.join(parent_dir, "self_trained_model")
os.makedirs(self_trained_model_dir, exist_ok=True)

# Define training arguments with compatibility for older versions
training_args = TrainingArguments(
    output_dir=self_trained_model_dir,
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    save_steps=500,             # Save model every 500 steps instead of using evaluation_strategy
    save_total_limit=1,         # Keep only the best model
    logging_steps=50,
    do_eval=True,               # Enable evaluation
    eval_steps=100,              # Evaluate every 100 steps
    label_smoothing_factor=0.1,  # helps with overconfidence

    eval_strategy="steps",  
    save_strategy="steps", 

    # Added for early stopping
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1",   # or e.g. "accuracy" if using accuracy
    greater_is_better=True,
)

# Step 6: Define trainer and train model
print("Step 6: Training the model...")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = np.mean(predictions == labels)
    
    # Calculate precision, recall, and F1 for the positive class (stress)
    true_positives = np.sum((predictions == 1) & (labels == 1))
    false_positives = np.sum((predictions == 1) & (labels == 0))
    false_negatives = np.sum((predictions == 0) & (labels == 1))
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],  # Stops if no improvement for 2 evals
)

# Train the model
print("Starting training...")
trainer.train()

# Step 7: Save the self-trained model
print("Step 7: Saving the self-trained model...")
trainer.save_model(self_trained_model_dir)
tokenizer.save_pretrained(self_trained_model_dir)

# Step 8: Evaluate the self-trained model on the validation set
print("Step 8: Evaluating the self-trained model...")
evaluation_result = trainer.evaluate()

# Handle the case where the metric names might be different in older versions
accuracy = evaluation_result.get('eval_accuracy', evaluation_result.get('accuracy', 0))
precision = evaluation_result.get('eval_precision', evaluation_result.get('precision', 0))
recall = evaluation_result.get('eval_recall', evaluation_result.get('recall', 0))
f1 = evaluation_result.get('eval_f1', evaluation_result.get('f1', 0))

print(f"Validation metrics:")
print(f"  - Accuracy: {accuracy:.4f}")
print(f"  - Precision: {precision:.4f}")
print(f"  - Recall: {recall:.4f}")
print(f"  - F1 Score: {f1:.4f}")

print("\nSelf-training complete!")
print(f"Self-trained model saved to: {self_trained_model_dir}")

# Optional Step 9: Apply the self-trained model to the entire dataset
print("\nStep 9: Applying self-trained model to entire dataset...")

# Load all predictions
all_predictions_path = os.path.join(parent_dir, "reddit_stress_all_predictions.csv")
if os.path.exists(all_predictions_path):
    # Load the entire dataset
    all_df = pd.read_csv(all_predictions_path)
    print(f"Loaded {len(all_df)} comments from full dataset")
    
    # Create dataset for inference
    all_dataset = Dataset.from_pandas(all_df)
    tokenized_all = all_dataset.map(preprocess_function, batched=True)
    
    # Run inference
    print("Running inference with self-trained model...")
    
    # Use the trainer to get raw predictions
    raw_predictions = trainer.predict(tokenized_all)
    
    # Convert logits to probabilities and predictions
    logits = raw_predictions.predictions
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
    predictions = np.argmax(probs, axis=1)
    confidence = np.max(probs, axis=1)
    
    # Update the dataframe with new predictions
    all_df["self_trained_prediction"] = predictions
    all_df["self_trained_confidence"] = confidence
    
    # Save updated predictions
    all_df.to_csv(os.path.join(parent_dir, "reddit_stress_self_trained_predictions.csv"), index=False)
    print(f"Updated predictions saved to: {os.path.join(parent_dir, 'reddit_stress_self_trained_predictions.csv')}")
    
    # Print statistics of the new predictions
    old_stress_count = (all_df["predicted_stress"] == 1).sum()
    new_stress_count = (all_df["self_trained_prediction"] == 1).sum()
    old_stress_pct = old_stress_count / len(all_df) * 100
    new_stress_pct = new_stress_count / len(all_df) * 100
    
    print("\nPrediction statistics:")
    print(f"Original model: {old_stress_count} stress comments ({old_stress_pct:.2f}%)")
    print(f"Self-trained model: {new_stress_count} stress comments ({new_stress_pct:.2f}%)")
    
    # Calculate agreement rate
    agreement = (all_df["predicted_stress"] == all_df["self_trained_prediction"]).mean() * 100
    print(f"Agreement between models: {agreement:.2f}%")
    
    # Calculate metrics for original vs. self-trained
    # Assuming original predictions as "ground truth" for comparative purposes
    true_positives = ((all_df["predicted_stress"] == 1) & (all_df["self_trained_prediction"] == 1)).sum()
    false_positives = ((all_df["predicted_stress"] == 0) & (all_df["self_trained_prediction"] == 1)).sum()
    false_negatives = ((all_df["predicted_stress"] == 1) & (all_df["self_trained_prediction"] == 0)).sum()
    true_negatives = ((all_df["predicted_stress"] == 0) & (all_df["self_trained_prediction"] == 0)).sum()
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\nComparison metrics (treating original model as baseline):")
    print(f"  - Precision: {precision:.4f}")
    print(f"  - Recall: {recall:.4f}")
    print(f"  - F1 Score: {f1:.4f}")
    
    # Calculate confidence distribution for changed predictions
    changed_predictions = all_df[all_df["predicted_stress"] != all_df["self_trained_prediction"]]
    changed_count = len(changed_predictions)
    
    print(f"\nChanged predictions: {changed_count} ({changed_count/len(all_df)*100:.2f}% of total)")
    if changed_count > 0:
        avg_original_conf = changed_predictions["confidence_score"].mean()
        avg_new_conf = changed_predictions["self_trained_confidence"].mean()
        print(f"  - Average original confidence for changed predictions: {avg_original_conf:.4f}")
        print(f"  - Average new confidence for changed predictions: {avg_new_conf:.4f}")
        
        # Save changed predictions for analysis
        changed_predictions.to_csv(os.path.join(parent_dir, "reddit_stress_changed_predictions.csv"), index=False)
        print(f"  - Changed predictions saved to: {os.path.join(parent_dir, 'reddit_stress_changed_predictions.csv')}")
    
else:
    print(f"All predictions file not found at {all_predictions_path}, skipping full prediction.")

print("\nProcess complete!")
