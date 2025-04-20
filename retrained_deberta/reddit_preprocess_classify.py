import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import numpy as np
import glob
import os
from sklearn.utils import resample

print("Starting Reddit mental health analysis...")

# Handle directory structure - code in subfolder, CSVs in parent directory
# Get the current directory and parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

print(f"Current directory: {current_dir}")
print(f"Parent directory: {parent_dir}")

# Step 1: Preprocess Reddit data
print("Step 1: Preprocessing Reddit data...")

# Map each filename to a subreddit label
file_map = {
    "nus_comments.csv": "NUS",
    "NationalServiceSG_comments.csv": "NationalServiceSG",
    "SGExams_comments.csv": "SGExams",
    "singapore_comments.csv": "Singapore"
}

# Read and process each CSV from the parent directory
dfs = []
total_deleted_removed = 0

for filename, subreddit in file_map.items():
    # Use parent directory path for CSV files
    file_path = os.path.join(parent_dir, filename)
    if os.path.exists(file_path):
        print(f"Processing {file_path}...")
        df = pd.read_csv(file_path)
        
        # Count comments before filtering
        comments_before = len(df)
        
        # Filter out [deleted] and [removed] comments
        deleted_mask = (df["comment_body"].str.strip() == "[deleted]") | (df["comment_body"].str.strip() == "[removed]")
        deleted_count = deleted_mask.sum()
        df = df[~deleted_mask]
        
        # Count comments after filtering
        comments_after = len(df)
        subreddit_deleted = comments_before - comments_after
        total_deleted_removed += subreddit_deleted
        
        print(f"  - Removed {subreddit_deleted} [deleted]/[removed] comments from {subreddit}")
        
        df["source_subreddit"] = subreddit  # Tag each row with its subreddit
        
        # Create text field combining post title and comment body
        df["text"] = df["comment_body"].fillna("")
        
        # Keep only what we need
        df_slim = df[["text", "source_subreddit", "post_id", "comment_id"]]
        dfs.append(df_slim)
    else:
        print(f"Warning: {file_path} not found, skipping")

print(f"Total [deleted]/[removed] comments filtered out: {total_deleted_removed}")

# Combine them
if dfs:
    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined {len(combined_df)} comments from {len(dfs)} subreddits")
else:
    raise ValueError("No Reddit data files found! Please check file paths.")

# Step 2: Load the trained model
print("Step 2: Loading trained model...")
# Check for model in current directory first, then parent directory
if os.path.exists("./deberta-stress-model-retrain-global"):
    model_path = "./deberta-stress-model-retrain-global"
elif os.path.exists(os.path.join(parent_dir, "deberta-stress-model-retrain-global")):
    model_path = os.path.join(parent_dir, "deberta-stress-model-retrain-global")
else:
    raise ValueError("Model directory not found in current or parent directory! Please train the model first.")

print(f"Loading model from: {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Step A: Determine the max sequence length from the model
# This will help us ensure consistent padding for all inputs
if hasattr(model.config, 'max_position_embeddings'):
    max_length = model.config.max_position_embeddings
else:
    # If not specified in model config, use a reasonable default
    max_length = 512
    
print(f"Using max sequence length: {max_length}")

# Step 3: Make predictions with confidence scores
print("Step 3: Generating predictions with confidence scores...")

# Set up a standard batching and processing function
def process_in_batches(dataframe, batch_size=4):
    total = len(dataframe)
    predictions = []
    confidence_scores = []
    
    # Process data in small batches to avoid memory issues
    for start_idx in range(0, total, batch_size):
        # Print progress more frequently
        if start_idx % (batch_size * 5) == 0:
            print(f"Processing batch {start_idx//batch_size + 1}/{(total+batch_size-1)//batch_size}...")
        
        end_idx = min(start_idx + batch_size, total)
        batch_texts = dataframe['text'][start_idx:end_idx].tolist()
        
        # Properly tokenize with explicit padding to max_length
        inputs = tokenizer(
            batch_texts,
            padding='max_length',  # Always pad to max_length to ensure consistent shapes
            truncation=True,       # Truncate if needed
            max_length=max_length, # Use the identified max_length
            return_tensors='pt'    # Return PyTorch tensors
        )
        
        # Move to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model.cuda()
        
        # Run prediction
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            
            # Get predicted class and confidence
            batch_preds = torch.argmax(probs, dim=1).cpu().numpy()
            batch_confidence = torch.max(probs, dim=1).values.cpu().numpy()
            
            predictions.extend(batch_preds)
            confidence_scores.extend(batch_confidence)
    
    return predictions, confidence_scores

# Set model to evaluation mode
model.eval()

# Process all data
print(f"Processing {len(combined_df)} comments...")
preds, confidence_scores = process_in_batches(combined_df, batch_size=4)

# Step 4: Attach predictions and confidence scores to original data
print("Step 4: Attaching predictions to data...")
combined_df["predicted_stress"] = preds
combined_df["confidence_score"] = confidence_scores

# Step 5: Perform class balancing on predicted stress
print("Step 5: Performing class balancing on predicted stress data...")

# Find imbalanced class counts
stress_df = combined_df[combined_df["predicted_stress"] == 1]
non_stress_df = combined_df[combined_df["predicted_stress"] == 0]
stress_count = len(stress_df)
non_stress_count = len(non_stress_df)

print(f"Original class distribution:")
print(f"  - Stress class (1): {stress_count} ({stress_count/len(combined_df)*100:.2f}%)")
print(f"  - Non-stress class (0): {non_stress_count} ({non_stress_count/len(combined_df)*100:.2f}%)")

# Determine which class is the minority
if stress_count < non_stress_count:
    print("Stress class (1) is the minority - upsampling...")
    minority_class = stress_df
    majority_class = non_stress_df
    upsample_size = non_stress_count
else:
    print("Non-stress class (0) is the minority - upsampling...")
    minority_class = non_stress_df
    majority_class = stress_df
    upsample_size = stress_count

# Upsample minority class to match majority class
minority_upsampled = resample(
    minority_class,
    replace=True,              # Sample with replacement
    n_samples=upsample_size,   # Match majority class count
    random_state=42            # For reproducibility
)

# Combine the upsampled minority class with the majority class
balanced_df = pd.concat([majority_class, minority_upsampled])

# Shuffle the balanced dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Recalculate the class distribution
balanced_stress_count = len(balanced_df[balanced_df["predicted_stress"] == 1])
balanced_non_stress_count = len(balanced_df[balanced_df["predicted_stress"] == 0])

print(f"Balanced class distribution:")
print(f"  - Stress class (1): {balanced_stress_count} ({balanced_stress_count/len(balanced_df)*100:.2f}%)")
print(f"  - Non-stress class (0): {balanced_non_stress_count} ({balanced_non_stress_count/len(balanced_df)*100:.2f}%)")

# Step 6: Filter by confidence threshold (now using balanced dataset)
print("Step 6: Filtering high-confidence predictions from balanced dataset...")

# Define confidence thresholds
high_confidence_threshold = 0.9
medium_confidence_threshold = 0.7

# Filter by confidence from the balanced dataset
high_confidence_df = balanced_df[balanced_df["confidence_score"] >= high_confidence_threshold]
medium_confidence_df = balanced_df[(balanced_df["confidence_score"] >= medium_confidence_threshold) & 
                                  (balanced_df["confidence_score"] < high_confidence_threshold)]
low_confidence_df = balanced_df[balanced_df["confidence_score"] < medium_confidence_threshold]

# Print statistics about high confidence predictions in balanced dataset
high_conf_stress_count = len(high_confidence_df[high_confidence_df["predicted_stress"] == 1])
high_conf_non_stress_count = len(high_confidence_df[high_confidence_df["predicted_stress"] == 0])

print(f"High confidence predictions after balancing:")
print(f"  - Total high confidence: {len(high_confidence_df)}")
print(f"  - High confidence stress (1): {high_conf_stress_count} ({high_conf_stress_count/len(high_confidence_df)*100:.2f}%)")
print(f"  - High confidence non-stress (0): {high_conf_non_stress_count} ({high_conf_non_stress_count/len(high_confidence_df)*100:.2f}%)")

# Save datasets
output_dir = parent_dir  # Save results to parent directory
combined_df.to_csv(os.path.join(output_dir, "reddit_stress_all_predictions.csv"), index=False)
balanced_df.to_csv(os.path.join(output_dir, "reddit_stress_balanced.csv"), index=False)
high_confidence_df.to_csv(os.path.join(output_dir, "reddit_stress_high_confidence_balanced.csv"), index=False)

# Count high-confidence stress predictions
high_conf_stress_count = len(high_confidence_df[high_confidence_df["predicted_stress"] == 1])

print(f"\nAnalysis complete!")
print(f"Total comments analyzed: {len(combined_df)}")
print(f"Comments in balanced dataset: {len(balanced_df)}")
print(f"High confidence predictions from balanced dataset (>={high_confidence_threshold}): {len(high_confidence_df)}")
print(f"Medium confidence predictions from balanced dataset ({medium_confidence_threshold}-{high_confidence_threshold}): {len(medium_confidence_df)}")
print(f"Low confidence predictions from balanced dataset (<{medium_confidence_threshold}): {len(low_confidence_df)}")
print(f"High confidence stress comments from balanced dataset: {high_conf_stress_count}")

# Step 7: Generate subreddit statistics
print("\nStep 7: Generating subreddit statistics...")

# Group by subreddit and calculate statistics for original dataset
subreddit_stats = combined_df.groupby("source_subreddit").agg(
    total_comments=("text", "count"),
    stress_percentage=("predicted_stress", lambda x: x.mean() * 100),  # Convert to percentage
    avg_confidence=("confidence_score", "mean"),
    high_confidence_stress=("predicted_stress", lambda x: 
                           ((combined_df.loc[x.index]["predicted_stress"] == 1) & 
                           (combined_df.loc[x.index]["confidence_score"] >= high_confidence_threshold)).sum())
).reset_index()

# Calculate percentage of high-confidence stress comments
subreddit_stats["high_confidence_stress_pct"] = subreddit_stats["high_confidence_stress"] / subreddit_stats["total_comments"] * 100

# Sort by stress percentage
subreddit_stats = subreddit_stats.sort_values("stress_percentage", ascending=False)

print("\nSubreddit Statistics (Original dataset):")
print(subreddit_stats.to_string(index=False))

# Save statistics to parent directory
subreddit_stats.to_csv(os.path.join(output_dir, "subreddit_stress_statistics.csv"), index=False)

print("\nResults saved to parent directory:")
print(f"- {os.path.join(output_dir, 'reddit_stress_all_predictions.csv')} (all predictions)")
print(f"- {os.path.join(output_dir, 'reddit_stress_balanced.csv')} (balanced dataset)")
print(f"- {os.path.join(output_dir, 'reddit_stress_high_confidence_balanced.csv')} (high confidence predictions from balanced dataset)")
print(f"- {os.path.join(output_dir, 'subreddit_stress_statistics.csv')} (subreddit statistics)")
