import torch
import sys
import time
print("Starting script...", flush=True)
print(f"Python version: {sys.version}", flush=True)
print(f"PyTorch version: {torch.__version__}", flush=True)
print(f"CUDA available: {torch.cuda.is_available()}", flush=True)

start_time = time.time()
print(f"Importing transformers at {time.time() - start_time:.2f}s", flush=True)

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# Path to your model
model_path = "./deberta-stress-model-retrain-global"
print(f"Model path: {model_path}", flush=True)
print(f"Model path exists: {os.path.exists(model_path)}", flush=True)

# Sample sentences
sentences = [
    "I'm so overwhelmed with everything lately.",
    "Had a relaxing day at the beach today.",
    "Deadlines are piling up and I can't cope.",
    "I just had dinner with my family. Feeling good.",
    "I don't think I can handle this anymore."
]

print(f"Loading tokenizer at {time.time() - start_time:.2f}s", flush=True)
try:
    # Load tokenizer with verbose output
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f"Tokenizer loaded at {time.time() - start_time:.2f}s", flush=True)
    
    # Load model with verbose output
    print(f"Loading model at {time.time() - start_time:.2f}s", flush=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    print(f"Model loaded at {time.time() - start_time:.2f}s", flush=True)
    
    # Check if CUDA is available and move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    model.to(device)
    model.eval()
    print(f"Model moved to device at {time.time() - start_time:.2f}s", flush=True)
    
    # Process each sentence
    for i, sentence in enumerate(sentences):
        print(f"Processing sentence {i+1}: '{sentence}'", flush=True)
        
        # Tokenize
        inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Inference
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = probs[0][predicted_class].item()
        
        # Print result
        label = "Stress" if predicted_class == 1 else "Non-Stress"
        print(f"Result: {label} (confidence: {confidence:.4f})", flush=True)
    
    print(f"All sentences processed at {time.time() - start_time:.2f}s", flush=True)

except Exception as e:
    print(f"Error: {str(e)}", flush=True)
    import traceback
    traceback.print_exc()

print("Script completed!", flush=True)
