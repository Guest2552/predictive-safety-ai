import numpy as np
import torch
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from flask import Flask, render_template, request, jsonify
import os
import google.generativeai as genai

# --- Flask App Initialization ---
app = Flask(__name__)

# --- NEW: Transformer Model Setup ---
MODEL_PATH = "./saved_model" # Path where train_transformer.py saved the model
print("--- Loading Fine-Tuned Transformer Model ---")
try:
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory not found at {MODEL_PATH}")
        print("Please run 'python train_transformer.py' first to train and save the model.")
        transformer_model = None
        tokenizer = None
        id_to_label = None
    else:
        # Load the tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
        
        # Load the fine-tuned model
        transformer_model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
        
        # Set model to evaluation mode (disables dropout, etc.)
        transformer_model.eval() 
        
        # Load the label mapping we saved during training
        id_to_label = transformer_model.config.id2label
        
        print("Transformer model and tokenizer loaded successfully.")

except Exception as e:
    print(f"Error loading Transformer model: {e}")
    transformer_model = None
    tokenizer = None
    id_to_label = None

# --- Gemini LLM Setup ---
print("--- Initializing Gemini LLM ---")
try:
    # Load API key from environment variable
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    # Initialize the Gemini model
    generation_config = {
      "temperature": 0.2,
      "top_p": 1,
      "top_k": 1,
      "max_output_tokens": 256,
    }
    llm_model = genai.GenerativeModel(model_name="gemini-2.0-flash",
                                      generation_config=generation_config)
    print("Gemini LLM model initialized successfully.")
except KeyError:
    print("Error: GOOGLE_API_KEY environment variable not set. LLM Analysis will be unavailable.")
    llm_model = None
except Exception as e:
    print(f"Error initializing Gemini: {e}")
    llm_model = None


# --- NEW: Helper function to get LLM analysis ---
def get_llm_analysis(report_text):
    if not llm_model:
        return "LLM analysis is not available (model not initialized)."
    
    prompt = f"""
    You are an expert industrial safety analyst. 
    Analyze the following incident report and provide a brief, 2-3 sentence analysis. 
    Focus on the key risks and a suggested mitigation action.
    
    Incident Report: "{report_text}"
    
    Analysis:
    """
    try:
        response = llm_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "Error: Could not retrieve analysis from the LLM."


# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the homepage."""
    return render_template('index.html')

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template('about.html')

@app.route('/demo')
def demo():
    """Renders the demo page."""
    return render_template('demo.html')

# --- MODIFIED: /predict route ---
@app.route('/predict', methods=['POST'])
def predict():
    """Handles the prediction request from the demo page."""
    local_prediction = "N/A"
    local_confidence = 0.0
    llm_analysis = "N/A"

    try:
        data = request.get_json()
        report_text = data.get('report_text', '')

        if not report_text.strip():
            return jsonify({'error': 'Report text cannot be empty'}), 400

        # --- 1. NEW: Transformer Model Prediction ---
        if not transformer_model or not tokenizer:
            return jsonify({'error': 'Local Transformer model not loaded. Run training script.'}), 500

        # Tokenize the input text
        inputs = tokenizer(report_text, return_tensors="pt", truncation=True, padding=True, max_length=256)
        
        # Get prediction
        with torch.no_grad(): # Disable gradient calculation for inference
            outputs = transformer_model(**inputs)
        
        logits = outputs.logits
        
        # Get probabilities using softmax
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        
        # Get the highest probability class ID
        predicted_class_id = torch.argmax(logits, dim=-1).item()
        
        # Map ID back to string label
        local_prediction = id_to_label[predicted_class_id]
        
        # Get the confidence score for that class
        local_confidence = float(probabilities[predicted_class_id].item())


        # --- 2. Gemini LLM Analysis (Existing) ---
        llm_analysis = get_llm_analysis(report_text)

        # --- 3. Return Combined Results ---
        return jsonify({
            'prediction': local_prediction,
            'confidence': local_confidence,
            'llm_analysis': llm_analysis,
            'processed_text': 'N/A (Using Transformer Tokenizer)' # NLTK processing no longer used
        })

    except Exception as e:
        print(f"Prediction Error: {e}") 
        return jsonify({'error': 'An error occurred during prediction.'}), 500

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')

