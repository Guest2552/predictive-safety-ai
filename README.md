README.md
# Predictive Safety Risk Analysis (Industrial Incident AI App)

A Flask-based web application that predicts the **risk severity** (Low, Medium, High, Critical) of industrial safety incident reports using a **fine-tuned DistilBERT Transformer model**.  
It also provides a **qualitative analysis** using Google’s **Gemini 2.0 Flash LLM**.

---

## 🚀 Features
- Fine-tuned **DistilBERT** model for multi-class text classification.
- Integrated **Gemini LLM** for expert narrative risk analysis.
- **Interactive web UI** built with HTML, CSS, and JS.
- RESTful `/predict` endpoint for serving predictions.
- Modular training script and live demo page.

---

## 🧠 Tech Stack
- **Backend:** Flask, PyTorch, Transformers (Hugging Face)
- **Frontend:** HTML, CSS, JavaScript
- **Model:** DistilBERT (fine-tuned)
- **LLM:** Gemini 2.0 Flash API
- **Dataset:** `safety_incidents.json` (Industrial incident reports)

---

## 📂 Project Structure


📁 predictive-safety-ai
│
├── server.py # Flask app & API integration
├── train_transformer.py # Model fine-tuning script
├── safety_incidents.json # Dataset
│
├── templates/ # HTML templates
│ ├── index.html
│ ├── about.html
│ └── demo.html
│
├── static/
│ ├── css/style.css
│ ├── js/script.js
│ └── images/
│
├── saved_model/ # Generated after training
│
├── requirements.txt # Dependencies
└── README.md


---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/<your-username>/predictive-safety-ai.git
cd predictive-safety-ai

2️⃣ Create a virtual environment
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)

3️⃣ Install dependencies
pip install -r requirements.txt

4️⃣ Train the model

Before starting the app, fine-tune the Transformer:

python train_transformer.py


This saves the model in ./saved_model/.

5️⃣ Run the Flask app
export GOOGLE_API_KEY="your_api_key_here"
python server.py


Then open your browser and visit 👉 http://127.0.0.1:5000

🧪 API Endpoint

POST /predict
Request Body:

{ "report_text": "Oil leak detected in turbine chamber, minor smoke observed." }


Response:

{
  "prediction": "High",
  "confidence": 0.89,
  "llm_analysis": "This incident indicates overheating risk due to oil contamination. Immediate isolation and inspection are recommended."
}

🧩 Notes

Set your Google Gemini API key using the environment variable GOOGLE_API_KEY.

Ensure your model is trained before launching server.py.

For demo visuals, place your image in static/images/ and name it Data_Flow.png.

🧑‍💻 Author

Developed by [Your Name]
For academic and research use at Woxsen University

📜 License

MIT License © 2025 [Your Name]


---

### 📦 **requirements.txt**

```txt
Flask
torch
transformers
datasets
scikit-learn
pandas
numpy
google-generativeai
