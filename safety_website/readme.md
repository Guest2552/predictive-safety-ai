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
