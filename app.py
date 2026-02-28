from fastapi import FastAPI
from pydantic import BaseModel
import requests
import os

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="Vrindavan Muni API")

# -----------------------------
# Hugging Face settings
# -----------------------------
HF_TOKEN = os.getenv("HF_TOKEN")  # set in Render Environment
MODEL_ID = "gpt2"  # free, light, works everywhere

HF_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

# -----------------------------
# Request schema
# -----------------------------
class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 120

# -----------------------------
# Health check
# -----------------------------
@app.get("/")
def home():
    return {"status": "Vrindavan Muni is awake 🕉️"}

# -----------------------------
# Chat endpoint
# -----------------------------
@app.post("/chat")
def chat(req: ChatRequest):
    prompt = (
        "You are Vrindavan Muni, a calm spiritual guide.\n"
        "You explain ideas simply, clearly, and compassionately.\n"
        "You speak to modern seekers with ancient wisdom.\n\n"
        f"User: {req.message}\n"
        "Muni:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": req.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True
        }
    }

    response = requests.post(
        HF_API_URL,
        headers=HEADERS,
        json=payload,
        timeout=60
    )

    if response.status_code != 200:
        return {
            "error": "Hugging Face API error",
            "details": response.text
        }

    result = response.json()

    # Hugging Face returns a list
    text = result[0]["generated_text"]

    # Clean reply
    reply = text.split("Muni:")[-1].strip()

    return {"reply": reply}
