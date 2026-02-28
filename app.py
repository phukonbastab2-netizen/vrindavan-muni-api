from fastapi import FastAPI
from pydantic import BaseModel
import os
import requests

app = FastAPI(title="Vrindavan Muni API")

HF_TOKEN = os.getenv("HF_TOKEN")
HF_MODEL = "gpt2"  # you can change later

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 120

@app.get("/")
def home():
    return {"status": "Vrindavan Muni is awake 🌿"}

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = (
        "You are Vrindavan Muni, a calm spiritual guide.\n"
        "You explain ideas simply, with compassion and clarity.\n\n"
        f"User: {req.message}\nMuni:"
    )

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": req.max_new_tokens,
            "temperature": 0.7,
            "top_p": 0.9
        }
    }

    response = requests.post(API_URL, headers=HEADERS, json=payload)
    result = response.json()

    if isinstance(result, list) and "generated_text" in result[0]:
        reply = result[0]["generated_text"].split("Muni:")[-1].strip()
        return {"reply": reply}

    return {"error": result}
