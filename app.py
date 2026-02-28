from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

app = FastAPI(title="Vrindavan Muni API")

MODEL_DIR = "model_en"  # model folder will be added next

device = "cpu"
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_DIR)
model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(device)
model.eval()

class ChatRequest(BaseModel):
    message: str
    max_new_tokens: int = 80

@app.get("/")
def home():
    return {"status": "Vrindavan Muni is awake 🌸"}

@app.post("/chat")
def chat(req: ChatRequest):
    prompt = (
        "You are Vrindavan Muni, a calm spiritual guide.\n"
        "You explain ideas simply, clearly, and compassionately.\n"
        "You speak to modern seekers with ancient wisdom.\n\n"
        f"User: {req.message}\nMuni:"
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=256,
    )

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=req.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    reply = decoded.split("Muni:")[-1].strip()

    return {"reply": reply}
