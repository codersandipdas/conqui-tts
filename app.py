from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from TTS.api import TTS
import os
import uuid
from typing import Optional
from pathlib import Path

# ================= INIT ==================
app = FastAPI(
    title="Conqui TTS API",
    description="Text-to-Speech API using Coqui TTS",
    version="1.0"
)


# Create outputs folder if it doesn't exist
OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (single speaker that doesn’t need espeak)
model_name = "tts_models/en/ljspeech/tacotron2-DDC"
tts = TTS(model_name).to(device)


# ================= REQUEST SCHEMA ==================
class TTSRequest(BaseModel):
    text: str
    filename: Optional[str] = None  # optional


# ================= API ROUTE ==================
@app.post("/tts")
def generate_tts(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Generate unique filename
    base_name = request.filename.strip() if request.filename and request.filename.strip() else "output"
    file_name = OUTPUTS_DIR / f"{base_name}_{uuid.uuid4()}.wav"

    # Generate speech
    tts.tts_to_file(text=request.text, file_path=file_name)

    # Return file as response
    if os.path.exists(file_name):
        return FileResponse(file_name, media_type="audio/wav", filename="speech.wav")
    else:
        raise HTTPException(status_code=500, detail="Failed to generate speech")


# ================= RUNNER ==================
@app.on_event("startup")
async def startup_event():
    print("🚀 Conqui TTS API server is up and running at http://127.0.0.1:8000")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
