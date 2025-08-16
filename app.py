from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import torch
from TTS.api import TTS
import os

# ================= INIT ==================
app = FastAPI(
    title="Conqui TTS API",
    description="Text-to-Speech API using Coqui TTS",
    version="1.0"
)

# Select device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model (single speaker that doesn’t need espeak)
model_name = "tts_models/en/ljspeech/tacotron2-DDC"
tts = TTS(model_name).to(device)

# Output folder
OUTPUT_FILE = "output.wav"


# ================= REQUEST SCHEMA ==================
class TTSRequest(BaseModel):
    text: str


# ================= API ROUTE ==================
@app.post("/tts")
def generate_tts(request: TTSRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Generate speech
    tts.tts_to_file(text=request.text, file_path=OUTPUT_FILE)

    # Return file as response
    if os.path.exists(OUTPUT_FILE):
        return FileResponse(OUTPUT_FILE, media_type="audio/wav", filename="speech.wav")
    else:
        raise HTTPException(status_code=500, detail="Failed to generate speech")


# ================= RUNNER ==================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
