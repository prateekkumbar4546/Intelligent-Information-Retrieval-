import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from rag_systems import RAGSystem
from agent_system import AgentSystem
import logging

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# Initialize RAG system and Agent system
rag_system = RAGSystem()
agent_system = AgentSystem(rag_system)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/query")
async def query(question: str = Form(...)):
    logger.info(f"Received query: {question}")
    
    try:
        text_response = agent_system.run(question)
        logger.info(f"Agent response: {text_response}")
        
        # Generate audio response using Sarvam TTS
        audio_base64 = agent_system.text_to_speech(text_response)
        
        response_content = {
            "text_response": text_response,
            "audio_response": audio_base64
        }
        
        if audio_base64 is None:
            response_content["audio_error"] = "Text-to-speech conversion failed. Audio not available."
        
        return JSONResponse(content=response_content)
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)