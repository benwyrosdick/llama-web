from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from typing import List, Dict, Any, Optional
import httpx
import json
import os
import uuid
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory storage for chat sessions (in production, use a database)
chat_sessions: Dict[str, List[Dict[str, str]]] = {}

class ChatMessage(BaseModel):
    role: str  # 'user' or 'assistant'
    content: str

class ChatRequest(BaseModel):
    content: str
    session_id: Optional[str] = None
    history: Optional[List[Dict[str, str]]] = None

# Ollama API configuration
OLLAMA_API_URL = os.getenv("OLLAMA_API_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama3.2")

def format_chat_history(messages: List[Dict[str, str]]) -> str:
    """Format chat history into a prompt for the model."""
    formatted = []
    for msg in messages:
        role = "User" if msg["role"] == "user" else "Assistant"
        formatted.append(f"{role}: {msg['content']}")
    return "\n".join(formatted)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(message: dict):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                OLLAMA_API_URL,
                json={
                    "model": DEFAULT_MODEL,
                    "prompt": message.get("content", ""),
                    "stream": False
                },
                timeout=60.0
            )
            response.raise_for_status()
            return response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=500, detail=f"Error connecting to Ollama: {str(e)}")

@app.post("/api/chat/stream")
async def chat_stream(chat_request: ChatRequest):
    """Handle streaming chat with conversation history."""
    print(f"Received chat request: {chat_request}")
    
    # Get or create session
    session_id = chat_request.session_id or str(uuid.uuid4())
    if session_id not in chat_sessions:
        chat_sessions[session_id] = []
    
    # Add user message to history
    chat_sessions[session_id].append({"role": "user", "content": chat_request.content})
    
    # Keep only the last 10 messages to avoid context window issues
    recent_messages = chat_sessions[session_id][-10:]
    
    # Format the prompt with conversation history
    history_text = format_chat_history(recent_messages)
    prompt = f"""Continue the conversation based on the following context. 
Be helpful, concise, and accurate in your responses.

{history_text}

Assistant:"""
    
    async def get_ollama_models():
        """Check if Ollama is running and get available models."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                # First try the /api/tags endpoint (newer versions)
                try:
                    response = await client.get("http://localhost:11434/api/tags")
                    if response.status_code == 200:
                        return True, "Ollama is running"
                except Exception:
                    pass
                
                # Fallback to the root endpoint (older versions)
                try:
                    response = await client.get("http://localhost:11434")
                    if response.status_code == 200 and "Ollama is running" in response.text:
                        return True, "Ollama is running (legacy version)"
                except Exception as e:
                    return False, f"Could not connect to Ollama: {str(e)}"
                
                return False, "Ollama is not responding as expected"
        except Exception as e:
            return False, f"Error checking Ollama status: {str(e)}"
    
    try:
        # First check if Ollama is running
        is_running, status_message = await get_ollama_models()
        if not is_running:
            print(f"Ollama check failed: {status_message}")
            raise HTTPException(
                status_code=503,
                detail=f"Ollama is not running or not accessible. {status_message}"
            )
            
        print(f"Ollama status: {status_message}")
        print(f"Sending request to Ollama with model: {DEFAULT_MODEL}")
        
        # Make the chat completion request
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(
                    OLLAMA_API_URL,
                    json={
                        "model": DEFAULT_MODEL,
                        "prompt": prompt,
                        "stream": True,
                        "options": {
                            "temperature": 0.7,
                            "top_p": 0.9,
                            "num_ctx": 4096  # Adjust based on your model's context window
                        }
                    },
                    timeout=300.0
                )
                
                response.raise_for_status()
                print(f"Received response from Ollama with status: {response.status_code}")
                
                async def generate():
                    try:
                        buffer = ""
                        async for chunk in response.aiter_text():
                            if not chunk:
                                continue
                                
                            # Process each line in the chunk
                            lines = (buffer + chunk).split("\n")
                            buffer = lines.pop()  # Save incomplete line for next chunk
                            
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    continue
                                    
                                # Ollama sends JSON objects, one per line
                                try:
                                    if line.startswith("data: "):
                                        line = line[6:]  # Remove 'data: ' prefix if present
                                    
                                    # Parse the JSON to validate it
                                    try:
                                        data = json.loads(line)
                                        if "response" in data:
                                            print(f"Received chunk: {data['response'][:50]}...")  # Log first 50 chars of response
                                            
                                            # Add assistant's response to history as we receive it
                                            if 'response' in data and data['response'].strip():
                                                if chat_sessions[session_id] and chat_sessions[session_id][-1]["role"] == "assistant":
                                                    # Update the last assistant message
                                                    chat_sessions[session_id][-1]["content"] += data['response']
                                                else:
                                                    # Add new assistant message
                                                    chat_sessions[session_id].append({"role": "assistant", "content": data['response']})
                                        else:
                                            print(f"Received non-response data: {data}")
                                    except json.JSONDecodeError as je:
                                        print(f"Invalid JSON received: {line}")
                                        continue
                                        
                                    yield f"data: {line}\n\n"
                                except Exception as e:
                                    print(f"Error processing line: {line}", str(e))
                                    continue
                        
                        # Process any remaining data in buffer
                        if buffer.strip():
                            try:
                                if buffer.startswith("data: "):
                                    buffer = buffer[6:]
                                yield f"data: {buffer}\n\n"
                            except Exception as e:
                                print(f"Error processing buffer: {buffer}", str(e))
                        
                        yield "data: [DONE]\n\n"
                        
                    except Exception as e:
                        error_msg = f"Error in generate(): {str(e)}"
                        print(error_msg)
                        yield f"data: {{\"error\": \"{error_msg}\"}}\n\n"
                        yield "data: [DONE]\n\n"
                
                return StreamingResponse(
                    generate(),
                    media_type="text/event-stream",
                    headers={
                        'Cache-Control': 'no-cache',
                        'Connection': 'keep-alive',
                        'X-Accel-Buffering': 'no',
                    }
                )
                
        except httpx.HTTPStatusError as e:
            error_msg = f"Ollama API error: {str(e)}"
            if e.response is not None:
                try:
                    error_detail = await e.response.text()
                    error_msg = f"{error_msg}\nResponse: {error_detail}"
                except:
                    pass
            print(error_msg)
            raise HTTPException(
                status_code=502,  # Bad Gateway
                detail=error_msg
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions as they are
        raise
        
    except Exception as e:
        error_msg = f"Unexpected error in chat_stream: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=error_msg
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
