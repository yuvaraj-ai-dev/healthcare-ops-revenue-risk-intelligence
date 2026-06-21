import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # For local running, binding to 127.0.0.1 is standard, but .env allows 0.0.0.0 for Docker
    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", 8000))
    print(f"Launching FastAPI Server at http://{host}:{port}...")
    print("API documentation is available at http://127.0.0.1:8000/docs")
    uvicorn.run("app.main:app", host=host, port=port, reload=True)
