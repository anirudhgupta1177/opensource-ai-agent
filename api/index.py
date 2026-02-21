"""Vercel serverless entry point for FastAPI app."""
import sys
import os

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from app.main import app
except Exception as e:
    # Fallback: return error details as a simple ASGI app
    import traceback
    error_msg = f"Import error: {e}\n\n{traceback.format_exc()}"

    from fastapi import FastAPI
    app = FastAPI()

    @app.get("/{path:path}")
    @app.post("/{path:path}")
    async def error_handler(path: str = ""):
        return {"error": "App failed to load", "details": error_msg}
