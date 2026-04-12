"""
Vercel Serverless Entry Point
Exports the FastAPI app for the Vercel Python runtime.
"""
from app.api.main import app

# Vercel needs the app variable named as 'app' or explicitly configured.
# We also set VECTOR_STORE_TYPE=pinecone in the Vercel dashboard env vars.
